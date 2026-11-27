package osp.jsettlers;

import java.io.File;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Timer;
import java.util.TimerTask;

import soc.game.SOCCity;
import soc.game.SOCDevCardConstants;
import soc.game.SOCGame;
import soc.game.SOCPlayer;
import soc.game.SOCPlayingPiece;
import soc.game.SOCRoad;
import soc.game.SOCSettlement;
import soc.game.SOCTradeOffer;
import soc.message.SOCMessage;
import soc.message.SOCTimingPing;
import soc.robot.SOCRobotBrain;
import soc.robot.SOCRobotClient;
import soc.util.CappedQueue;
import soc.util.SOCRobotParameters;

/**
 * The brain of a JSettlers robot whose decisions an Open Seat Agent makes (see ../../PROTOCOL.md). It runs inside
 * JSettlers' server (OSPServer) and reads the true game there; whenever the game waits for its seat, it asks the agent
 * with the state and the legal moves as OSPGame writes them (the true state for an open-hand agent, the seat's view
 * otherwise) and plays the answer through the robot client's requests, a protocol move that takes JSettlers two steps
 * (a bought piece and its place, a card and its resources, a robber move and its victim) one after the other. It doesn't
 * keep JSettlers' own robot logic or game tracking. An agent that observes gets every move of the game as a step from
 * the server's move log (MoveLog), from the game's start, before each decision; one that doesn't has to take its game
 * from each decision's state. The brain reads the game under the server's locks for it (OSPServer.read), so it never
 * sees a move half made. It speaks version 2: it serves view and truth (no worlds), steps, and lean decisions (the full
 * state again when the agent asks, and on its own after a change no step explains); the agent takes each game (ready)
 * or turns it down; a decision has the budget's time (120 s by default). Whatever fails (a game turned down, an illegal,
 * late or malformed answer, the agent's own failure, its process ending) is told to the agent where it still listens and
 * recorded in the seat's line of the game's result record (MoveLog), and the robot plays the first legal move for the
 * rest of the game (fallbacks). The agent's command comes from the system property osp.agent, its standard error goes to
 * the directory osp.log if set.
 */
public class OSPRobotBrain extends SOCRobotBrain
{
    /**
     * The rules JSettlers plays, as the protocol names them: the rulebook, narrower where JSettlers is (see Recorder), and
     * trades between players as JSettlers makes them (player_trading first_accept: answers in any order, the first accept
     * trades at once; no limit on offers), or none with the game option NT
     */
    static Map<String, Object> ruleset(SOCGame game)
    {
        boolean trading = ! game.isGameOptionSet("NT");
        Map<String, Object> ruleset = Json.object("preset", "standard", "player_trading", trading ? "first_accept" : "none", "bank_shortage", "unlimited",
            "bank_rates", "best_only", "robber_victims", "with_cards", "road_building", "forced");
        if (trading)
            ruleset.put("offer_limit", null);
        return ruleset;
    }

    // How long the seat's own offer waits for its answers before the agent takes a counteroffer or it is withdrawn (JSettlers'
    // robots wait 5 s for a robot's answer; ours count as robots), and after how long a trade request that made no step is
    // decided again (e.g. a counteroffer withdrawn while the seat took it)
    private static final long OFFER_WAIT_MILLIS = Long.getLong("osp.offerWait", 5_000), TRADE_RETRY_MILLIS = 3_000;
    // The seat's own offers per turn that its decisions name at most: JSettlers has no limit, but an agent that repeated
    // itself would hold the game up for good (the robots answer every offer)
    private static final int MAX_OFFERS_PER_TURN = 10;
    // How long after an offer closed the seat waits before it may offer again: JSettlers' robots answer after a pause, and a
    // rejection names no offer, so a late no to the last offer would count as a no to the next one (and the robot may then
    // still accept that one, a trade the protocol can't write)
    private static final long OFFER_SETTLE_MILLIS = 1_000;
    // The game's budget: how long the agent may think about one decision; the handshake and ready get a minute
    private static final long DECIDE_MILLIS = 120_000, HANDSHAKE_MILLIS = 60_000;
    // Any threads and a GPU allowed, as at every referee unless the game says otherwise
    static final Map<String, Object> BUDGET = Json.object("decide_ms", DECIDE_MILLIS, "game_ms", null, "threads", null, "gpu", true);

    private OSPGame osp;
    private AgentProcess agent;
    private boolean agentFailed, truth, observe, lean, after;
    // The move log's gaps at the last decision: a change no step explains sends a lean agent the full state
    private long gapsAtDecision;
    // The seat's line in the game's result record
    private MoveLog.SeatRecord record;
    // The steps the agent has been sent
    private int sent;
    private String agentName = "?";
    // Where the move log was (its moves and gaps) when the seat last moved: the next move waits until the log has gone on,
    // so the server took that one
    private long actedAt = -1;
    // The second step of a two-step move, and the game state its first step was sent in
    private List<Object> pending;
    // The brain's own once-a-second ping, and whether it has woken the brain yet (for the debug line)
    private Timer pings;
    private boolean pinged;
    private int pendingFrom;
    private int decisions, asked, fallbacks;
    private long thinkingMillis;
    // Trades between players: when the seat's own offer went out, the offer (as the protocol has it) the seat last made a
    // trade request on and when, and a trade move decided but not yet sent (it is sent at the next look if the offer it
    // answers still stands, else it expired and the seat's record counts it)
    private long offeredAt, tradeAt;
    private String tradeSnapshot;
    private List<Object> tradeMove;
    private String tradeFor;
    // The seat's offers this turn (counted from its roll), and when the seat last saw an offer open
    private int offersThisTurn;
    private long offerSeenAt;

    public OSPRobotBrain(SOCRobotClient rc, SOCRobotParameters params, SOCGame ga, CappedQueue<SOCMessage> mq)
    {
        super(rc, params, ga, mq);
    }

    /** Before the server removes the game: its end for the agent and the statistics line, while the game is still there */
    synchronized void gameEnding(String name)
    {
        if (osp != null && ourPlayerData != null && game != null && name.equals(game.getName()))
            finish(osp.seatOf(ourPlayerNumber));
    }

    /** At the server's shutdown: the game's end for the agent and the statistics line, if not yet written */
    synchronized void finishOnShutdown()
    {
        if (osp != null && ourPlayerData != null)
            finish(osp.seatOf(ourPlayerNumber));
    }

    @Override
    public void run()
    {
        OSPServer.BRAINS.add(this);
        try
        {
            Thread.currentThread().setName("ospBrain-" + client.getNickname() + "-" + game.getName());
        }
        catch (Throwable th) {}
        // JSettlers' brains get a TIMINGPING once a second from a pinger that SOCRobotBrain.run starts (its class isn't
        // public, and this run replaces that one): the brain pings itself, so it looks at the game again even when no game
        // message comes, e.g. after it read the game while the server changed it
        final CappedQueue<SOCMessage> queue = gameEventQ;
        final SOCTimingPing ping = new SOCTimingPing(game.getName());
        pings = new Timer("ospPing-" + client.getNickname() + "-" + game.getName(), true);
        pings.scheduleAtFixedRate(new TimerTask()
        {
            @Override
            public void run()
            {
                try
                {
                    queue.put(ping);
                }
                catch (Exception e)
                {
                    cancel();
                }
            }
        }, 1000, 1000);
        try
        {
            while (alive)
            {
                SOCMessage message = gameEventQ.get();
                if (message == null)
                    continue;
                if (message.getType() == SOCMessage.ROBOTDISMISS)
                {
                    client.leaveGame(game, "dismiss msg", false, false);
                    alive = false;
                    break;
                }
                if (message.getType() == SOCMessage.TIMINGPING && ! pinged)
                {
                    pinged = true;
                    debug("pinged");
                }
                try
                {
                    // Read outside the brain's lock: a server thread may wait for it (gameEnding) while holding the game's
                    Seen seen = look();
                    if (seen != null)
                        synchronized (this)
                        {
                            act(seen);
                        }
                }
                catch (Exception e)
                {
                    System.err.println(client.getNickname() + " in " + game.getName() + ": " + e);
                }
            }
        }
        catch (Exception e)
        {
            System.err.println(client.getNickname() + " in " + game.getName() + " stopped: " + e);
        }
        finally
        {
            synchronized (this)
            {
                if (osp != null && osp.isOver() && ourPlayerData != null)
                    finish(osp.seatOf(ourPlayerNumber));
                close();
            }
            pings.cancel();
            OSPServer.BRAINS.remove(this);
        }
    }

    /** What the brain read of the true game at one moment */
    private static final class Seen
    {
        int state, acting;
        boolean over, winnerNamed;
        // A free road of a Road Building card to place without a place for it (see OSPGame.strandedFreeRoad)
        boolean stranded;
        // The state the next move starts from (in the protocol's order of discards)
        Map<String, Object> position;
        List<Object> legal;
        // The game's moves so far, with its gaps (changes no move explains), and those the agent hasn't been sent
        long moves, progress, gaps;
        List<Map<String, Object>> steps;
        MoveLog log;
        // Trades between players: the open offer (as the protocol has it), the seat's answers while it is asked, whether the
        // offer is the seat's own and still waits for answers, and the seat's picks on it; whether the game trades at all
        Map<String, Object> offer;
        List<Object> answers, picks;
        boolean ourOffer, waiting, trading;
        String prompt;
        // The seat's resources as read (an offer or counteroffer of the agent's is checked against them)
        soc.game.SOCResourceSet hand;
    }

    // Reads the true game under the server's locks, or null if there's no game to play yet
    private Seen look()
    {
        final SOCGame truthGame = OSPServer.game(game.getName());
        if (truthGame == null || truthGame.getGameState() < SOCGame.START1A || ourPlayerData == null)
            return null;
        // The game's move log, begun here if the seat moves first (before the server handled any request)
        final MoveLog log = OSPServer.logOf(truthGame);
        return OSPServer.read(truthGame, log, () ->
        {
            // JSettlers empties a game it destroys
            if (truthGame.getPlayer(0) == null || truthGame.getBoard() == null)
                return null;
            // The move log's view of the game, so that both see the same answers to an open offer (OSPGame.offer)
            if (osp == null)
                osp = log != null ? log.osp : new OSPGame(truthGame);
            Seen seen = new Seen();
            seen.log = log;
            seen.over = osp.isOver();
            seen.winnerNamed = truthGame.getPlayerWithWin() != null;
            seen.state = truthGame.getGameState();
            seen.position = log != null ? log.state() : osp.state(-1);
            seen.acting = osp.acting();
            int seat = osp.seatOf(ourPlayerNumber);
            seen.legal = seen.acting == seat && ! seen.over ? osp.legal(seat) : null;
            seen.stranded = seen.acting == seat && ! seen.over && osp.strandedFreeRoad();
            seen.trading = ! truthGame.isGameOptionSet("NT");
            seen.prompt = osp.prompt();
            seen.offer = seen.over ? null : osp.offer();
            seen.answers = seen.offer == null ? new ArrayList<>() : osp.answerMoves(seat);
            seen.ourOffer = seen.offer != null && Json.integer(seen.offer.get("from")) == seat;
            boolean waiting = false;
            if (seen.ourOffer)
                for (int s = 0; s < truthGame.maxPlayers; s++)
                    waiting |= OSPGame.pending(seen.offer, s);
            seen.waiting = waiting;
            seen.picks = seen.ourOffer ? osp.pickMoves(seat) : null;
            seen.hand = new soc.game.SOCResourceSet(truthGame.getPlayer(osp.playerOf(seat)).getResources());
            seen.moves = log != null ? log.size() : 0;
            seen.progress = log != null ? log.size() + log.gaps() : 0;
            seen.gaps = log != null ? log.gaps() : 0;
            seen.steps = log != null ? log.stepsFrom(sent) : null;
            return seen;
        });
    }

    // Plays the seat's move if the true game waits for it
    private void act(Seen seen) throws Exception
    {
        int seat = osp.seatOf(ourPlayerNumber);
        if (seen.over)
        {
            // JSettlers ends the game a moment before it names the winner: the line waits for the name (the next ping)
            if (seen.winnerNamed)
                finish(seat);
            return;
        }
        int state = seen.state;
        int acting = seen.acting;
        List<Object> legal = seen.legal;

        // The second step of a two-step move, once the server asks for it; until the server took the first step the game
        // stays in the state the first step was sent in (messages from before it still come in)
        if (pending != null)
        {
            if (acting == seat && secondStep(pending, state))
            {
                // Until the server took the second step the move log stays where it was (the server waits for the piece,
                // say): no new move until then, which the server would take as a new first step
                pending = null;
                return;
            }
            if (state == pendingFrom)
                return;
            // The server went elsewhere (the move needed no second step, or it refused the first): choose anew
            pending = null;
        }
        // A free road without a place: given up, as the protocol's rules have it, instead of waiting for the server to end
        // the turn; no step follows, so it stays pending (and isn't sent again) until the server left the state
        if (acting == seat && seen.stranded)
        {
            pending = Json.array("give_up_free_road", null);
            pendingFrom = state;
            client.cancelBuildRequest(game, SOCPlayingPiece.ROAD);
            return;
        }
        if (seen.offer != null)
            offerSeenAt = System.currentTimeMillis();
        if (trade(seat, seen, state))
            return;
        if (acting != seat || seen.progress == actedAt || legal.isEmpty())
            return;
        // On its turn, with trades on and no offer open, the seat may offer (any cards, to any of the others), up to
        // MAX_OFFERS_PER_TURN a turn (counted from its roll)
        if ("roll".equals(seen.prompt))
            offersThisTurn = 0;
        Map<String, Object> trades = seen.trading && "main".equals(seen.prompt) && seen.offer == null && offersThisTurn < MAX_OFFERS_PER_TURN
            ? Json.object("offer", Json.object("to", others(seat), "left", null)) : null;
        // Right after an offer closed, late answers to it may still come: the decision (which may offer) waits a moment (the
        // next ping looks again)
        if (trades != null && System.currentTimeMillis() - offerSeenAt < OFFER_SETTLE_MILLIS)
            return;
        decisions++;
        if (seen.log != null)
            seen.log.seatRecord(seat).robotDecided();
        @SuppressWarnings("unchecked")
        List<Object> move = (List<Object>) (legal.size() == 1 && trades == null ? legal.get(0) : ask(seat, legal, seen, trades));
        actedAt = seen.progress;
        pendingFrom = state;
        if ("offer".equals(move.get(0)))
        {
            offeredAt = System.currentTimeMillis();
            offersThisTurn++;
        }
        firstStep(move, state);
    }

    // The other seats, in order
    private List<Object> others(int seat)
    {
        List<Object> others = new ArrayList<>();
        for (int s = 0; s < game.maxPlayers; s++)
            if (s != seat)
                others.add((long) s);
        return others;
    }

    // The offer's terms and addressees: what an answer or a pick is about
    private static String identity(Map<String, Object> offer)
    {
        return offer == null ? null : Json.write(Json.object("from", offer.get("from"), "give", offer.get("give"), "get", offer.get("get"), "to", offer.get("to")), true);
    }

    /**
     * Trades between players (player_trading first_accept): an offer to the seat is answered whenever it stands, in any
     * order (with a counteroffer if the agent makes one); the seat's own offer waits for its answers (OFFER_WAIT_MILLIS at
     * most), then the agent takes a counteroffer or the offer is withdrawn. A decided move is sent at the next look, if the
     * offer it is about still stands as it did (else the decision expired: JSettlers' other players don't wait). Returns
     * whether the seat is busy with a trade.
     */
    @SuppressWarnings("unchecked")
    private boolean trade(int seat, Seen seen, int state) throws Exception
    {
        // A decided move: sent if its offer still stands and the move is still one the seat may make, else dropped
        if (tradeMove != null)
        {
            List<Object> move = tradeMove;
            tradeMove = null;
            boolean answer = ! seen.ourOffer;
            boolean still = Objects.equals(identity(seen.offer), tradeFor)
                && (answer ? ! seen.answers.isEmpty() && (seen.answers.contains(move) || "counter".equals(move.get(0)))
                    : seen.picks != null && seen.picks.contains(move));
            if (! still)
            {
                if (record != null)
                    record.tradeExpired();
                debug("trade decision expired: " + Json.write(move));
                return seen.offer != null && (! seen.answers.isEmpty() || seen.ourOffer);
            }
            tradeSnapshot = Json.write(seen.offer, true);
            tradeAt = System.currentTimeMillis();
            tradeStep(move, seen.offer);
            return true;
        }
        if (seen.offer == null || (seen.answers.isEmpty() && ! seen.ourOffer))
            return false;
        long now = System.currentTimeMillis();
        // A request about the offer as it stands went out already: wait for the server (or, if it made no step, decide again
        // after a while)
        if (Json.write(seen.offer, true).equals(tradeSnapshot) && now - tradeAt < TRADE_RETRY_MILLIS)
            return true;
        List<Object> moves;
        Map<String, Object> trades = null;
        if (! seen.answers.isEmpty())
        {
            moves = seen.answers;
            trades = Json.object("counter", Json.object());
        }
        else
        {
            // The seat's own offer: the answers first
            if (seen.waiting && now - offeredAt < OFFER_WAIT_MILLIS)
                return true;
            moves = seen.picks;
        }
        decisions++;
        if (seen.log != null)
            seen.log.seatRecord(seat).robotDecided();
        tradeMove = (List<Object>) (moves.size() == 1 && trades == null ? moves.get(0) : ask(seat, moves, seen, trades));
        tradeFor = identity(seen.offer);
        // Sent at the next look, outside the brain's lock (which a server thread may wait for while it holds the game)
        gameEventQ.put(new SOCTimingPing(game.getName()));
        return true;
    }

    // Sends a trade move: an answer to the offer (accept, decline or a counteroffer, the seat's own offer to the offerer in
    // its own terms), or the offerer's pick (taking a counteroffer, which JSettlers makes an accept of the counterer's offer)
    // or withdrawal
    @SuppressWarnings("unchecked")
    private void tradeStep(List<Object> move, Map<String, Object> offer)
    {
        debug("trade: " + Json.write(move));
        String type = (String) move.get(0);
        Map<String, Object> value = move.size() > 1 ? (Map<String, Object>) move.get(1) : null;
        int offerer = osp.playerOf(Json.integer(offer.get("from")));
        switch (type)
        {
        case "respond":
            if (Boolean.TRUE.equals(value.get("accept")))
                client.acceptOffer(game, offerer);
            else
                client.rejectOffer(game);
            break;
        case "counter":
        {
            boolean[] to = new boolean[game.maxPlayers];
            to[offerer] = true;
            client.offerTrade(game, new SOCTradeOffer(game.getName(), ourPlayerNumber, to, OSPGame.resourceSet(value.get("get")), OSPGame.resourceSet(value.get("give"))));
            break;
        }
        case "confirm":
            client.acceptOffer(game, osp.playerOf(Json.integer(value.get("partner"))));
            break;
        case "cancel":
            // The offer and the counteroffers to it: those stand at JSettlers until their makers withdraw them, which they do
            // once their offer is turned down
            client.clearOffer(game);
            if (offer.get("counters") != null)
                client.rejectOffer(game);
            break;
        default:
            System.err.println(client.getNickname() + " in " + game.getName() + ": no JSettlers request for " + type);
        }
    }

    /** A failure of the agent's seat: its code (see PROTOCOL.md) and whether the agent reported it itself */
    private static final class AgentFailure extends Exception
    {
        final String code;
        final boolean byAgent;

        AgentFailure(String code, String message, boolean byAgent)
        {
            super(message);
            this.code = code;
            this.byAgent = byAgent;
        }
    }

    // Tells the agent what failed (if it still listens) and returns the failure
    private AgentFailure fail(Object n, String code, String message, Map<String, Object> detail)
    {
        try
        {
            Map<String, Object> failure = Json.object("type", "failure", "game_id", game.getName(), "n", n, "code", code, "message", message);
            if (detail != null)
                failure.put("detail", detail);
            agent.send(failure);
        }
        catch (Exception e) {}
        return new AgentFailure(code, message, false);
    }

    // The agent's own failure, as the exception for it
    private static AgentFailure reported(Map<String, Object> failure)
    {
        Object code = failure.get("code");
        return new AgentFailure(code instanceof String ? (String) code : "internal", "the agent reported a failure: " + failure.get("message"), true);
    }

    // Asks the agent; the first legal move once it failed
    private Object ask(int seat, List<Object> legal, Seen seen)
    {
        return ask(seat, legal, seen, null);
    }

    /**
     * Whether a trade the agent wrote is one the decision named, with valid terms: an offer to some of the seats named (or
     * a counteroffer, in the offerer's terms), giving at least one card and asking for at least one, of resources only and
     * none on both sides, with the seat holding what it gives
     */
    @SuppressWarnings("unchecked")
    private boolean namedTrade(Object answer, Map<String, Object> trades, soc.game.SOCResourceSet hand)
    {
        if (trades == null || ! (answer instanceof List) || ((List<Object>) answer).size() != 2 || ! (((List<Object>) answer).get(1) instanceof Map))
            return false;
        List<Object> move = (List<Object>) answer;
        Map<String, Object> terms = (Map<String, Object>) move.get(1);
        int[] give = cards(terms.get("give")), get = cards(terms.get("get"));
        if (give == null || get == null)
            return false;
        int gives = 0, gets = 0;
        for (int type = 0; type < give.length; type++)
        {
            if (give[type] > 0 && get[type] > 0)
                return false;
            gives += give[type];
            gets += get[type];
        }
        if (gives == 0 || gets == 0)
            return false;
        // What the seat gives: an offer's give, a counteroffer's get (it is written in the offerer's terms)
        boolean offer = "offer".equals(move.get(0)) && trades.containsKey("offer");
        boolean counter = "counter".equals(move.get(0)) && trades.containsKey("counter");
        if (! offer && ! counter)
            return false;
        if (! hand.contains(OSPGame.resourceSet(OSPGame.cardSet(offer ? give : get))))
            return false;
        if (counter)
            return terms.size() == 2;
        if (! (terms.get("to") instanceof List) || ((List<Object>) terms.get("to")).isEmpty() || terms.size() != 3)
            return false;
        List<Object> named = (List<Object>) ((Map<String, Object>) trades.get("offer")).get("to");
        List<Object> to = new ArrayList<>();
        for (Object s : (List<Object>) terms.get("to"))
        {
            if (! (s instanceof Long) || ! named.contains(s) || to.contains(s))
                return false;
            to.add(s);
        }
        return true;
    }

    // A card set of the protocol (resource names and positive counts) as counts by JSettlers' resource type, or null
    @SuppressWarnings("unchecked")
    private static int[] cards(Object set)
    {
        if (! (set instanceof Map))
            return null;
        int[] counts = new int[soc.game.SOCResourceConstants.WOOD + 1];
        for (Map.Entry<String, Object> card : ((Map<String, Object>) set).entrySet())
        {
            int type;
            try
            {
                type = OSPGame.resourceType(card.getKey());
            }
            catch (RuntimeException e)
            {
                return null;
            }
            if (! (card.getValue() instanceof Long) || (Long) card.getValue() <= 0 || type < 0 || type >= counts.length)
                return null;
            counts[type] += (int) (long) (Long) card.getValue();
        }
        return counts;
    }

    // Asks the agent, naming the trades it may propose in this decision (null for none); the first legal move once it failed
    private Object ask(int seat, List<Object> legal, Seen seen, Map<String, Object> trades)
    {
        asked++;
        long started = System.currentTimeMillis();
        Long n = seen.moves;
        try
        {
            if (agent == null && ! agentFailed)
                start(seat, seen);
            if (agent == null)
                return fallback(legal);
            sendSteps(seat, seen.steps);
            Map<String, Object> state = truth ? seen.position : OSPGame.hide(seen.position, seat);
            // A lean agent gets only the hash, unless a change no step explains came since its last decision
            boolean force = seen.gaps != gapsAtDecision;
            gapsAtDecision = seen.gaps;
            Map<String, Object> decide = Json.object("type", "decide", "game_id", game.getName(), "n", n, "seat", (long) seat, "hash", OSPGame.hash(state));
            if (! lean || force)
                decide.put("state", state);
            decide.put("legal", legal);
            if (trades != null)
                decide.put("trades", trades);
            agent.send(decide);
            while (true)
            {
                long left = DECIDE_MILLIS - (System.currentTimeMillis() - started);
                Map<String, Object> answer;
                try
                {
                    if (left <= 0)
                        throw new AgentProcess.Timeout("no time left");
                    answer = agent.receive(left);
                }
                catch (AgentProcess.Timeout e)
                {
                    throw fail(n, "timeout", "the agent gave no answer within " + DECIDE_MILLIS + " ms", null);
                }
                Object type = answer.get("type");
                if ("failure".equals(type))
                    throw reported(answer);
                if ("resync".equals(type) && n.equals(answer.get("n")))
                {
                    record.resynced();
                    Map<String, Object> full = new LinkedHashMap<>(decide);
                    full.put("state", state);
                    agent.send(full);
                    continue;
                }
                if (! "decide".equals(type) || ! answer.containsKey("action"))
                    throw fail(n, "malformed", "the agent answered a decision with " + Json.write(answer), null);
                if (! n.equals(answer.get("n")))
                    throw fail(n, "illegal", "the agent answered decision " + answer.get("n") + " when asked for " + n, null);
                String chosen = Json.write(answer.get("action"), true);
                for (Object move : legal)
                    if (Json.write(move, true).equals(chosen))
                    {
                        record.decided(System.currentTimeMillis() - started);
                        return move;
                    }
                // Or a trade the decision named, with terms of the agent's own
                if (namedTrade(answer.get("action"), trades, seen.hand))
                {
                    record.decided(System.currentTimeMillis() - started);
                    return answer.get("action");
                }
                throw fail(n, "illegal", "the agent answered " + chosen + ", which is neither one of the legal moves nor a trade with valid terms", null);
            }
        }
        catch (AgentFailure e)
        {
            return failed(n, e.code, e.byAgent, e.getMessage(), legal);
        }
        catch (Exception e)
        {
            // Its process ended (or broke off): nothing to tell it
            return failed(n, "ended", false, e.getMessage(), legal);
        }
        finally
        {
            thinkingMillis += System.currentTimeMillis() - started;
        }
    }

    // The agent failed: recorded, closed, and the robot plays on with the first legal move
    private Object failed(Long n, String code, boolean byAgent, String message, List<Object> legal)
    {
        System.err.println(client.getNickname() + " in " + game.getName() + ": the agent failed (" + code + "): " + message);
        if (record != null)
            record.failed(n, code, byAgent, message);
        agentFailed = true;
        close();
        return fallback(legal);
    }

    private Object fallback(List<Object> legal)
    {
        fallbacks++;
        if (record != null)
            record.fellBack();
        return legal.get(0);
    }

    // The moves the agent hasn't been sent, if it observes: as the referee knows them for an agent that needs the truth
    // (with the state before each move), else as the seat sees them
    @SuppressWarnings("unchecked")
    private void sendSteps(int seat, List<Map<String, Object>> steps) throws Exception
    {
        if (steps == null)
            return;
        for (Map<String, Object> step : steps)
        {
            sent++;
            if (! observe)
                continue;
            Map<String, Object> message = OSPGame.copy(step);
            if (! truth)
            {
                message.remove("state");
                int mover = Json.integer(message.get("seat"));
                List<Object> action = (List<Object>) message.get("action");
                Map<String, Object> result = (Map<String, Object>) message.get("result");
                if (action.get(0).equals("buy_development") && mover != seat)
                    result.put("card", null);
                if (action.get(0).equals("move_robber") && result != null && mover != seat
                    && Json.integer(((Map<String, Object>) action.get(1)).get("victim")) != seat)
                    result.put("stolen", "unknown");
            }
            agent.send(message);
        }
    }

    // Starts the agent and the game for it: the handshake (what it plays and wants), before, and its ready
    private void start(int seat, Seen seen) throws Exception
    {
        String command = System.getProperty("osp.agent");
        if (command == null)
            throw new IllegalStateException("No agent: set the system property osp.agent");
        String logs = System.getProperty("osp.log");
        File log = logs == null ? null : new File(logs, game.getName() + "-" + client.getNickname() + ".log");
        record = seen.log.seatRecord(seat);
        gapsAtDecision = seen.gaps;
        agent = new AgentProcess(command, log);
        agent.send(Json.object("type", "handshake", "protocol", "osp", "version", 2L));
        Map<String, Object> handshake = receiveOrTimeout(HANDSHAKE_MILLIS, null);
        if ("failure".equals(handshake.get("type")))
            throw reported(handshake);
        if (! "handshake".equals(handshake.get("type")) || ! "osp".equals(handshake.get("protocol")) || ! Long.valueOf(2).equals(handshake.get("version")))
            throw fail(null, "version", "the agent didn't answer the handshake in protocol osp version 2: " + Json.write(handshake), null);
        agentName = String.valueOf(handshake.get("name"));
        Object needs = handshake.containsKey("needs") ? handshake.get("needs") : "view";
        Object observing = handshake.containsKey("observe") ? handshake.get("observe") : false, leaning = handshake.containsKey("lean") ? handshake.get("lean") : false;
        Object rulesets = handshake.get("rulesets");
        record.agent(agentName, needs instanceof String ? (String) needs : null, Boolean.TRUE.equals(observing), Boolean.TRUE.equals(leaning));
        if (! (observing instanceof Boolean) || ! (leaning instanceof Boolean) || ! (rulesets == null || rulesets instanceof List))
            throw fail(null, "malformed", "the agent's handshake has a field of the wrong type: " + Json.write(handshake), null);
        // What this referee can't serve, it refuses instead of sending something else
        if (! "view".equals(needs) && ! "truth".equals(needs))
            throw fail(null, "unsupported", "this robot serves view and truth, not " + Json.write(needs) + ": it draws no worlds", null);
        if (rulesets == null || ! ((List<?>) rulesets).contains("standard"))
            throw fail(null, "unsupported", "JSettlers plays the standard preset with options, the agent plays " + Json.write(rulesets), Json.object("preset", "standard"));
        truth = "truth".equals(needs);
        lean = (Boolean) leaning;
        if (lean && ! (Boolean) observing)
            throw fail(null, "unsupported", "a lean agent has to observe: it keeps its own game from the steps", null);
        // An observing agent follows the game from its start; one that doesn't starts from now
        observe = (Boolean) observing && seen.log != null;
        Map<String, Object> from = observe ? seen.log.start() : seen.position;
        if (! observe)
            sent = (int) seen.moves;
        agent.send(Json.object("type", "before", "game_id", game.getName(), "seat", (long) seat, "players", (long) game.maxPlayers,
            "ruleset", ruleset(osp.game), "budget", BUDGET, "seed", (long) seen.log.seed, "board", osp.board.boardJson(), "state", truth ? from : OSPGame.hide(from, seat)));
        // The agent takes the game or turns it down
        Map<String, Object> ready = receiveOrTimeout(HANDSHAKE_MILLIS, null);
        if ("failure".equals(ready.get("type")))
            throw reported(ready);
        if (! "ready".equals(ready.get("type")) || ! game.getName().equals(ready.get("game_id")))
            throw fail(null, "malformed", "the agent answered before with " + Json.write(ready), null);
    }

    // The agent's next message within the time, or a timeout failure
    private Map<String, Object> receiveOrTimeout(long millis, Object n) throws Exception
    {
        try
        {
            return agent.receive(millis);
        }
        catch (AgentProcess.Timeout e)
        {
            throw fail(n, "timeout", "the agent gave no answer within " + millis + " ms", null);
        }
    }

    private static final boolean DEBUG = System.getProperty("osp.debug") != null;

    private void debug(String what)
    {
        if (DEBUG)
            System.err.println("[osp " + client.getNickname() + " " + game.getName() + "] " + what);
    }

    // Plays a protocol move, or its first step
    @SuppressWarnings("unchecked")
    private void firstStep(List<Object> move, int state)
    {
        debug("state " + state + ": " + Json.write(move));
        String type = (String) move.get(0);
        Object value = move.size() > 1 ? move.get(1) : null;
        boolean placingFree = state < SOCGame.ROLL_OR_CARD || state == SOCGame.PLACING_FREE_ROAD1 || state == SOCGame.PLACING_FREE_ROAD2;
        switch (type)
        {
        case "roll": client.rollDice(game); break;
        case "end_turn": client.endTurn(game); break;
        case "build_settlement":
            if (placingFree)
                client.putPiece(game, new SOCSettlement(ourPlayerData, osp.board.node(value), game.getBoard()));
            else
            {
                pending = move;
                client.buildRequest(game, SOCPlayingPiece.SETTLEMENT);
            }
            break;
        case "build_road":
            if (placingFree)
                client.putPiece(game, new SOCRoad(ourPlayerData, osp.board.edge(value), game.getBoard()));
            else
            {
                pending = move;
                client.buildRequest(game, SOCPlayingPiece.ROAD);
            }
            break;
        case "build_city":
            pending = move;
            client.buildRequest(game, SOCPlayingPiece.CITY);
            break;
        case "buy_development": client.buyDevCard(game); break;
        case "play_knight": client.playDevCard(game, SOCDevCardConstants.KNIGHT); break;
        case "play_road_building": client.playDevCard(game, SOCDevCardConstants.ROADS); break;
        case "play_year_of_plenty":
            pending = move;
            client.playDevCard(game, SOCDevCardConstants.DISC);
            break;
        case "play_monopoly":
            pending = move;
            client.playDevCard(game, SOCDevCardConstants.MONO);
            break;
        case "move_robber":
        {
            Map<String, Object> robbery = (Map<String, Object>) value;
            if (robbery.get("victim") != null)
                pending = move;
            client.moveRobber(game, ourPlayerData, osp.board.hexCoord((List<?>) robbery.get("hex")));
            break;
        }
        case "discard": client.discard(game, OSPGame.resourceSet(value)); break;
        case "offer":
        {
            Map<String, Object> offer = (Map<String, Object>) value;
            boolean[] to = new boolean[game.maxPlayers];
            for (Object s : (List<Object>) offer.get("to"))
                to[osp.playerOf(Json.integer(s))] = true;
            client.offerTrade(game, new SOCTradeOffer(game.getName(), ourPlayerNumber, to, OSPGame.resourceSet(offer.get("give")), OSPGame.resourceSet(offer.get("get"))));
            break;
        }
        case "bank_trade":
        {
            Map<String, Object> trade = (Map<String, Object>) value;
            client.bankTrade(game, OSPGame.resourceSet(trade.get("give")), OSPGame.resourceSet(trade.get("get")));
            break;
        }
        default:
            System.err.println(client.getNickname() + " in " + game.getName() + ": no JSettlers request for " + type);
        }
    }

    // Plays the second step of a pending move if the server waits for it; returns whether it did
    @SuppressWarnings("unchecked")
    private boolean secondStep(List<Object> move, int state)
    {
        debug("state " + state + ", pending " + Json.write(move));
        String type = (String) move.get(0);
        Object value = move.get(1);
        switch (type)
        {
        case "build_settlement":
            if (state != SOCGame.PLACING_SETTLEMENT) return false;
            client.putPiece(game, new SOCSettlement(ourPlayerData, osp.board.node(value), game.getBoard()));
            return true;
        case "build_road":
            if (state != SOCGame.PLACING_ROAD) return false;
            client.putPiece(game, new SOCRoad(ourPlayerData, osp.board.edge(value), game.getBoard()));
            return true;
        case "build_city":
            if (state != SOCGame.PLACING_CITY) return false;
            client.putPiece(game, new SOCCity(ourPlayerData, osp.board.node(value), game.getBoard()));
            return true;
        case "play_year_of_plenty":
        {
            if (state != SOCGame.WAITING_FOR_DISCOVERY) return false;
            int[] counts = new int[6];
            for (Object name : (List<Object>) value)
                counts[OSPGame.resourceType((String) name)]++;
            client.pickResources(game, OSPGame.resourceSet(OSPGame.cardSet(counts)));
            return true;
        }
        case "play_monopoly":
            if (state != SOCGame.WAITING_FOR_MONOPOLY) return false;
            client.pickResourceType(game, OSPGame.resourceType((String) value));
            return true;
        case "move_robber":
            if (state != SOCGame.WAITING_FOR_ROB_CHOOSE_PLAYER) return false;
            client.choosePlayer(game, osp.playerOf(Json.integer(((Map<String, Object>) value).get("victim"))));
            return true;
        default:
            return false;
        }
    }

    // Tells the agent the game's end and writes the statistics line
    private void finish(int seat)
    {
        if (after)
            return;
        after = true;
        SOCGame truthGame = osp.game;
        List<Object> vp = Json.array();
        for (int s = 0; s < truthGame.maxPlayers; s++)
            vp.add(truthGame.getPlayer(osp.playerOf(s)) == null ? null : (Object) (long) truthGame.getPlayer(osp.playerOf(s)).getTotalVP());
        // The winner, or (ended by the server before it named one) the player on turn with the points to win
        SOCPlayer won = truthGame.getPlayerWithWin();
        if (won == null && osp.isOver() && truthGame.getPlayer(truthGame.getCurrentPlayerNumber()).getTotalVP() >= truthGame.vp_winner)
            won = truthGame.getPlayer(truthGame.getCurrentPlayerNumber());
        Object winner = won == null ? null : (Object) (long) osp.seatOf(won.getPlayerNumber());
        try
        {
            if (agent != null)
            {
                // The moves since the last decision, up to the end (read without the log's lock, which the server's thread
                // may hold while it waits for this brain)
                MoveLog log = OSPServer.log(game.getName());
                if (log != null)
                    sendSteps(seat, log.stepsFrom(sent));
                Map<String, Object> end = Json.object("type", "after", "game_id", game.getName(), "winner", winner, "vp", vp, "truncated", winner == null);
                if (log != null && log.result() != null)
                    end.put("result", log.result());
                agent.send(end);
            }
        }
        catch (Exception e) {}
        close();
    }

    private void close()
    {
        if (agent != null)
        {
            agent.close();
            agent = null;
        }
    }
}
