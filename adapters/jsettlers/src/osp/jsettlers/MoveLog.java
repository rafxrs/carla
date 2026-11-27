package osp.jsettlers;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.lang.management.ManagementFactory;
import java.lang.management.ThreadMXBean;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;
import java.util.concurrent.CopyOnWriteArrayList;

import soc.game.SOCDevCardConstants;
import soc.game.SOCGame;
import soc.game.SOCPlayer;
import soc.game.SOCPlayingPiece;
import soc.message.SOCAcceptOffer;
import soc.message.SOCBankTrade;
import soc.message.SOCChoosePlayer;
import soc.message.SOCDiscard;
import soc.message.SOCMessage;
import soc.message.SOCMessageForGame;
import soc.message.SOCMoveRobber;
import soc.message.SOCPickResourceType;
import soc.message.SOCPickResources;
import soc.message.SOCPlayDevCardRequest;
import soc.message.SOCPutPiece;
import soc.server.GameMessageHandler;
import soc.server.genericServer.Connection;

/**
 * Every move of a JSettlers game on this process's server as Open Seat Protocol steps (see ../../PROTOCOL.md), with its
 * result as the referee knows it and the true state before it. JSettlers' server hands each player's request to its
 * message handler on one thread; OSPServer's handler passes it through dispatch here, which reads the game before and
 * after JSettlers handles it and turns the change into a step: one protocol move where JSettlers takes two requests (a
 * bought piece and its place, a card and its resources, a robber move and its victim), and the discards after a 7, which
 * JSettlers' players make at the same time, one after another from the lowest seat, each with the state the protocol's
 * order gives it. A change no request made (the server ending a slow player's turn, say) counts as a gap: the next step
 * then starts from another state than the last one ended in.
 *
 * The log's monitor is the lock for reading the game: a robot's brain that reads under it (OSPServer.read) sees no move
 * half handled. With the system property osp.record set, each game is also written there as a referee log while it
 * runs (GAME.jsonl, or GAME-2.jsonl and so on if an earlier server run wrote that game's name: before, a decide and a step
 * per move, after with the result record), the input of Lanzenbrecher's replay harness and relay. At its end each game writes
 * its result record (see PROTOCOL.md) as one osp-result line to standard error: the players by seat, the winner, and per
 * seat the agent and its decisions, their time, resyncs, fallbacks and failures (filled in by the seat's brain), its
 * moves, those with a choice (only for a recorded game: the legal moves are listed only for its decide lines, to spare
 * the server's message thread) and the CPU time of its robot's brain thread in this game.
 */
final class MoveLog
{
    final String gameId;
    final SOCGame game;
    final OSPGame osp;

    // The steps so far: read without the log's lock by a brain finishing its game (which may hold its own lock)
    private final List<Map<String, Object>> steps = new CopyOnWriteArrayList<>();
    // The state before the first move, and after the last request handled
    private final Map<String, Object> start;
    private Map<String, Object> last;
    // The first request of a two-step move: the move's kind and value so far, the state before it, its legal moves
    private String pendingKind;
    private Object pendingValue;
    private Map<String, Object> pendingBefore;
    private List<Object> pendingLegal;
    // After a 7: the state before the first discard, the seats that have to discard (lowest first), the discards come in
    // (seat: card set and legal moves) and how many of the seats are written as steps
    private Map<String, Object> discardStart;
    private List<Integer> discarders;
    private final Map<Integer, Object[]> discarded = new TreeMap<>();
    private int discardsDone;
    private int gaps;
    private boolean ended;
    private final int[] moves, choices;
    // Each seat's line of the result record, and the seed every agent of the game gets; the record once the game ended
    private final SeatRecord[] records;
    // A fresh seed per game: JSettlers' game draws its first player, dice and development cards from an unseeded Random of
    // its own, so a game doesn't repeat, and the seed fixes only the agents' side
    final int seed = new java.util.Random().nextInt(Integer.MAX_VALUE);
    private int turns;
    private volatile Map<String, Object> result;
    // Whether the game is recorded (osp.record), and its record while it is open
    private final boolean recording;
    private Writer record;

    MoveLog(SOCGame game, Map<String, Object> start)
    {
        this.game = game;
        this.gameId = game.getName();
        this.osp = new OSPGame(game);
        this.start = start;
        this.last = start;
        this.moves = new int[game.maxPlayers];
        this.choices = new int[game.maxPlayers];
        this.records = new SeatRecord[game.maxPlayers];
        for (int seat = 0; seat < game.maxPlayers; seat++)
            records[seat] = new SeatRecord();
        String dir = System.getProperty("osp.record");
        recording = dir != null;
        if (dir != null)
        {
            try
            {
                new File(dir).mkdirs();
                // JSettlers numbers its games afresh in each run: a name an earlier run recorded gets a number
                String name = gameId.replaceAll("[^A-Za-z0-9._-]", "_");
                File file = new File(dir, name + ".jsonl");
                for (int copy = 2; ! file.createNewFile(); copy++)
                    file = new File(dir, name + "-" + copy + ".jsonl");
                record = new OutputStreamWriter(new FileOutputStream(file), StandardCharsets.UTF_8);
                write(Json.object("type", "before", "game_id", gameId, "players", (long) game.maxPlayers, "jsettlers_first", (long) osp.first,
                    "ruleset", OSPRobotBrain.ruleset(game), "seed", (long) seed, "board", osp.board.boardJson(), "state", start));
            }
            catch (IOException e)
            {
                System.err.println("osp: can't record " + gameId + ": " + e);
                record = null;
            }
        }
    }

    /**
     * A seat's line in the game's result record (see PROTOCOL.md): the brain of an Open Seat Agent's robot fills it in
     * (the agent, what it received, its decisions and their time, resyncs, fallbacks, failures, and the robot's own
     * decisions); JSettlers' own robots leave it empty
     */
    static final class SeatRecord
    {
        private String agent, needs;
        private Boolean observe, lean;
        private int decisions, resyncs, fallbacks, robotDecisions, expiredTrades;
        private long timeMs, maxMs;
        private final List<Object> failures = new ArrayList<>();

        synchronized void agent(String name, String needs, boolean observe, boolean lean)
        {
            this.agent = name;
            this.needs = needs;
            this.observe = observe;
            this.lean = lean;
        }

        synchronized void decided(long ms)
        {
            decisions++;
            timeMs += ms;
            maxMs = Math.max(maxMs, ms);
        }

        synchronized void resynced()
        {
            resyncs++;
        }

        synchronized void fellBack()
        {
            fallbacks++;
        }

        // A trade decision of the agent whose offer was taken or withdrawn before its answer went out (dropped, not sent)
        synchronized void tradeExpired()
        {
            expiredTrades++;
        }

        synchronized void robotDecided()
        {
            robotDecisions++;
        }

        synchronized void failed(Long n, String code, boolean byAgent, String message)
        {
            failures.add(Json.object("n", n, "code", code, "by", byAgent ? "agent" : "referee", "message", message));
        }

        synchronized Map<String, Object> json(int seat)
        {
            return Json.object("seat", (long) seat, "agent", agent, "needs", needs, "observe", observe, "lean", lean,
                "decisions", (long) decisions, "time_ms", timeMs, "max_ms", maxMs, "resyncs", (long) resyncs, "fallbacks", (long) fallbacks,
                "failures", new ArrayList<>(failures), "robot_decisions", (long) robotDecisions,
                "expired_trades", (long) expiredTrades);
        }
    }

    /** A seat's line in the result record */
    SeatRecord seatRecord(int seat)
    {
        return records[seat];
    }

    /** The game's result record once it ended with a winner, else null */
    Map<String, Object> result()
    {
        return result;
    }

    /** The state before the game's first move */
    Map<String, Object> start()
    {
        return start;
    }

    /** How many moves the game has had so far (the next one's n) */
    int size()
    {
        return steps.size();
    }

    /** The steps from the index on */
    List<Map<String, Object>> stepsFrom(int index)
    {
        Object[] all = steps.toArray();
        List<Map<String, Object>> from = new ArrayList<>();
        for (int i = index; i < all.length; i++)
        {
            @SuppressWarnings("unchecked")
            Map<String, Object> step = (Map<String, Object>) all[i];
            from.add(step);
        }
        return from;
    }

    int gaps()
    {
        return gaps;
    }

    /**
     * The state the next move starts from, in the protocol's order: the true state, or while discards go on, the state
     * before the first discard with the discards before the next seat's done (call under the log's lock and the game's
     * monitor)
     */
    Map<String, Object> state()
    {
        if (discardStart != null)
            return discardState(discardsDone);
        return osp.state(-1);
    }

    /**
     * JSettlers' handling of one request, with the step it makes: the game is read before and after (under its
     * monitor, which JSettlers' handler takes itself in between). A robot-only game's last move ends inside it: JSettlers
     * destroys the game before the handler returns, and OSPServer lets the log read the end first (gameDestroyed).
     */
    static boolean dispatch(GameMessageHandler inner, SOCGame game, SOCMessageForGame message, Connection connection) throws Exception
    {
        MoveLog log = game.getGameState() >= SOCGame.START1A ? OSPServer.logOf(game) : null;
        if (log == null)
            return inner.dispatch(game, message, connection);
        SOCPlayer player = connection == null ? null : game.getPlayer(connection.getData());
        return dispatchAs(log, inner, game, message, connection, player == null ? -1 : log.osp.seatOf(player.getPlayerNumber()));
    }

    /** JSettlers' handling of a request from a seat (-1: from no player), with the step it makes */
    static boolean dispatchAs(MoveLog log, GameMessageHandler inner, SOCGame game, SOCMessageForGame message, Connection connection, int seat) throws Exception
    {
        synchronized (log)
        {
            if (log.ended)
                return inner.dispatch(game, message, connection);
            game.takeMonitor();
            try
            {
                Map<String, Object> before = log.osp.state(-1);
                int acting = log.osp.acting();
                // Listed only for the record's decide lines (and its choices), not on every request of the message thread: the
                // sending seat's moves (with an offer open, a seat that isn't the one to act in seat order can answer or pick)
                Map<String, Object> offer = (Map<String, Object>) before.get("offer");
                List<Object> legal = ! log.recording ? null
                    : message.getType() == SOCMessage.DISCARD && seat >= 0 ? log.osp.discardMoves(seat)
                    : offer != null && seat >= 0 && seat != acting ? tradeMoves(log.osp, seat)
                    : acting >= 0 ? log.osp.legal(acting) : new ArrayList<>();
                // The offerer's picks, for a withdrawal that another seat's request makes (the last counteroffer going)
                int offerer = offer == null ? -1 : Json.integer(offer.get("from"));
                List<Object> offererLegal = ! log.recording || offer == null ? null : offerer == acting ? log.osp.legal(acting) : log.osp.pickMoves(offerer);
                List<Integer> mustDiscard = new ArrayList<>();
                for (int s = 0; s < game.maxPlayers; s++)
                    if (game.getPlayer(log.osp.playerOf(s)).getNeedToDiscard())
                        mustDiscard.add(s);
                log.inFlight = new InFlight(message, seat, before, legal, offererLegal, mustDiscard);
            }
            finally
            {
                game.releaseMonitor();
            }
            try
            {
                return inner.dispatch(game, message, connection);
            }
            finally
            {
                if (log.inFlight != null && log.alive())
                {
                    game.takeMonitor();
                    try
                    {
                        log.afterRequest();
                    }
                    finally
                    {
                        game.releaseMonitor();
                    }
                }
                log.inFlight = null;
            }
        }
    }

    // Whether a request is one of a trade between players
    private static boolean isTrade(SOCMessageForGame message)
    {
        int type = message.getType();
        return type == SOCMessage.MAKEOFFER || type == SOCMessage.REJECTOFFER || type == SOCMessage.ACCEPTOFFER || type == SOCMessage.CLEAROFFER;
    }

    // The trade moves of a seat that isn't the one to act in seat order: its answers while it is asked, else the offerer's picks
    private static List<Object> tradeMoves(OSPGame osp, int seat)
    {
        List<Object> answers = osp.answerMoves(seat);
        return ! answers.isEmpty() ? answers : osp.pickMoves(seat);
    }

    /** A request JSettlers is handling: who sent it, and the game before it */
    private static final class InFlight
    {
        final SOCMessageForGame message;
        final int seat;
        final Map<String, Object> before;
        final List<Object> legal;
        final List<Object> offererLegal;
        final List<Integer> mustDiscard;

        InFlight(SOCMessageForGame message, int seat, Map<String, Object> before, List<Object> legal, List<Object> offererLegal, List<Integer> mustDiscard)
        {
            this.message = message;
            this.seat = seat;
            this.before = before;
            this.legal = legal;
            this.offererLegal = offererLegal;
            this.mustDiscard = mustDiscard;
        }
    }

    private InFlight inFlight;
    // Whether the request just handled withdrew an open offer that JSettlers still holds (see handled)
    private boolean closedAfter;

    // Whether JSettlers still has the game (it empties a destroyed one)
    private boolean alive()
    {
        return game.getPlayer(0) != null && game.getBoard() != null;
    }

    // The request handled: its step, and the game's end if it ended (a failure here never stops JSettlers' handling: it
    // counts as a gap)
    private void afterRequest()
    {
        InFlight request = inFlight;
        inFlight = null;
        try
        {
            Map<String, Object> after = osp.state(-1);
            // Something changed the game since the last request, which no request made (the server ended a slow player's
            // turn): a gap, which loses what was pending, and a turn if the player on turn changed; found at any request,
            // also one that changes nothing
            if (! request.before.equals(last))
            {
                gaps++;
                pendingKind = null;
                discardStart = null;
                discarded.clear();
                if (! setup(last) && ! Objects.equals(last.get("on_turn"), request.before.get("on_turn")))
                    turns++;
            }
            closedAfter = false;
            if (request.seat >= 0)
                handled(request.message, request.seat, request.before, after, request.legal, request.offererLegal, request.mustDiscard);
            // A withdrawn offer JSettlers still holds is closed from now on: the next request starts from the state without it
            last = closedAfter ? osp.state(-1) : after;
            if (osp.isOver() && game.getPlayerWithWin() != null)
                end(after);
        }
        catch (RuntimeException e)
        {
            gaps++;
            System.err.println("osp: move log of " + gameId + " failed: " + e);
            e.printStackTrace();
        }
    }

    /**
     * JSettlers is destroying the game (OSPServer). On its message thread, while it still handles the game's last request
     * (and holds the log's lock for it), the last move and the end are read first, then the record is closed. Another
     * thread (the server ending a slow player's turn, which may end the game) holds the game's monitor: it mustn't wait for
     * the log's lock, which a brain may hold while it waits for that monitor (OSPServer.read), so the record is closed
     * beside it.
     */
    void gameDestroyed()
    {
        if (Thread.holdsLock(this))
        {
            if (inFlight != null && alive())
                afterRequest();
            close();
            return;
        }
        Thread closer = new Thread(this::close, "ospClose-" + gameId);
        closer.setDaemon(true);
        closer.start();
    }

    // Whether a state is in the setup, whose placements aren't player turns
    private static boolean setup(Map<String, Object> state)
    {
        Object prompt = state.get("prompt");
        return "setup_settlement".equals(prompt) || "setup_road".equals(prompt);
    }

    // Turns a handled request into its step, if it made one
    private void handled(SOCMessageForGame message, int seat, Map<String, Object> before, Map<String, Object> after, List<Object> legal,
        List<Object> offererLegal, List<Integer> mustDiscard)
    {
        // The states are maps, lists and values as OSPGame writes them, so they compare as their JSON would; a Monopoly that
        // takes nothing changes nothing after its card was played, and its step still comes
        boolean monopolyPick = message.getType() == SOCMessage.PICKRESOURCETYPE && "play_monopoly".equals(pendingKind);
        if (before.equals(after) && ! monopolyPick)
            return;
        if (isTrade(message))
        {
            traded(message, seat, before, after, legal, offererLegal);
            return;
        }
        // JSettlers lets the player on turn build, trade with the bank, play a card or end the turn with its offer standing;
        // the protocol's offer has to close first: the move withdraws it, and the offer stays closed though JSettlers still
        // holds it (its robots sometimes go on so; ours withdraws first)
        if (before.get("offer") != null && seat == Json.integer(before.get("on_turn")))
        {
            step(seat, Json.array("cancel"), null, before, legal);
            before = withoutOffer(before);
            osp.closeOffer();
            closedAfter = true;
            // The legal moves were the withdrawal's; those of the move after it, with the offer closed, weren't listed
            legal = null;
        }
        switch (message.getType())
        {
        case SOCMessage.ROLLDICE:
        {
            int total = game.getCurrentDice();
            int a = Math.min(6, total - 1);
            step(seat, Json.array("roll"), Json.object("dice", Json.array((long) a, (long) (total - a))), before, legal);
            break;
        }
        case SOCMessage.PUTPIECE:
        {
            SOCPutPiece put = (SOCPutPiece) message;
            String kind = put.getPieceType() == SOCPlayingPiece.ROAD ? "build_road" : put.getPieceType() == SOCPlayingPiece.SETTLEMENT ? "build_settlement" : "build_city";
            Object place = kind.equals("build_road") ? osp.board.path(put.getCoordinates()) : osp.board.intersection(put.getCoordinates());
            if (pendingKind != null && pendingKind.equals(kind))
                step(seat, Json.array(kind, place), null, pendingBefore, pendingLegal);
            else
                step(seat, Json.array(kind, place), null, before, legal);
            break;
        }
        case SOCMessage.BUILDREQUEST:
        {
            int type = ((soc.message.SOCBuildRequest) message).getPieceType();
            pending(type == SOCPlayingPiece.ROAD ? "build_road" : type == SOCPlayingPiece.SETTLEMENT ? "build_settlement" : "build_city", null, before, legal);
            break;
        }
        case SOCMessage.CANCELBUILDREQUEST:
            pendingKind = null;
            break;
        case SOCMessage.BUYDEVCARDREQUEST:
        {
            String card = null;
            Map<String, Object> had = development(before, seat), has = development(after, seat);
            for (String name : has.keySet())
                if (Json.integer(has.get(name)) > Json.integer(had.get(name)))
                    card = name;
            step(seat, Json.array("buy_development"), Json.object("card", card), before, legal);
            break;
        }
        case SOCMessage.PLAYDEVCARDREQUEST:
            switch (((SOCPlayDevCardRequest) message).getDevCard())
            {
            case SOCDevCardConstants.KNIGHT: step(seat, Json.array("play_knight"), null, before, legal); break;
            case SOCDevCardConstants.ROADS: step(seat, Json.array("play_road_building"), null, before, legal); break;
            case SOCDevCardConstants.DISC: pending("play_year_of_plenty", null, before, legal); break;
            case SOCDevCardConstants.MONO: pending("play_monopoly", null, before, legal); break;
            default: gaps++;
            }
            break;
        case SOCMessage.PICKRESOURCES:
            if ("play_year_of_plenty".equals(pendingKind))
            {
                List<Object> names = new ArrayList<>();
                for (Map.Entry<String, Object> picked : OSPGame.cardSet(((SOCPickResources) message).getResources()).entrySet())
                    for (int i = 0; i < Json.integer(picked.getValue()); i++)
                        names.add(picked.getKey());
                names.sort(null);
                step(seat, Json.array("play_year_of_plenty", names), null, pendingBefore, pendingLegal);
            }
            break;
        case SOCMessage.PICKRESOURCETYPE:
            if ("play_monopoly".equals(pendingKind))
            {
                String resource = OSPGame.resourceName(((SOCPickResourceType) message).getResourceType());
                Map<String, Object> taken = new LinkedHashMap<>();
                for (int other = 0; other < game.maxPlayers; other++)
                    if (other != seat)
                        taken.put(Integer.toString(other), (long) (Json.integer(hand(pendingBefore, other).get(resource)) - Json.integer(hand(after, other).get(resource))));
                step(seat, Json.array("play_monopoly", resource), Json.object("taken", taken), pendingBefore, pendingLegal);
            }
            break;
        case SOCMessage.MOVEROBBER:
        {
            List<Object> hex = osp.board.hex(((SOCMoveRobber) message).getCoordinates());
            if (game.getGameState() == SOCGame.WAITING_FOR_ROB_CHOOSE_PLAYER)
                pending("move_robber", hex, before, legal);
            else
                robbery(seat, hex, before, after, legal);
            break;
        }
        case SOCMessage.CHOOSEPLAYER:
            if ("move_robber".equals(pendingKind))
                robbery(seat, pendingValue, pendingBefore, after, pendingLegal);
            break;
        case SOCMessage.DISCARD:
        {
            if (discardStart == null)
            {
                discardStart = before;
                discarders = mustDiscard;
                discardsDone = 0;
                discarded.clear();
            }
            discarded.put(seat, new Object[] {OSPGame.cardSet(((SOCDiscard) message).getResources()), legal});
            // In the protocol's order, as far as the seats before have discarded
            while (discardsDone < discarders.size() && discarded.containsKey(discarders.get(discardsDone)))
            {
                int next = discarders.get(discardsDone);
                Object[] discard = discarded.get(next);
                @SuppressWarnings("unchecked")
                List<Object> moves = (List<Object>) discard[1];
                step(next, Json.array("discard", discard[0]), null, discardState(discardsDone), moves);
                discardsDone++;
            }
            if (discardsDone == discarders.size())
            {
                discardStart = null;
                discarded.clear();
            }
            break;
        }
        case SOCMessage.BANKTRADE:
        {
            SOCBankTrade trade = (SOCBankTrade) message;
            step(seat, Json.array("bank_trade", Json.object("give", OSPGame.cardSet(trade.getGiveSet()), "get", OSPGame.cardSet(trade.getGetSet()))), null, before, legal);
            break;
        }
        case SOCMessage.ENDTURN:
            step(seat, Json.array("end_turn"), null, before, legal);
            break;
        default:
            // A request no protocol move makes changed the game
            gaps++;
        }
    }

    /**
     * A trade request's steps under player_trading "first_accept", by what changed in the protocol's offer: the player on
     * turn offers (withdrawing a standing offer first), an addressee declines, counters (its own offer to the player on
     * turn) or accepts (the trade happens at once), the player on turn takes a counteroffer or withdraws. A change none of
     * these explains is a gap.
     */
    @SuppressWarnings("unchecked")
    private void traded(SOCMessageForGame message, int seat, Map<String, Object> before, Map<String, Object> after, List<Object> legal,
        List<Object> offererLegal)
    {
        Map<String, Object> was = (Map<String, Object>) before.get("offer"), is = (Map<String, Object>) after.get("offer");
        int onTurn = Json.integer(before.get("on_turn"));
        String key = Integer.toString(seat);
        switch (message.getType())
        {
        case SOCMessage.MAKEOFFER:
            if (seat == onTurn)
            {
                // A new offer while one stands replaces it: the protocol withdraws the old one first
                Map<String, Object> from = before;
                if (was != null)
                {
                    step(seat, Json.array("cancel"), null, before, legal);
                    from = withoutOffer(before);
                }
                // After a withdrawal the legal moves listed were its own; those with the offer closed weren't listed
                if (is != null && Json.integer(is.get("from")) == seat)
                    step(seat, Json.array("offer", Json.object("give", is.get("give"), "get", is.get("get"), "to", is.get("to"))), null, from, was != null ? null : legal);
                return;
            }
            // An addressee's own offer to the player on turn: its counteroffer
            if (was != null && is != null && OSPGame.pending(was, seat) && "counter".equals(answer(is, key)))
            {
                step(seat, Json.array("counter", ((Map<String, Object>) is.get("counters")).get(key)), null, before, legal);
                return;
            }
            break;
        case SOCMessage.REJECTOFFER:
            // An addressee's no (the last one closes the offer)
            if (was != null && seat != onTurn && OSPGame.pending(was, seat) && (is == null || "decline".equals(answer(is, key))))
            {
                step(seat, Json.array("respond", Json.object("accept", false)), null, before, legal);
                return;
            }
            break;
        case SOCMessage.ACCEPTOFFER:
        {
            int offering = osp.seatOf(((SOCAcceptOffer) message).getOfferingNumber());
            // An addressee's accept: the trade happened at once and closed the offer
            if (was != null && is == null && seat != onTurn && offering == onTurn && OSPGame.pending(was, seat))
            {
                step(seat, Json.array("respond", Json.object("accept", true)), null, before, legal);
                return;
            }
            // The player on turn takes a counteroffer
            if (was != null && is == null && seat == onTurn && "counter".equals(answer(was, Integer.toString(offering))))
            {
                step(seat, Json.array("confirm", Json.object("partner", (long) offering)), null, before, legal);
                return;
            }
            break;
        }
        case SOCMessage.CLEAROFFER:
            // The offer closed without a trade: the player on turn withdrew it, or (it had withdrawn it already) the last
            // counteroffer that kept it open went. The protocol's withdrawal is the offerer's.
            if (was != null && is == null)
            {
                step(Json.integer(was.get("from")), Json.array("cancel"), null, before, offererLegal);
                return;
            }
            break;
        default:
            break;
        }
        // A change of the game no protocol step explains (e.g. a trade on an offer nobody asked for)
        gaps++;
    }

    @SuppressWarnings("unchecked")
    private static Object answer(Map<String, Object> offer, String seat)
    {
        return ((Map<String, Object>) offer.get("answers")).get(seat);
    }

    // The state after the open offer closed without a trade: the player on turn goes on
    private static Map<String, Object> withoutOffer(Map<String, Object> state)
    {
        Map<String, Object> closed = OSPGame.copy(state);
        closed.put("offer", null);
        closed.put("prompt", "main");
        closed.put("current", closed.get("on_turn"));
        return closed;
    }

    private void pending(String kind, Object value, Map<String, Object> before, List<Object> legal)
    {
        pendingKind = kind;
        pendingValue = value;
        pendingBefore = before;
        pendingLegal = legal;
    }

    // A robber move: the victim is who lost a card, the card what the thief got
    private void robbery(int seat, Object hex, Map<String, Object> before, Map<String, Object> after, List<Object> legal)
    {
        Object victim = null;
        String stolen = null;
        for (int other = 0; other < game.maxPlayers; other++)
            if (other != seat && handSize(after, other) < handSize(before, other))
                victim = (long) other;
        Map<String, Object> had = hand(before, seat), has = hand(after, seat);
        for (String name : has.keySet())
            if (Json.integer(has.get(name)) > Json.integer(had.get(name)))
                stolen = name;
        step(seat, Json.array("move_robber", Json.object("hex", hex, "victim", victim)), victim == null ? null : Json.object("stolen", stolen), before, legal);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> player(Map<String, Object> state, int seat)
    {
        return (Map<String, Object>) ((List<Object>) state.get("players")).get(seat);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> hand(Map<String, Object> state, int seat)
    {
        return (Map<String, Object>) player(state, seat).get("hand");
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> development(Map<String, Object> state, int seat)
    {
        return (Map<String, Object>) player(state, seat).get("development");
    }

    private static int handSize(Map<String, Object> state, int seat)
    {
        return Json.integer(player(state, seat).get("hand_size"));
    }

    // The state before the first discard with the first done discards made, the next discarding seat to act
    @SuppressWarnings("unchecked")
    private Map<String, Object> discardState(int done)
    {
        Map<String, Object> state = OSPGame.copy(discardStart);
        Map<String, Object> bank = (Map<String, Object>) state.get("bank");
        for (int i = 0; i < done; i++)
        {
            int seat = discarders.get(i);
            Map<String, Object> set = (Map<String, Object>) discarded.get(seat)[0];
            Map<String, Object> player = player(state, seat), hand = (Map<String, Object>) player.get("hand");
            for (Map.Entry<String, Object> card : set.entrySet())
            {
                int count = Json.integer(card.getValue());
                hand.put(card.getKey(), (long) (Json.integer(hand.get(card.getKey())) - count));
                bank.put(card.getKey(), (long) (Json.integer(bank.get(card.getKey())) + count));
                player.put("hand_size", (long) (Json.integer(player.get("hand_size")) - count));
            }
        }
        if (done < discarders.size())
            state.put("current", (long) (int) discarders.get(done));
        return state;
    }

    // A move made: its step, and its decision and step in the record
    private void step(int seat, List<Object> action, Map<String, Object> result, Map<String, Object> before, List<Object> legal)
    {
        pendingKind = null;
        long n = steps.size();
        steps.add(Json.object("type", "step", "game_id", gameId, "n", n, "seat", (long) seat, "action", action, "result", result, "state", before));
        moves[seat]++;
        if (legal != null && legal.size() > 1)
            choices[seat]++;
        if ("end_turn".equals(action.get(0)))
            turns++;
        write(Json.object("type", "decide", "game_id", gameId, "n", n, "seat", (long) seat, "state", before, "legal", legal));
        write(Json.object("type", "step", "game_id", gameId, "n", n, "seat", (long) seat, "action", action, "result", result));
    }

    // The game is over and its winner named: the record's end and the game's statistics line
    private void end(Map<String, Object> state)
    {
        if (ended)
            return;
        ended = true;
        List<Object> vp = Json.array();
        List<Object> players = Json.array();
        for (int seat = 0; seat < game.maxPlayers; seat++)
        {
            SOCPlayer player = game.getPlayer(osp.playerOf(seat));
            vp.add((long) player.getTotalVP());
            players.add(player.getName());
        }
        SOCPlayer won = game.getPlayerWithWin();
        Object winner = won == null ? null : (Object) (long) osp.seatOf(won.getPlayerNumber());
        // The result record: each seat's line with its moves, choices and brain time added
        List<Object> seats = Json.array();
        Map<String, Long> millis = brainMillis();
        for (int seat = 0; seat < game.maxPlayers; seat++)
        {
            Map<String, Object> line = records[seat].json(seat);
            line.put("moves", (long) moves[seat]);
            line.put("choices", recording ? (Object) (long) choices[seat] : null);
            line.put("brain_cpu_ms", millis.get((String) players.get(seat)));
            seats.add(line);
        }
        result = Json.object("seed", (long) seed, "moves", (long) steps.size(), "turns", (long) turns, "seats", seats);
        write(Json.object("type", "after", "game_id", gameId, "winner", winner, "vp", vp, "truncated", winner == null,
            "rounds", (long) game.getRoundCount(), "result", result, "state", state));
        System.err.println(Json.write(Json.object("type", "osp-result", "game_id", gameId, "players", players, "winner", winner, "vp", vp,
            "rounds", (long) game.getRoundCount(), "gaps", (long) gaps, "result", result)));
        close();
        // A game whose Open Seat seat failed counts toward the run's stop rule
        FailureLimit.ended(result);
    }

    // The CPU time of each robot brain's thread in this game so far, by robot name (JSettlers' brains and ours name their
    // threads robotBrain- or ospBrain- with the robot and the game)
    private Map<String, Long> brainMillis()
    {
        Map<String, Long> millis = new LinkedHashMap<>();
        ThreadMXBean threads = ManagementFactory.getThreadMXBean();
        for (Thread thread : Thread.getAllStackTraces().keySet())
        {
            String name = thread.getName();
            String suffix = "-" + gameId;
            for (String prefix : new String[] {"robotBrain-", "ospBrain-"})
                if (name.startsWith(prefix) && name.endsWith(suffix))
                {
                    long nanos = threads.getThreadCpuTime(thread.getId());
                    if (nanos >= 0)
                        millis.put(name.substring(prefix.length(), name.length() - suffix.length()), nanos / 1_000_000);
                }
        }
        return millis;
    }

    private void write(Map<String, Object> message)
    {
        if (record == null)
            return;
        try
        {
            record.write(Json.write(message));
            record.write('\n');
            record.flush();
        }
        catch (IOException e)
        {
            System.err.println("osp: can't record " + gameId + ": " + e);
            close();
        }
    }

    /** The game is gone: a record of a game that ended without a winner gets its end */
    synchronized void close()
    {
        if (record == null)
            return;
        try
        {
            if (! ended)
            {
                ended = true;
                record.write(Json.write(Json.object("type", "after", "game_id", gameId, "winner", null, "vp", null, "truncated", true)));
                record.write('\n');
            }
            record.close();
        }
        catch (IOException e) {}
        record = null;
    }
}
