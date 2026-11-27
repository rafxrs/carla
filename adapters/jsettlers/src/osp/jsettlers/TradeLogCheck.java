package osp.jsettlers;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;

import soc.game.SOCDevCardConstants;
import soc.game.SOCGame;
import soc.game.SOCInventory;
import soc.game.SOCPlayer;
import soc.game.SOCResourceConstants;
import soc.game.SOCResourceSet;
import soc.game.SOCTradeOffer;
import soc.message.SOCAcceptOffer;
import soc.message.SOCBankTrade;
import soc.message.SOCClearOffer;
import soc.message.SOCEndTurn;
import soc.message.SOCRollDice;
import soc.message.SOCMakeOffer;
import soc.message.SOCMessageForGame;
import soc.message.SOCPickResourceType;
import soc.message.SOCPlayDevCardRequest;
import soc.message.SOCRejectOffer;

/**
 * A check for the bridge's tests (test_jsettlers_steps.py): JSettlers' trade requests, handed to the move log as JSettlers'
 * server handles them, must become the protocol's steps under player_trading "first_accept": an offer by the player on
 * turn, a decline, a counteroffer (an addressee's own offer to the player on turn, written in the offerer's terms), an
 * accept that trades at once, the offerer taking a counteroffer, a withdrawal, a new offer replacing the standing one; and
 * no step for what changes nothing in the protocol's terms (a withdrawal after everybody declined, a refused accept, the
 * offerer declining counteroffers, an offer nobody asked for). Each step's state shows the offer with the answers so far.
 * Seats are written as offsets from the seat on turn. Prints one line per scenario; exits with 1 if one is wrong.
 */
public final class TradeLogCheck
{
    private TradeLogCheck() {}

    private static final int LUMBER = SOCResourceConstants.WOOD, ORE = SOCResourceConstants.ORE;
    private static int failures;

    public static void main(String[] args) throws Exception
    {
        // Answers out of seat order: a decline, a counteroffer, then an accept that trades at once
        scenario("decline, counteroffer, accept", (t) ->
        {
            t.offer(1, LUMBER, 1, ORE, 1, 2, 3);
            t.reject(3);
            t.counter(2, ORE, 1, LUMBER, 2);
            t.accept(1);
            t.expect("offer", "respond", "counter", "respond");
            t.expectSeats(0, 3, 2, 1);
            t.expectAction(0, Json.array("offer", Json.object("give", cards("lumber", 1), "get", cards("ore", 1), "to", t.seats(1, 2, 3))));
            t.expectAction(1, Json.array("respond", Json.object("accept", false)));
            // In the offerer's terms: it would give two lumber for one ore
            t.expectAction(2, Json.array("counter", Json.object("give", cards("lumber", 2), "get", cards("ore", 1))));
            t.expectAction(3, Json.array("respond", Json.object("accept", true)));
            t.expectOffer(0, null);
            t.expectOffer(1, t.offerState(cards("lumber", 1), cards("ore", 1), new int[] {1, 2, 3}, Json.object(), null));
            t.expectOffer(3, t.offerState(cards("lumber", 1), cards("ore", 1), new int[] {1, 2, 3},
                Json.object(t.key(2), "counter", t.key(3), "decline"), Json.object(t.key(2), Json.object("give", cards("lumber", 2), "get", cards("ore", 1)))));
            t.expectHand(0, 1, 3);
            t.expectHand(1, 3, 1);
            t.expectNoOffer();
            t.expectGaps(0);
        });
        // Everybody declines: the offer closes, and the offerer's withdrawal afterwards changes nothing
        scenario("everybody declines, then a withdrawal", (t) ->
        {
            t.offer(1, LUMBER, 1, ORE, 1, 2, 3);
            t.reject(2);
            t.reject(3);
            t.reject(1);
            t.clear(0);
            t.expect("offer", "respond", "respond", "respond");
            t.expectNoOffer();
            t.expectGaps(0);
        });
        // The offerer takes a counteroffer while others are still asked
        scenario("the offerer takes a counteroffer", (t) ->
        {
            t.offer(1, LUMBER, 1, ORE, 1, 2, 3);
            t.counter(3, ORE, 1, LUMBER, 2);
            t.acceptCounter(3);
            t.expect("offer", "counter", "confirm");
            t.expectSeats(0, 3, 0);
            t.expectAction(2, Json.array("confirm", Json.object("partner", (long) t.seat(3))));
            t.expectHand(0, 0, 3);
            t.expectHand(3, 4, 1);
            t.expectNoOffer();
            t.expectGaps(0);
        });
        // A withdrawal while answers are pending, and a new offer that replaces the standing one
        scenario("withdrawal and a replacing offer", (t) ->
        {
            t.offer(1, LUMBER, 1, ORE, 1, 2, 3);
            t.clear(0);
            t.offer(1, LUMBER, 1, ORE, 1, 2);
            t.offer(2, LUMBER, 1, ORE, 2);
            t.expect("offer", "cancel", "offer", "cancel", "offer");
            t.expectAction(4, Json.array("offer", Json.object("give", cards("lumber", 2), "get", cards("ore", 1), "to", t.seats(2))));
            t.expectOffer(4, null);
            t.expectGaps(0);
        });
        // What changes nothing in the protocol's terms: an accept JSettlers refuses (the addressee lacks the cards), the
        // offerer declining counteroffers
        scenario("refused accept, declined counteroffer", (t) ->
        {
            t.offer(1, LUMBER, 3, ORE, 1);
            t.accept(1);
            t.counter(1, LUMBER, 1, ORE, 1);
            t.reject(0);
            t.clear(0);
            // The counteroffer keeps the offer open until its maker, turned down, withdraws it
            t.expect("offer", "counter");
            t.clear(1);
            t.expect("offer", "counter", "cancel");
            t.expectSeats(0, 1, 0);
            t.expectHand(0, 2, 2);
            t.expectGaps(0);
        });
        // An offer nobody asked for (only humans make them) is no step; trading on it is a gap
        scenario("unprompted offer to the player on turn", (t) ->
        {
            t.counter(2, ORE, 1, LUMBER, 1);
            t.acceptCounter(2);
            t.expect();
            t.expectGaps(1);
        });
        // JSettlers keeps an addressee's counteroffer after the offer it answered: it is no answer to a later offer
        scenario("a counteroffer to an earlier offer", (t) ->
        {
            t.offer(1, LUMBER, 1, ORE, 1, 2, 3);
            t.counter(2, ORE, 1, LUMBER, 2);
            t.clear(0);
            t.offer(1, LUMBER, 1, ORE, 1, 2, 3);
            t.expect("offer", "counter", "cancel", "offer");
            t.expectOpenOffer(t.offerState(cards("lumber", 1), cards("ore", 1), new int[] {1, 2, 3}, Json.object(), null));
            t.expectGaps(0);
        });
        // A robot may go on (a bank trade, the end of its turn) with its offer standing: the protocol withdraws it once,
        // and it stays closed though JSettlers still holds it
        scenario("the player on turn goes on with its offer standing", (t) ->
        {
            t.lumber(0, 4);
            Map<String, Object> bank = t.bankTradeMove(0);
            t.offer(1, LUMBER, 1, ORE, 1, 2, 3);
            t.bankTrade(0, bank);
            t.endTurn(0);
            t.expect("offer", "cancel", "bank_trade", "end_turn");
            t.expectGaps(0);
        });
        // An offer made before the roll (a robot's late counteroffer, after the turn changed) is none for the protocol, also
        // after the roll
        scenario("an offer before the roll", (t) ->
        {
            t.endTurn(0);
            t.offerBy(1, 1, LUMBER, 1, ORE, 2);
            t.roll(1);
            t.expect("end_turn", "roll");
            t.expectNoOffer();
            t.expectGaps(0);
        });
        // A JSettlers robot takes a counteroffer as a no to its offer: it withdraws its offer, then may accept the counterer's,
        // which JSettlers keeps standing. For the protocol the counteroffer belongs to the offer: it stays open until the
        // counteroffers are gone, and taking one is the offerer's pick
        scenario("the offerer withdraws, then takes the counteroffer", (t) ->
        {
            t.offer(1, LUMBER, 1, ORE, 1, 2, 3);
            t.reject(1);
            t.counter(3, ORE, 1, LUMBER, 2);
            t.clear(0);
            t.acceptCounter(3);
            t.expect("offer", "respond", "counter", "confirm");
            t.expectSeats(0, 1, 3, 0);
            t.expectHand(0, 0, 3);
            t.expectNoOffer();
            t.expectGaps(0);
        });
        scenario("the offerer withdraws, then the counterer", (t) ->
        {
            t.offer(1, LUMBER, 1, ORE, 1, 2, 3);
            t.counter(2, ORE, 1, LUMBER, 2);
            t.clear(0);
            t.clear(2);
            t.expect("offer", "counter", "cancel");
            t.expectSeats(0, 2, 0);
            t.expectNoOffer();
            t.expectGaps(0);
        });
        scenario("the offerer withdraws and goes on with a counteroffer standing", (t) ->
        {
            t.offer(1, LUMBER, 1, ORE, 1, 2, 3);
            t.counter(2, ORE, 1, LUMBER, 2);
            t.clear(0);
            t.endTurn(0);
            t.expect("offer", "counter", "cancel", "end_turn");
            t.expectNoOffer();
            t.expectGaps(0);
        });
        // An addressee's late accept of an offer whose maker went on with it standing: the protocol withdrew the offer, so the
        // bridge withdraws it at JSettlers too, which then refuses the trade
        scenario("a late accept of an offer its maker went on from", (t) ->
        {
            t.lumber(0, 4);
            Map<String, Object> bank = t.bankTradeMove(0);
            t.offer(1, LUMBER, 1, ORE, 1, 2, 3);
            t.bankTrade(0, bank);
            t.accept(1);
            t.expect("offer", "cancel", "bank_trade");
            t.expectHand(1, 2, 2);
            t.expectGaps(0);
        });
        // An addressee that countered and then accepts the offer after all: JSettlers trades, the protocol's answers are final,
        // so it is a gap, not an accept
        scenario("an addressee that countered accepts after all", (t) ->
        {
            t.offer(1, LUMBER, 1, ORE, 1, 2, 3);
            t.counter(1, ORE, 1, LUMBER, 2);
            t.accept(1);
            t.expect("offer", "counter");
            t.expectGaps(1);
        });
        // Not a trade, but a two-request move the same log writes: a Monopoly that takes nothing (nobody else holds ore) leaves
        // the state as the card's play left it, and its step still comes, with nothing taken
        scenario("a Monopoly that takes nothing", (t) ->
        {
            t.monopoly(ORE);
            t.expect("play_monopoly");
            t.expectAction(0, Json.array("play_monopoly", "ore"));
            t.expectGaps(0);
        });
        System.out.println(failures == 0 ? "trade steps checked" : failures + " trade check(s) failed");
        System.exit(failures == 0 ? 0 : 1);
    }

    private static Map<String, Object> cards(String name, long count)
    {
        return Json.object(name, count);
    }

    private interface Body
    {
        void run(Trading t) throws Exception;
    }

    private static void scenario(String name, Body body) throws Exception
    {
        Trading t = new Trading();
        try
        {
            body.run(t);
            System.out.println("ok: " + name);
        }
        catch (AssertionError | RuntimeException e)
        {
            failures++;
            System.out.println("FAILED: " + name + ": " + e);
        }
    }

    /** A game after a roll, every hand two of each resource, its move log, and requests handed to it as the server would */
    private static final class Trading
    {
        final SOCGame game;
        final OSPGame osp;
        MoveLog log;
        // The seat on turn: the others are written as offsets from it
        final int onTurn;

        @SuppressWarnings("unchecked")
        Trading()
        {
            game = OSPBoard.newGame("trade-check");
            osp = new OSPGame(game);
            Random random = new Random(3);
            // Setup by random placements, then rolls and ends of turn until the player on turn may trade
            while (! osp.isOver() && ! (game.getGameState() == SOCGame.PLAY1 && game.getRoundCount() > 2))
            {
                int seat = osp.acting();
                List<Object> legal = game.getGameState() == SOCGame.WAITING_FOR_DISCARDS ? osp.discardMoves(seat) : osp.legal(seat);
                List<Object> move = (List<Object>) legal.get(random.nextInt(legal.size()));
                for (Object m : legal)
                    if (((List<Object>) m).get(0).equals("roll") || ((List<Object>) m).get(0).equals("end_turn"))
                        move = (List<Object>) m;
                osp.apply(move, seat);
            }
            onTurn = osp.seatOf(game.getCurrentPlayerNumber());
            for (int pn = 0; pn < game.maxPlayers; pn++)
            {
                SOCResourceSet hand = game.getPlayer(pn).getResources();
                hand.clear();
                for (int type = SOCResourceConstants.CLAY; type <= SOCResourceConstants.WOOD; type++)
                    hand.add(2, type);
            }
            log = new MoveLog(game, osp.state(-1));
        }

        // The protocol seat of an offset from the seat on turn, and its JSettlers player
        int seat(int offset)
        {
            return (onTurn + offset) % game.maxPlayers;
        }

        String key(int offset)
        {
            return Integer.toString(seat(offset));
        }

        int pn(int offset)
        {
            return osp.playerOf(seat(offset));
        }

        // The seats of offsets as the protocol lists them: in seat order
        List<Object> seats(int... offsets)
        {
            int[] seats = Arrays.stream(offsets).map(this::seat).sorted().toArray();
            List<Object> list = new ArrayList<>();
            for (int s : seats)
                list.add((long) s);
            return list;
        }

        Map<String, Object> offerState(Map<String, Object> give, Map<String, Object> get, int[] to, Map<String, Object> answers, Map<String, Object> counters)
        {
            Map<String, Object> offer = Json.object("from", (long) onTurn, "give", give, "get", get, "to", seats(to), "answers", answers);
            if (counters != null)
                offer.put("counters", counters);
            return offer;
        }

        private interface Handler
        {
            void handle();
        }

        private void send(int offset, SOCMessageForGame message, Handler handler) throws Exception
        {
            MoveLog.dispatchAs(log, (g, m, c) ->
            {
                handler.handle();
                return true;
            }, game, message, null, seat(offset));
        }

        private static SOCResourceSet set(int type, int count)
        {
            SOCResourceSet set = new SOCResourceSet();
            set.add(count, type);
            return set;
        }

        // The player on turn offers to the given offsets, stored as SOCGameMessageHandler.handleMAKEOFFER does
        void offer(int give, int giveType, int get, int getType, int... to) throws Exception
        {
            offerBy(0, give, giveType, get, getType, to);
        }

        // An offer by the seat at the offset (the player on turn, or any seat before the roll) to the given offsets
        void offerBy(int by, int give, int giveType, int get, int getType, int... to) throws Exception
        {
            boolean[] addressed = new boolean[game.maxPlayers];
            for (int offset : to)
                addressed[pn(offset)] = true;
            SOCTradeOffer offer = new SOCTradeOffer(game.getName(), pn(by), addressed, set(giveType, give), set(getType, get));
            send(by, new SOCMakeOffer(game.getName(), offer), () -> game.getPlayer(pn(by)).setCurrentOffer(offer));
        }

        // The seat's lumber set to a count; the move log starts anew from there (the change is no request's)
        void lumber(int offset, int count)
        {
            SOCResourceSet hand = game.getPlayer(pn(offset)).getResources();
            hand.setAmount(count, LUMBER);
            log = new MoveLog(game, osp.state(-1));
        }

        // The first bank trade the legal moves of the player on turn list (at its best rate: a harbor changes it), read before
        // an offer opens (the legal moves are then the answers)
        @SuppressWarnings("unchecked")
        Map<String, Object> bankTradeMove(int offset)
        {
            for (Object move : osp.legal(seat(offset)))
                if ("bank_trade".equals(((List<Object>) move).get(0)))
                    return (Map<String, Object>) ((List<Object>) move).get(1);
            throw new AssertionError("no bank trade listed");
        }

        // A bank trade of the player on turn, as JSettlers' server makes it
        void bankTrade(int offset, Map<String, Object> trade) throws Exception
        {
            SOCResourceSet giving = OSPGame.resourceSet(trade.get("give")), getting = OSPGame.resourceSet(trade.get("get"));
            send(offset, new SOCBankTrade(game.getName(), giving, getting, pn(offset)), () -> game.makeBankTrade(giving, getting));
        }

        // The player on turn plays a Monopoly on the resource while nobody else holds any, as JSettlers' server handles its two
        // requests (the card, then the resource); the log starts anew after the hands and the card are set
        void monopoly(int type) throws Exception
        {
            for (int offset = 1; offset < game.maxPlayers; offset++)
                game.getPlayer(pn(offset)).getResources().setAmount(0, type);
            game.getPlayer(pn(0)).getInventory().addDevCard(1, SOCInventory.OLD, SOCDevCardConstants.MONO);
            log = new MoveLog(game, osp.state(-1));
            send(0, new SOCPlayDevCardRequest(game.getName(), SOCDevCardConstants.MONO), game::playMonopoly);
            send(0, new SOCPickResourceType(game.getName(), type), () -> game.doMonopolyAction(type));
        }

        void endTurn(int offset) throws Exception
        {
            send(offset, new SOCEndTurn(game.getName()), game::endTurn);
        }

        void roll(int offset) throws Exception
        {
            send(offset, new SOCRollDice(game.getName()), game::rollDice);
        }

        // The open offer now
        void expectOpenOffer(Map<String, Object> offer)
        {
            Object got = log.osp.state(-1).get("offer");
            String a = got == null ? null : Json.write(got, true), b = Json.write(offer, true);
            check(Objects.equals(a, b), "open offer " + a + ", expected " + b);
        }

        // An addressee's own offer to the player on turn: it gives give of giveType and gets get of getType
        void counter(int offset, int giveType, int give, int getType, int get) throws Exception
        {
            boolean[] toCurrent = new boolean[game.maxPlayers];
            toCurrent[pn(0)] = true;
            SOCTradeOffer offer = new SOCTradeOffer(game.getName(), pn(offset), toCurrent, set(giveType, give), set(getType, get));
            send(offset, new SOCMakeOffer(game.getName(), offer), () -> game.getPlayer(pn(offset)).setCurrentOffer(offer));
        }

        // REJECTOFFER: no to every offer made to the player (SOCGame.rejectTradeOffersTo)
        void reject(int offset) throws Exception
        {
            send(offset, new SOCRejectOffer(game.getName(), pn(offset)), () -> game.rejectTradeOffersTo(pn(offset)));
        }

        // ACCEPTOFFER of the offer of the player on turn by an addressee: the trade happens if JSettlers allows it, then every
        // offer is cleared (SOCGameMessageHandler.executeTrade)
        void accept(int offset) throws Exception
        {
            send(offset, new SOCAcceptOffer(game.getName(), pn(offset), pn(0)), () -> trade(pn(0), pn(offset)));
        }

        // ACCEPTOFFER by the player on turn of an addressee's own offer (a counteroffer)
        void acceptCounter(int offset) throws Exception
        {
            send(0, new SOCAcceptOffer(game.getName(), pn(0), pn(offset)), () -> trade(pn(offset), pn(0)));
        }

        private void trade(int offering, int accepting)
        {
            if (! game.canMakeTrade(offering, accepting))
                return;
            game.makeTrade(offering, accepting);
            for (int pn = 0; pn < game.maxPlayers; pn++)
                game.getPlayer(pn).setCurrentOffer(null);
        }

        // CLEAROFFER: the player withdraws its own offer
        void clear(int offset) throws Exception
        {
            send(offset, new SOCClearOffer(game.getName(), pn(offset)), () -> game.getPlayer(pn(offset)).setCurrentOffer(null));
        }

        List<Map<String, Object>> steps()
        {
            return log.stepsFrom(0);
        }

        @SuppressWarnings("unchecked")
        void expect(String... kinds)
        {
            List<String> got = new ArrayList<>();
            for (Map<String, Object> step : steps())
                got.add((String) ((List<Object>) step.get("action")).get(0));
            check(got.equals(List.of(kinds)), "steps " + got + ", expected " + List.of(kinds));
        }

        void expectSeats(int... offsets)
        {
            List<Map<String, Object>> steps = steps();
            for (int i = 0; i < offsets.length; i++)
                check(Json.integer(steps.get(i).get("seat")) == seat(offsets[i]), "step " + i + " by seat " + steps.get(i).get("seat") + ", expected " + seat(offsets[i]));
        }

        void expectAction(int step, List<Object> action)
        {
            String got = Json.write(steps().get(step).get("action"), true), wanted = Json.write(action, true);
            check(got.equals(wanted), "step " + step + " action " + got + ", expected " + wanted);
        }

        // The offer in the state a step was made in
        @SuppressWarnings("unchecked")
        void expectOffer(int step, Map<String, Object> offer)
        {
            Object got = ((Map<String, Object>) steps().get(step).get("state")).get("offer");
            String a = got == null ? null : Json.write(got, true), b = offer == null ? null : Json.write(offer, true);
            check(Objects.equals(a, b), "offer before step " + step + ": " + a + ", expected " + b);
        }

        void expectNoOffer()
        {
            // The move log's view, whose memory of offers the protocol closed is the one the robots share
            check(log.osp.state(-1).get("offer") == null, "an offer still stands: " + log.osp.state(-1).get("offer") + " (game state " + game.getGameState() + ")");
        }

        // The seat's lumber and ore
        void expectHand(int offset, int lumber, int ore)
        {
            SOCPlayer player = game.getPlayer(pn(offset));
            int a = player.getResources().getAmount(LUMBER), b = player.getResources().getAmount(ORE);
            check(a == lumber && b == ore, "seat offset " + offset + " holds " + a + " lumber and " + b + " ore, expected " + lumber + " and " + ore);
        }

        void expectGaps(long gaps)
        {
            check(log.gaps() == gaps, log.gaps() + " gaps, expected " + gaps);
        }

        private static void check(boolean ok, String message)
        {
            if (! ok)
                throw new AssertionError(message);
        }
    }
}
