package osp.jsettlers;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;

import soc.game.SOCGame;
import soc.game.SOCResourceSet;
import soc.message.SOCDiscard;

/**
 * A check for the bridge's tests (test_jsettlers_steps.py): after a 7 that makes several players discard, JSettlers'
 * players discard at the same time, here from the highest seat down; the move log must write the discards one after
 * another from the lowest seat up, each with the state the protocol's order gives it (the lower seats' discards made,
 * its own not). Plays a JSettlers game by its rules until such a 7, then hands the discards to the log as JSettlers'
 * server would. Prints what it found; exits with 1 if the order or a state is wrong.
 */
public final class MoveLogCheck
{
    private MoveLogCheck() {}

    @SuppressWarnings("unchecked")
    public static void main(String[] args) throws Exception
    {
        for (long seed = 1; seed < 200; seed++)
        {
            SOCGame game = OSPBoard.newGame("discard-check-" + seed);
            OSPGame osp = new OSPGame(game);
            Random random = new Random(seed);
            // Players who hoard their cards: they roll and end their turns, and build only in the setup
            while (! osp.isOver() && game.getRoundCount() < 60)
            {
                int seat = osp.acting();
                List<Object> legal = osp.legal(seat);
                if (game.getGameState() == SOCGame.WAITING_FOR_DISCARDS)
                    break;
                List<Object> move = (List<Object>) legal.get(0);
                for (Object m : legal)
                    if (((List<Object>) m).get(0).equals("roll") || ((List<Object>) m).get(0).equals("end_turn"))
                        move = (List<Object>) m;
                if (move.get(0).equals("build_settlement") || move.get(0).equals("build_road") || move.get(0).equals("move_robber"))
                    move = (List<Object>) legal.get(random.nextInt(legal.size()));
                osp.apply(move, seat);
            }
            List<Integer> discarders = new ArrayList<>();
            for (int seat = 0; seat < game.maxPlayers; seat++)
                if (game.getPlayer(osp.playerOf(seat)).getNeedToDiscard())
                    discarders.add(seat);
            if (game.getGameState() != SOCGame.WAITING_FOR_DISCARDS || discarders.size() < 2)
                continue;

            MoveLog log = new MoveLog(game, osp.state(-1));
            // The highest seat first, as JSettlers' players may send them
            for (int i = discarders.size() - 1; i >= 0; i--)
            {
                int seat = discarders.get(i), pn = osp.playerOf(seat);
                List<Object> discard = (List<Object>) osp.discardMoves(seat).get(0);
                SOCResourceSet set = OSPGame.resourceSet(discard.get(1));
                MoveLog.dispatchAs(log, (g, message, connection) ->
                {
                    g.discard(pn, set);
                    return true;
                }, game, new SOCDiscard(game.getName(), pn, set), null, seat);
            }
            List<Map<String, Object>> steps = log.stepsFrom(0);
            boolean ok = steps.size() == discarders.size();
            for (int i = 0; ok && i < steps.size(); i++)
            {
                Map<String, Object> step = steps.get(i), state = (Map<String, Object>) step.get("state");
                int seat = Json.integer(step.get("seat"));
                ok &= seat == discarders.get(i) && Json.integer(state.get("current")) == seat;
                // The lower seats have discarded, this one not yet
                for (int j = 0; ok && j < i; j++)
                {
                    Map<String, Object> earlier = steps.get(j), earlierState = (Map<String, Object>) earlier.get("state");
                    int lower = Json.integer(earlier.get("seat"));
                    Map<String, Object> set = (Map<String, Object>) ((List<Object>) earlier.get("action")).get(1);
                    for (String card : new String[] {"brick", "lumber", "ore", "grain", "wool"})
                    {
                        int had = Json.integer(hand(earlierState, lower).get(card)), has = Json.integer(hand(state, lower).get(card));
                        int given = set.containsKey(card) ? Json.integer(set.get(card)) : 0;
                        ok &= has == had - given;
                    }
                }
                Map<String, Object> start = (Map<String, Object>) steps.get(0).get("state");
                ok &= Json.write(hand(state, seat), true).equals(Json.write(hand(start, seat), true));
            }
            System.out.println((ok ? "discards in seat order" : "discards out of order or with wrong states") + ": seats " + discarders
                + " discarded from the highest, steps " + seatsOf(steps) + " (game seed " + seed + ")");
            System.exit(ok ? 0 : 1);
        }
        System.out.println("no 7 with two players discarding");
        System.exit(1);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> hand(Map<String, Object> state, int seat)
    {
        return (Map<String, Object>) ((Map<String, Object>) ((List<Object>) state.get("players")).get(seat)).get("hand");
    }

    private static List<Object> seatsOf(List<Map<String, Object>> steps)
    {
        List<Object> seats = new ArrayList<>();
        for (Map<String, Object> step : steps)
            seats.add(step.get("seat"));
        return seats;
    }
}
