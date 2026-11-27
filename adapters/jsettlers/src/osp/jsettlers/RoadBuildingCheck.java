package osp.jsettlers;

import java.util.ArrayList;
import java.util.List;

import soc.game.SOCBoard;
import soc.game.SOCDevCardConstants;
import soc.game.SOCGame;
import soc.game.SOCInventory;
import soc.game.SOCPlayer;
import soc.game.SOCRoad;

/**
 * A check for the bridge's tests (test_jsettlers_steps.py): road_building forced. JSettlers lets a player play Road
 * Building without a place for a road, and waits for a free road that has no place; the protocol's rules allow the card
 * only with a place, and a first free road that leaves none gives the second up. Plays the setup with the first legal
 * moves, then (1) takes every place for a road from the player on turn, who holds the card, and (2) leaves that player
 * one place, whose ends another player's roads block, plays the card and places the road. Prints what it found; exits
 * with 1 if a check failed.
 */
public final class RoadBuildingCheck
{
    private RoadBuildingCheck() {}

    // A game after its setup, in the main phase of the first player, who holds a Road Building card
    private static final class Setup
    {
        final SOCGame game;
        final OSPGame osp;
        final int pn, seat;
        final SOCPlayer player;

        Setup(String name)
        {
            game = OSPBoard.newGame(name);
            osp = new OSPGame(game);
            while (osp.prompt().startsWith("setup_"))
            {
                int acting = osp.acting();
                @SuppressWarnings("unchecked")
                List<Object> move = (List<Object>) osp.legal(acting).get(0);
                osp.apply(move, acting);
            }
            // Rolled (no 7), so the card is the only thing that matters
            game.setCurrentDice(6);
            game.setGameState(SOCGame.PLAY1);
            pn = game.getCurrentPlayerNumber();
            seat = osp.seatOf(pn);
            player = game.getPlayer(pn);
            player.getInventory().addDevCard(1, SOCInventory.OLD, SOCDevCardConstants.ROADS);
        }
    }

    private static final List<Object> PLAY = Json.array("play_road_building");

    public static void main(String[] args)
    {
        List<String> failed = new ArrayList<>();

        // (1) No place for a road: JSettlers would let the card be played, the protocol's rules don't
        Setup none = new Setup("road-building-check-1");
        check(failed, none.osp.legal(none.seat).contains(PLAY), "the card is offered with places for roads");
        for (int edge : none.osp.board.edges())
            none.player.clearPotentialRoad(edge);
        check(failed, none.game.canPlayRoadBuilding(none.pn), "JSettlers lets the card be played without a place");
        check(failed, ! none.osp.legal(none.seat).contains(PLAY), "the card isn't offered without a place for a road");

        // (2) One place, whose ends another player's roads block: the first free road leaves no place for the second
        Setup one = new Setup("road-building-check-2");
        SOCBoard board = one.game.getBoard();
        List<Integer> land = new ArrayList<>();
        for (int edge : one.osp.board.edges())
            land.add(edge);
        int site = -1;
        for (int edge : land)
            if (one.player.isPotentialRoad(edge))
            {
                site = edge;
                break;
            }
        SOCPlayer other = one.game.getPlayer((one.pn + 1) % 4);
        for (int node : board.getAdjacentNodesToEdge_arr(site))
            for (int edge : board.getAdjacentEdgesToNode(node))
                if (edge != site && land.contains(edge) && board.roadOrShipAtEdge(edge) == null)
                    one.game.putPiece(new SOCRoad(other, edge, board));
        for (int edge : land)
            if (edge != site)
                one.player.clearPotentialRoad(edge);
        check(failed, one.game.getGameState() == SOCGame.PLAY1, "the blocking roads leave the turn as it was");
        check(failed, one.osp.legal(one.seat).contains(PLAY), "the card is offered with one place");
        one.osp.apply(PLAY, one.seat);
        List<Object> road = Json.array("build_road", one.osp.board.path(site));
        check(failed, one.osp.legal(one.seat).equals(Json.array(road)), "the first free road goes to the one place");
        one.osp.apply(road, one.seat);
        check(failed, one.game.getGameState() == SOCGame.PLAY1, "the second free road is given up (game state " + one.game.getGameState() + ")");
        check(failed, "main".equals(one.osp.prompt()) && one.osp.legal(one.seat).contains(Json.array("end_turn")), "the turn goes on");
        check(failed, ! one.player.getInventory().hasPlayable(SOCDevCardConstants.ROADS) && one.player.hasPlayedDevCard(), "the card stays played");

        System.out.println(failed.isEmpty() ? "road building checked" : "failed: " + String.join("; ", failed));
        System.exit(failed.isEmpty() ? 0 : 1);
    }

    private static void check(List<String> failed, boolean ok, String what)
    {
        if (! ok)
            failed.add(what);
    }
}
