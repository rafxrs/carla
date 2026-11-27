package osp.jsettlers;

import java.util.Map;

import soc.game.SOCGame;
import soc.game.SOCResourceConstants;

/**
 * A check for the bridge's tests (test_jsettlers_steps.py): JSettlers has no resource bank, so a player can hold more
 * than the 19 cards of a resource; the protocol's state then counts that resource's bank as 0, not below. Deals 21 ore
 * to one player and reads the state. Prints what it found; exits with 1 if a count went below 0.
 */
public final class BankCheck
{
    private BankCheck() {}

    @SuppressWarnings("unchecked")
    public static void main(String[] args) throws Exception
    {
        SOCGame game = OSPBoard.newGame("bank-check");
        OSPGame osp = new OSPGame(game);
        game.getPlayer(osp.playerOf(0)).getResources().setAmount(21, SOCResourceConstants.ORE);
        Map<String, Object> bank = (Map<String, Object>) osp.state(-1).get("bank");
        boolean ok = Json.integer(bank.get("ore")) == 0;
        for (Object count : bank.values())
            ok &= Json.integer(count) >= 0;
        System.out.println(ok ? "the bank stops at 0" : "the bank counts " + Json.write(bank));
        System.exit(ok ? 0 : 1);
    }
}
