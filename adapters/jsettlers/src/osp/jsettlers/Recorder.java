package osp.jsettlers;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.zip.GZIPOutputStream;

import soc.game.SOCGame;

/**
 * Records JSettlers games as Open Seat Protocol referee messages, the input of Lanzenbrecher's replay harness (OpenSeat
 * replay): one JSON object per line, gzipped; per game a before (board, ruleset, full state), a decide before every move
 * (full state, the legal moves by JSettlers' rules) and a step after it (the move and its result), and an after. The
 * games are played by JSettlers' own rules (SOCGame) with weighted random choices, like Catanatron's weighted random bot:
 * cities before settlements before development cards before anything else, so games end.
 *
 * Usage: Recorder --games N [--seed S] [--ruleset standard|jsettlers] --out FILE.jsonl.gz
 * The ruleset the games announce: the rulebook without trades between players (default), or with the options in which
 * JSettlers' rules are known to be narrower (JSETTLERS_OPTIONS), so a replay folds them in and finds what else differs.
 */
public final class Recorder
{
    // Rounds after which a game is cut short
    private static final int ROUND_LIMIT = 300;

    // Where JSettlers' rules are narrower than the rulebook: bank trades only at the best rate, only players with cards
    // robbed, a Road Building card's roads placed at once (the card only with a road left and a place for it)
    private static final Object[] JSETTLERS_OPTIONS = {"bank_rates", "best_only", "robber_victims", "with_cards", "road_building", "forced"};

    public static void main(String[] args) throws IOException
    {
        int games = 10;
        long seed = 1;
        String out = null;
        Map<String, Object> ruleset = Json.object("preset", "standard", "player_trading", "none");
        for (int i = 0; i < args.length; i++)
        {
            switch (args[i])
            {
            case "--games": games = Integer.parseInt(args[++i]); break;
            case "--seed": seed = Long.parseLong(args[++i]); break;
            case "--out": out = args[++i]; break;
            case "--ruleset":
                if (args[++i].equals("jsettlers"))
                    for (int o = 0; o < JSETTLERS_OPTIONS.length; o += 2)
                        ruleset.put((String) JSETTLERS_OPTIONS[o], JSETTLERS_OPTIONS[o + 1]);
                break;
            default:
                System.err.println("Usage: Recorder --games N [--seed S] --out FILE.jsonl.gz");
                System.exit(2);
            }
        }
        if (out == null)
        {
            System.err.println("Usage: Recorder --games N [--seed S] --out FILE.jsonl.gz");
            System.exit(2);
        }
        int won = 0;
        try (Writer writer = new OutputStreamWriter(new GZIPOutputStream(new FileOutputStream(out)), StandardCharsets.UTF_8))
        {
            for (int g = 0; g < games; g++)
            {
                String id = "jsettlers-" + (seed + g);
                if (record(OSPBoard.newGame(id), id, new Random(seed + g), ruleset, writer))
                    won++;
                System.err.println("game " + (g + 1) + "/" + games + ": " + id);
            }
        }
        System.err.println(games + " games to " + out + ", " + won + " won, " + (games - won) + " cut short");
    }

    /** Plays the game to its end and writes its messages; returns whether someone won */
    public static boolean record(SOCGame game, String id, Random random, Map<String, Object> ruleset, Writer writer) throws IOException
    {
        OSPGame osp = new OSPGame(game);
        write(writer, Json.object("type", "before", "game_id", id, "players", (long) game.maxPlayers, "jsettlers_first", (long) osp.first,
            "ruleset", ruleset, "board", osp.board.boardJson(), "state", osp.state(-1)));
        long n = 0;
        while (! osp.isOver() && game.getRoundCount() <= ROUND_LIMIT)
        {
            int seat = osp.acting();
            List<Object> legal = osp.legal(seat);
            if (legal.isEmpty())
                throw new IllegalStateException(id + " move " + n + ": no legal move for seat " + seat + " in game state " + game.getGameState());
            write(writer, Json.object("type", "decide", "game_id", id, "n", n, "seat", (long) seat, "state", osp.state(-1), "legal", legal));
            @SuppressWarnings("unchecked")
            List<Object> move = (List<Object>) choose(legal, random);
            int before = game.getGameState();
            Map<String, Object> result = osp.apply(move, seat);
            write(writer, Json.object("type", "step", "game_id", id, "n", n, "seat", (long) seat, "action", move, "result", result));
            n++;
            if (game.getGameState() == before && move.get(0).equals("end_turn") && game.getCurrentPlayerNumber() == seat)
                throw new IllegalStateException(id + " move " + n + ": JSettlers didn't end the turn");
        }
        boolean over = osp.isOver();
        List<Object> vp = new ArrayList<>();
        for (int seat = 0; seat < game.maxPlayers; seat++)
            vp.add((long) game.getPlayer(osp.playerOf(seat)).getTotalVP());
        Object winner = over && game.getPlayerWithWin() != null ? (Object) (long) osp.seatOf(game.getPlayerWithWin().getPlayerNumber()) : null;
        write(writer, Json.object("type", "after", "game_id", id, "winner", winner, "vp", vp, "truncated", winner == null,
            "rounds", (long) game.getRoundCount(), "state", osp.state(-1)));
        return winner != null;
    }

    // Weighted random: a city before a settlement before a development card before anything else
    private static Object choose(List<Object> legal, Random random)
    {
        double total = 0;
        double[] weights = new double[legal.size()];
        for (int i = 0; i < legal.size(); i++)
        {
            String type = (String) ((List<?>) legal.get(i)).get(0);
            weights[i] = type.equals("build_city") ? 10000 : type.equals("build_settlement") ? 1000 : type.equals("buy_development") ? 100 : 1;
            total += weights[i];
        }
        double pick = random.nextDouble() * total;
        for (int i = 0; i < legal.size(); i++)
        {
            pick -= weights[i];
            if (pick < 0)
                return legal.get(i);
        }
        return legal.get(legal.size() - 1);
    }

    private static void write(Writer writer, Map<String, Object> message) throws IOException
    {
        writer.write(Json.write(message));
        writer.write('\n');
    }
}
