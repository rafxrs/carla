package osp.jsettlers;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.Arrays;

import soc.game.SOCGame;

/**
 * A check for the bridge's tests (test_jsettlers_steps.py): JSettlers numbers its robot-only games afresh in every server
 * run (~botsOnly~1, ...), so a second run recording into the same directory (-Dosp.record) must keep the first run's
 * records. Records a game of the same name twice, as two runs would. Prints what it found; exits with 1 if a record was
 * overwritten.
 */
public final class RecordCheck
{
    private RecordCheck() {}

    public static void main(String[] args) throws Exception
    {
        File directory = Files.createTempDirectory("osp-record-check").toFile();
        System.setProperty("osp.record", directory.getPath());
        for (int run = 0; run < 2; run++)
        {
            SOCGame game = OSPBoard.newGame("~botsOnly~1");
            new MoveLog(game, new OSPGame(game).state(-1)).close();
        }
        File[] records = directory.listFiles();
        boolean ok = records != null && records.length == 2;
        if (ok)
            for (File record : records)
                ok &= Files.readAllLines(record.toPath(), StandardCharsets.UTF_8).get(0).contains("\"before\"");
        System.out.println(ok ? "both runs' records kept" : "records: " + (records == null ? "none" : Arrays.toString(directory.list())));
        System.exit(ok ? 0 : 1);
    }
}
