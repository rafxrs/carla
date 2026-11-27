package osp.jsettlers;

import java.util.concurrent.atomic.AtomicBoolean;

import soc.game.SOCGame;

/**
 * A check for the bridge's tests (test_jsettlers_steps.py): a robot's read of the server's game (OSPServer.read) waits
 * while the server holds the game's monitor, and while its move log is handling a request; and the server destroying the
 * game from a thread that holds its monitor (ending a slow player's turn, which may end the game) doesn't wait for a read
 * that waits for that monitor under the log's lock. Prints one line per check; exits with 1 if one failed.
 */
public final class LockCheck
{
    private LockCheck() {}

    public static void main(String[] args) throws Exception
    {
        SOCGame game = OSPBoard.newGame("lock-check");
        MoveLog log = new MoveLog(game, new OSPGame(game).state(-1));
        boolean ok = true;

        // The server changes the game under its monitor
        game.takeMonitor();
        ok &= waits("the game's monitor", game, log, game::releaseMonitor);

        // The move log handles a request under its own lock (the server's thread holds it while JSettlers handles it)
        Thread holder = new Thread(() ->
        {
            synchronized (log)
            {
                try
                {
                    Thread.sleep(800);
                }
                catch (InterruptedException e) {}
            }
        });
        holder.start();
        Thread.sleep(100);
        ok &= waits("the move log", game, log, () -> {});
        holder.join();

        ok &= destroysBesideAWaitingRead(game, log);
        System.exit(ok ? 0 : 1);
    }

    // Whether the game's destruction goes through while a read waits under the log's lock for the monitor the destroying
    // thread holds (here the main thread holds it for that thread)
    private static boolean destroysBesideAWaitingRead(SOCGame game, MoveLog log) throws Exception
    {
        game.takeMonitor();
        Thread reader = new Thread(() -> OSPServer.read(game, log, () -> null));
        reader.start();
        Thread.sleep(300);
        Thread destroyer = new Thread(log::gameDestroyed);
        destroyer.start();
        destroyer.join(3000);
        boolean free = ! destroyer.isAlive();
        game.releaseMonitor();
        reader.join(5000);
        destroyer.join(5000);
        System.out.println(free ? "destroyed beside a waiting read" : "the destruction waited for the read (a deadlock with the server)");
        return free && ! reader.isAlive();
    }

    // Whether a read started now is still waiting a moment later, and goes through once the lock is let go
    private static boolean waits(String lock, SOCGame game, MoveLog log, Runnable release) throws Exception
    {
        AtomicBoolean read = new AtomicBoolean();
        Thread reader = new Thread(() -> OSPServer.read(game, log, () ->
        {
            read.set(true);
            return null;
        }));
        reader.start();
        Thread.sleep(300);
        boolean waited = ! read.get();
        release.run();
        reader.join(5000);
        if (waited && read.get())
            System.out.println("waited for " + lock);
        else
            System.out.println("didn't wait for " + lock + " (read " + read.get() + ")");
        return waited && read.get();
    }
}
