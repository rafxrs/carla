package osp.jsettlers;

import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * The run's stop rule. A game whose Open Seat Agent seat failed (a failure in its result record, or moves the referee
 * made for the seat: fallbacks) says nothing about the agent's strength, and many of them mean something is broken. Once
 * more such games ended than the system property osp.max_failed_games allows (default: 3, or one in a hundred of the
 * run's games, jsettlers.bots.botgames.total; negative: no limit), the server writes a line starting with STOPPED to
 * standard error and exits (code 3), so no more games start. Games still running then end without a result.
 */
final class FailureLimit
{
    /** The start of the line the server writes when it stops */
    static final String STOPPED = "osp: stopped:";
    private static final AtomicInteger FAILED = new AtomicInteger();
    private static final AtomicBoolean STOPPING = new AtomicBoolean();
    // The run's games, if known, for the default limit
    static volatile int games;

    private FailureLimit() {}

    /** The failed games the run allows (negative: any number) */
    static int limit()
    {
        String set = System.getProperty("osp.max_failed_games");
        if (set != null && ! set.isBlank())
            return Integer.parseInt(set.trim());
        return Math.max(3, games / 100);
    }

    /** Whether a result record's Open Seat Agent seat failed: a failure, or moves the referee made for it */
    @SuppressWarnings("unchecked")
    static boolean failed(Map<String, Object> result)
    {
        for (Object line : (List<Object>) result.get("seats"))
        {
            Map<String, Object> seat = (Map<String, Object>) line;
            // JSettlers' own robots have no agent and never fail
            if (seat.get("agent") == null)
                continue;
            if (! ((List<Object>) seat.get("failures")).isEmpty() || Json.integer(seat.get("fallbacks")) > 0)
                return true;
        }
        return false;
    }

    /** A game ended with this result record: counted if its Open Seat seat failed, and past the limit the run stops */
    static void ended(Map<String, Object> result)
    {
        if (! failed(result))
            return;
        int count = FAILED.incrementAndGet(), limit = limit();
        // Only the first game past the limit stops the run (games end on several threads)
        if (limit < 0 || count <= limit || ! STOPPING.compareAndSet(false, true))
            return;
        System.err.println(STOPPED + " " + count + " games ended with a failed Open Seat seat, more than osp.max_failed_games=" + limit + " allows");
        System.err.flush();
        // Exits on a thread of its own: the caller holds the game's locks, which the shutdown's hooks may need
        new Thread(() -> System.exit(3), "osp-stop").start();
    }
}
