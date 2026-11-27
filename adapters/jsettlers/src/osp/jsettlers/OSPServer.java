package osp.jsettlers;

import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Supplier;

import soc.game.SOCGame;
import soc.message.SOCMessageForGame;
import soc.server.GameMessageHandler;
import soc.server.SOCGameHandler;
import soc.server.SOCServer;
import soc.server.genericServer.Connection;

/**
 * JSettlers' server with Open Seat Agents among its robots: it starts JSettlers' server (SOCServer) with the given
 * properties and keeps it, so the Open Seat Agents' robots (OSPRobotClient, started by the server through
 * jsettlers.bots.start3p) can read the true game they play in. They run inside the server's process; a robot on another
 * machine would only see its seat's view. Every request the server handles for a game passes through MoveLog, which
 * keeps the game's moves as protocol steps (for agents that observe) and, with -Dosp.record=DIR, writes each game as a
 * referee log while it runs. JSettlers starts 30% of its own robots with its fast strategy and the rest with its smart
 * one (by name: "droid" and "robot"); -Dosp.bots=smart or fast gives all of them the one strategy.
 *
 * Usage: java -cp build:JSETTLERS_JAR -Dosp.agent="COMMAND" osp.jsettlers.OSPServer [-Dname=value ...] [port]
 * e.g. robot-only games with one robot of Lanzenbrecher's: -Djsettlers.startrobots=7
 * -Djsettlers.bots.start3p=2,osp.jsettlers.OSPRobotClient -Djsettlers.bots.percent3p=25 -Djsettlers.bots.botgames.total=100
 * -Djsettlers.bots.botgames.shutdown=Y -Djsettlers.bots.fast_pause_percent=10 -Djsettlers.gameopt.NT=t
 */
public final class OSPServer
{
    private static volatile SOCServer server;
    // The robots' brains in games, so a shutdown lets them tell their agents the game's end first
    static final Set<OSPRobotBrain> BRAINS = ConcurrentHashMap.newKeySet();
    // Each game's moves, from its first move on
    private static final Map<String, MoveLog> LOGS = new ConcurrentHashMap<>();

    private OSPServer() {}

    /** The true game of that name on this process's server, or null */
    public static SOCGame game(String name)
    {
        SOCServer s = server;
        return s == null ? null : s.getGame(name);
    }

    /** The move log of a game on this process's server, or null */
    static MoveLog log(String name)
    {
        return LOGS.get(name);
    }

    /**
     * The move log of a started game, begun at its first request or when a robot first looks at it, before any (the
     * caller holds neither the log's lock nor the game's monitor)
     */
    static MoveLog logOf(SOCGame game)
    {
        return LOGS.computeIfAbsent(game.getName(), name ->
        {
            game.takeMonitor();
            try
            {
                return new MoveLog(game, new OSPGame(game).state(-1));
            }
            finally
            {
                game.releaseMonitor();
            }
        });
    }

    /**
     * Reads the server's game under its locks: the move log's, so no request is half handled, and the game's own
     * monitor, which JSettlers takes while it changes the game (also outside requests, e.g. ending a slow player's turn).
     * The caller must hold neither (JSettlers' monitor isn't reentrant).
     */
    static <T> T read(SOCGame game, MoveLog log, Supplier<T> reader)
    {
        Object lock = log != null ? log : game;
        synchronized (lock)
        {
            game.takeMonitor();
            try
            {
                return reader.get();
            }
            finally
            {
                game.releaseMonitor();
            }
        }
    }

    /** JSettlers' server, letting the robots' brains finish a game before it is removed */
    private static final class Server extends SOCServer
    {
        Server(int port, Properties props) throws Exception
        {
            super(port, props);
        }

        @Override
        public void destroyGame(String name)
        {
            // The game's last move first, so the brains can hand it to their agents
            MoveLog log = LOGS.get(name);
            if (log != null)
                log.gameDestroyed();
            for (OSPRobotBrain brain : BRAINS)
                brain.gameEnding(name);
            LOGS.remove(name);
            super.destroyGame(name);
        }

        @Override
        protected SOCGameHandler buildGameHandler()
        {
            return new Handler(this);
        }
    }

    /** JSettlers' game handler, its requests passed through the games' move logs */
    private static final class Handler extends SOCGameHandler
    {
        private GameMessageHandler wrapped;

        Handler(SOCServer server)
        {
            super(server);
        }

        @Override
        public synchronized GameMessageHandler getMessageHandler()
        {
            if (wrapped == null)
            {
                final GameMessageHandler inner = super.getMessageHandler();
                wrapped = (SOCGame game, SOCMessageForGame message, Connection connection) -> MoveLog.dispatch(inner, game, message, connection);
            }
            return wrapped;
        }
    }

    public static void main(String[] args) throws Exception
    {
        Properties props = SOCServer.parseCmdline_DashedArgs(args);
        if (props == null)
            System.exit(1);
        int port = Integer.parseInt(props.getProperty(SOCServer.PROP_JSETTLERS_PORT, Integer.toString(SOCServer.SOC_PORT_DEFAULT)));
        // The run's robot-only games, for the stop rule's default
        FailureLimit.games = Integer.parseInt(props.getProperty("jsettlers.bots.botgames.total", "0").trim());
        // JSettlers' server exits right after its last robot-only game: the brains finish theirs first
        Runtime.getRuntime().addShutdownHook(new Thread(() ->
        {
            for (OSPRobotBrain brain : BRAINS)
                brain.finishOnShutdown();
        }, "osp-shutdown"));
        String bots = System.getProperty("osp.bots", "");
        if (bots.equals("smart"))
            SOCServer.ROBOT_PARAMS_DEFAULT = SOCServer.ROBOT_PARAMS_SMARTER;
        else if (bots.equals("fast"))
            SOCServer.ROBOT_PARAMS_SMARTER = SOCServer.ROBOT_PARAMS_DEFAULT;
        server = new Server(port, props);
        server.setPriority(5);
        server.start();
    }
}
