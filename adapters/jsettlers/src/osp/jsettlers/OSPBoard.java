package osp.jsettlers;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import soc.game.SOCBoard;
import soc.game.SOCGame;

/**
 * JSettlers' classic board in the Open Seat Protocol's keys (see ../../PROTOCOL.md): hexes in axial coordinates [q, r]
 * (pointy-top, q to the east, r to the south-east, the center hex [0, 0]), an intersection as its three hexes (sea
 * included), a path as its two intersections. Everything follows from JSettlers' own adjacency: it numbers a hex's
 * corners clockwise from north, so the neighbor across each side is the hex that shares those two corners, and walking
 * from the center hex gives every land hex its axial coordinate; a corner's key is its hex and the two neighbors on
 * either side of it. No coordinate tables are written down.
 */
public final class OSPBoard
{
    // The protocol's axial steps: east, south-east, south-west, west, north-west, north-east
    private static final int[][] STEPS = { {1, 0}, {0, 1}, {-1, 1}, {-1, 0}, {0, -1}, {1, -1} };
    // The side towards each step lies between two of a hex's corners (clockwise from north: 0 north, 1 north-east, ...);
    // the neighbor there has the same two corners as its corners on the opposite side
    private static final int[][] SIDE_CORNERS = { {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 0}, {0, 1} };
    private static final int[][] FACING_CORNERS = { {5, 4}, {0, 5}, {1, 0}, {2, 1}, {3, 2}, {4, 3} };
    // A corner lies between the neighbors of two steps: north between north-west and north-east, and so on clockwise
    private static final int[][] CORNER_STEPS = { {4, 5}, {5, 0}, {0, 1}, {1, 2}, {2, 3}, {3, 4} };

    // SGN's tile codes by JSettlers' hex type, and its harbor codes by port type
    private static final Map<Integer, String> TILE_CODES = new HashMap<>();
    private static final Map<Integer, String> HARBOR_CODES = new HashMap<>();
    static
    {
        TILE_CODES.put(SOCBoard.CLAY_HEX, "H");
        TILE_CODES.put(SOCBoard.ORE_HEX, "M");
        TILE_CODES.put(SOCBoard.SHEEP_HEX, "P");
        TILE_CODES.put(SOCBoard.WHEAT_HEX, "F");
        TILE_CODES.put(SOCBoard.WOOD_HEX, "FO");
        TILE_CODES.put(SOCBoard.DESERT_HEX, "D");
        HARBOR_CODES.put(SOCBoard.MISC_PORT, "HR");
        HARBOR_CODES.put(SOCBoard.CLAY_PORT, "HRB");
        HARBOR_CODES.put(SOCBoard.ORE_PORT, "HRO");
        HARBOR_CODES.put(SOCBoard.SHEEP_PORT, "HRW");
        HARBOR_CODES.put(SOCBoard.WHEAT_PORT, "HRG");
        HARBOR_CODES.put(SOCBoard.WOOD_PORT, "HRL");
    }

    public final SOCBoard board;
    private final int[] landHexes;
    // Axial coordinates of every land hex, and back
    private final Map<Integer, int[]> axial = new HashMap<>();
    private final Map<String, Integer> hexAt = new HashMap<>();
    // Every land node's key and every land edge's path, as canonical JSON text, and back
    private final Map<Integer, List<Object>> nodeKeys = new TreeMap<>();
    private final Map<String, Integer> nodeAt = new HashMap<>();
    private final Map<Integer, List<Object>> edgePaths = new TreeMap<>();
    private final Map<String, Integer> edgeAt = new HashMap<>();

    public OSPBoard(SOCBoard board)
    {
        this.board = board;
        landHexes = board.getLandHexCoords();

        // The center: the land hex whose six neighbors are all land
        Map<Integer, Integer[]> neighbors = new HashMap<>();
        for (int hex : landHexes)
        {
            Integer[] around = new Integer[6];
            for (int s = 0; s < 6; s++)
                for (int other : landHexes)
                    if (other != hex
                        && board.getAdjacentNodeToHex(other, FACING_CORNERS[s][0]) == board.getAdjacentNodeToHex(hex, SIDE_CORNERS[s][0])
                        && board.getAdjacentNodeToHex(other, FACING_CORNERS[s][1]) == board.getAdjacentNodeToHex(hex, SIDE_CORNERS[s][1]))
                        around[s] = other;
            neighbors.put(hex, around);
        }
        Integer center = null;
        for (int hex : landHexes)
        {
            boolean inner = true;
            for (Integer n : neighbors.get(hex))
                inner &= n != null;
            boolean innermost = inner;
            if (inner)
                for (Integer n : neighbors.get(hex))
                    for (Integer nn : neighbors.get(n))
                        innermost &= nn != null;
            if (innermost)
                center = hex;
        }
        if (center == null)
            throw new IllegalArgumentException("Not a classic board: no hex with land all around");

        // Walk out from the center, one step per side
        List<Integer> agenda = new ArrayList<>();
        agenda.add(center);
        axial.put(center, new int[] {0, 0});
        while (! agenda.isEmpty())
        {
            int hex = agenda.remove(0);
            int[] at = axial.get(hex);
            Integer[] around = neighbors.get(hex);
            for (int s = 0; s < 6; s++)
                if (around[s] != null && ! axial.containsKey(around[s]))
                {
                    axial.put(around[s], new int[] {at[0] + STEPS[s][0], at[1] + STEPS[s][1]});
                    agenda.add(around[s]);
                }
        }
        if (axial.size() != landHexes.length)
            throw new IllegalArgumentException("Not a classic board: " + (landHexes.length - axial.size()) + " land hexes out of reach");
        for (Map.Entry<Integer, int[]> entry : axial.entrySet())
            hexAt.put(key(entry.getValue()), entry.getKey());

        // A corner's key: its hex and the neighbors on either side of the corner
        for (int hex : landHexes)
            for (int c = 0; c < 6; c++)
            {
                int node = board.getAdjacentNodeToHex(hex, c);
                if (nodeKeys.containsKey(node))
                    continue;
                int[] at = axial.get(hex);
                List<int[]> hexes = new ArrayList<>();
                hexes.add(at);
                for (int s : CORNER_STEPS[c])
                    hexes.add(new int[] {at[0] + STEPS[s][0], at[1] + STEPS[s][1]});
                hexes.sort(HEX_ORDER);
                List<Object> nodeKey = new ArrayList<>();
                for (int[] h : hexes)
                    nodeKey.add(hexJson(h));
                nodeKeys.put(node, nodeKey);
                nodeAt.put(Json.write(nodeKey), node);
            }

        // A side of a land hex: the edge both its corners touch
        for (int hex : landHexes)
            for (int s = 0; s < 6; s++)
            {
                int a = board.getAdjacentNodeToHex(hex, SIDE_CORNERS[s][0]), b = board.getAdjacentNodeToHex(hex, SIDE_CORNERS[s][1]);
                Integer edge = null;
                for (int e : board.getAdjacentEdgesToNode(a))
                    if (board.getAdjacentNodesToEdge(e).contains(b))
                        edge = e;
                if (edge == null)
                    throw new IllegalArgumentException("No edge between nodes " + a + " and " + b);
                if (edgePaths.containsKey(edge))
                    continue;
                List<Object> path = pathOf(a, b);
                edgePaths.put(edge, path);
                edgeAt.put(Json.write(path), edge);
            }
    }

    private static final Comparator<int[]> HEX_ORDER = (x, y) -> x[0] != y[0] ? Integer.compare(x[0], y[0]) : Integer.compare(x[1], y[1]);

    private static String key(int[] hex)
    {
        return hex[0] + "," + hex[1];
    }

    private static List<Object> hexJson(int[] hex)
    {
        return Json.array((long) hex[0], (long) hex[1]);
    }

    // A path: its two intersections, sorted by their hexes
    private List<Object> pathOf(int a, int b)
    {
        List<Object> ka = nodeKeys.get(a), kb = nodeKeys.get(b);
        List<Object> path = new ArrayList<>();
        if (compareKeys(ka, kb) <= 0)
        {
            path.add(ka);
            path.add(kb);
        }
        else
        {
            path.add(kb);
            path.add(ka);
        }
        return path;
    }

    // Orders keys as the protocol sorts them: hex by hex, numbers by value
    @SuppressWarnings("unchecked")
    private static int compareKeys(List<Object> a, List<Object> b)
    {
        for (int i = 0; i < Math.min(a.size(), b.size()); i++)
        {
            List<Object> x = (List<Object>) a.get(i), y = (List<Object>) b.get(i);
            for (int j = 0; j < 2; j++)
            {
                int order = Long.compare(((Number) x.get(j)).longValue(), ((Number) y.get(j)).longValue());
                if (order != 0)
                    return order;
            }
        }
        return Integer.compare(a.size(), b.size());
    }

    /** The protocol's hex [q, r] of a land hex */
    public List<Object> hex(int hexCoord)
    {
        return hexJson(axial.get(hexCoord));
    }

    /** JSettlers' coordinate of the protocol's land hex, or -1 */
    public int hexCoord(List<?> hex)
    {
        Integer coord = hexAt.get(Json.integer(hex.get(0)) + "," + Json.integer(hex.get(1)));
        return coord == null ? -1 : coord;
    }

    /** The protocol's intersection of a land node */
    public List<Object> intersection(int node)
    {
        return nodeKeys.get(node);
    }

    /** JSettlers' node of an intersection (its three hexes in any order), or -1 */
    @SuppressWarnings("unchecked")
    public int node(Object key)
    {
        List<Object> sorted = new ArrayList<>((List<Object>) key);
        sorted.sort((x, y) -> compareKeys(Json.array(x), Json.array(y)));
        List<Object> normal = new ArrayList<>();
        for (Object h : sorted)
            normal.add(Json.array((long) Json.integer(((List<Object>) h).get(0)), (long) Json.integer(((List<Object>) h).get(1))));
        Integer node = nodeAt.get(Json.write(normal));
        return node == null ? -1 : node;
    }

    /** The protocol's path of a land edge */
    public List<Object> path(int edge)
    {
        return edgePaths.get(edge);
    }

    /** JSettlers' edge of a path (its two intersections in any order), or -1 */
    @SuppressWarnings("unchecked")
    public int edge(Object path)
    {
        List<Object> ends = (List<Object>) path;
        int a = node(ends.get(0)), b = node(ends.get(1));
        if (a < 0 || b < 0)
            return -1;
        Integer edge = edgeAt.get(Json.write(pathOf(a, b)));
        return edge == null ? -1 : edge;
    }

    /** Every land node and land edge */
    public Iterable<Integer> nodes()
    {
        return nodeKeys.keySet();
    }

    public Iterable<Integer> edges()
    {
        return edgePaths.keySet();
    }

    /** The protocol's board: every land hex with its SGN code and number, the sea ring, the harbors, the robber */
    public Map<String, Object> boardJson()
    {
        List<int[]> hexes = new ArrayList<>(axial.values());
        for (int[] sea : seaRing())
            hexes.add(sea);
        hexes.sort(HEX_ORDER);
        List<Object> tiles = new ArrayList<>();
        for (int[] h : hexes)
        {
            Integer coord = hexAt.get(key(h));
            if (coord == null)
                tiles.add(Json.object("tile", hexJson(h), "tCode", "S", "rNumber", null));
            else
            {
                int type = board.getHexTypeFromCoord(coord);
                int number = board.getNumberOnHexFromCoord(coord);
                tiles.add(Json.object("tile", hexJson(h), "tCode", TILE_CODES.get(type), "rNumber", type == SOCBoard.DESERT_HEX || number <= 0 ? null : (Object) (long) number));
            }
        }
        List<Object> harbors = new ArrayList<>();
        for (Object[] harbor : harbors())
            harbors.add(Json.object("path", harbor[0], "hCode", harbor[1]));
        return Json.object("tiles", tiles, "harbors", harbors, "robber", hex(board.getRobberHex()));
    }

    // The sea around the land: every hex 3 steps from the center
    private static List<int[]> seaRing()
    {
        List<int[]> sea = new ArrayList<>();
        for (int q = -3; q <= 3; q++)
            for (int r = -3; r <= 3; r++)
                if (Math.max(Math.abs(q), Math.max(Math.abs(r), Math.abs(q + r))) == 3)
                    sea.add(new int[] {q, r});
        return sea;
    }

    // Each harbor's path and SGN code, from JSettlers' port edges and the port type at their nodes
    private List<Object[]> harbors()
    {
        List<Object[]> result = new ArrayList<>();
        for (int edge : board.getPortsEdges())
        {
            if (edge < 0)
                continue;
            List<Integer> ends = board.getAdjacentNodesToEdge(edge);
            result.add(new Object[] {pathOf(ends.get(0), ends.get(1)), HARBOR_CODES.get(board.getPortTypeFromNodeCoord(ends.get(0)))});
        }
        return result;
    }

    /** The board's shape as the protocol's reference file holds it (../../osp/classic-board.json) */
    public Map<String, Object> shape()
    {
        List<Object> land = new ArrayList<>(), sea = new ArrayList<>(), intersections = new ArrayList<>(), paths = new ArrayList<>(), harbors = new ArrayList<>();
        for (int[] h : axial.values())
            land.add(hexJson(h));
        for (int[] h : seaRing())
            sea.add(hexJson(h));
        intersections.addAll(nodeKeys.values());
        paths.addAll(edgePaths.values());
        for (Object[] harbor : harbors())
            harbors.add(harbor[0]);
        return Json.object("land", land, "sea", sea, "intersections", intersections, "paths", paths, "harbors", harbors);
    }

    /** A classic game with a new random board, the four seats taken */
    public static SOCGame newGame(String name)
    {
        SOCGame game = new SOCGame(name);
        for (int seat = 0; seat < 4; seat++)
            game.addPlayer("seat" + seat, seat);
        game.startGame();
        return game;
    }

    /**
     * describe SEED: a new classic board with its shape, the protocol's board, every land hex with its type, number and
     * corners' keys clockwise from north, and the ports with their types (for the tests)
     */
    public static void main(String[] args)
    {
        if (args.length < 1 || ! args[0].equals("describe"))
        {
            System.err.println("Usage: OSPBoard describe SEED");
            System.exit(2);
        }
        OSPBoard osp = new OSPBoard(newGame("describe-" + (args.length > 1 ? args[1] : "0")).getBoard());
        List<Object> hexes = new ArrayList<>();
        for (int hex : osp.landHexes)
        {
            List<Object> corners = new ArrayList<>();
            for (int c = 0; c < 6; c++)
                corners.add(osp.intersection(osp.board.getAdjacentNodeToHex(hex, c)));
            hexes.add(Json.object("hex", osp.hex(hex), "type", (long) osp.board.getHexTypeFromCoord(hex),
                "number", (long) osp.board.getNumberOnHexFromCoord(hex), "corners", corners));
        }
        List<Object> ports = new ArrayList<>();
        for (int edge : osp.board.getPortsEdges())
            if (edge >= 0)
                ports.add(Json.object("edge", (long) edge, "type", (long) osp.board.getPortTypeFromNodeCoord(osp.board.getAdjacentNodesToEdge(edge).get(0))));
        System.out.println(Json.write(Json.object("shape", osp.shape(), "board", osp.boardJson(), "hexes", hexes, "ports", ports)));
    }
}
