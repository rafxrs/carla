package osp.jsettlers;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import soc.game.ResourceSet;
import soc.game.SOCBoard;
import soc.game.SOCCity;
import soc.game.SOCDevCardConstants;
import soc.game.SOCGame;
import soc.game.SOCInventory;
import soc.game.SOCMoveRobberResult;
import soc.game.SOCPlayer;
import soc.game.SOCPlayingPiece;
import soc.game.SOCResourceConstants;
import soc.game.SOCResourceSet;
import soc.game.SOCRoad;
import soc.game.SOCRoutePiece;
import soc.game.SOCSettlement;
import soc.game.SOCTradeOffer;

/**
 * A JSettlers game (its SOCGame, which holds JSettlers' rules) in the Open Seat Protocol (see ../../PROTOCOL.md): its
 * state as the referee or a seat sees it, the legal moves of the seat that acts, and playing a move with its result.
 * The legal moves are the protocol's moves JSettlers' own checks allow (SOCGame's and SOCPlayer's can... methods), one
 * protocol move where JSettlers takes two steps: a robber move with its victim, a Year of Plenty or Monopoly card with its
 * resources, a bought road, settlement or city with its place. JSettlers' players discard at the same time after a 7;
 * the protocol asks them one after another, from the lowest seat. JSettlers keeps no resource bank: the bank is what the
 * hands leave of 19 cards a resource. JSettlers draws who starts; the protocol's seats are the turn order from the first
 * player (seat 0 starts), so a JSettlers player number becomes seatOf it.
 */
public final class OSPGame
{
    // The protocol's resource names by JSettlers' resource type, and back
    private static final String[] RESOURCE_NAMES = new String[SOCResourceConstants.WOOD + 1];
    static
    {
        RESOURCE_NAMES[SOCResourceConstants.CLAY] = "brick";
        RESOURCE_NAMES[SOCResourceConstants.ORE] = "ore";
        RESOURCE_NAMES[SOCResourceConstants.SHEEP] = "wool";
        RESOURCE_NAMES[SOCResourceConstants.WHEAT] = "grain";
        RESOURCE_NAMES[SOCResourceConstants.WOOD] = "lumber";
    }
    private static final int[] RESOURCES = {SOCResourceConstants.CLAY, SOCResourceConstants.ORE, SOCResourceConstants.SHEEP,
        SOCResourceConstants.WHEAT, SOCResourceConstants.WOOD};

    public final SOCGame game;
    public final OSPBoard board;

    // JSettlers' player number of seat 0
    public final int first;

    public OSPGame(SOCGame game)
    {
        this.game = game;
        this.board = new OSPBoard(game.getBoard());
        this.first = game.getFirstPlayer();
    }

    /** The protocol's seat of a JSettlers player number: its place in the turn order from the first player */
    public int seatOf(int player)
    {
        return (player - first + game.maxPlayers) % game.maxPlayers;
    }

    /** JSettlers' player number of a protocol seat */
    public int playerOf(int seat)
    {
        return (seat + first) % game.maxPlayers;
    }

    public static String resourceName(int type)
    {
        return RESOURCE_NAMES[type];
    }

    public static int resourceType(String name)
    {
        for (int type : RESOURCES)
            if (RESOURCE_NAMES[type].equals(name))
                return type;
        throw new IllegalArgumentException("No resource " + name);
    }

    /** The protocol's name of a development card type (every victory point card is one) */
    public static String cardName(int type)
    {
        switch (type)
        {
        case SOCDevCardConstants.KNIGHT: return "knight";
        case SOCDevCardConstants.ROADS: return "road_building";
        case SOCDevCardConstants.DISC: return "year_of_plenty";
        case SOCDevCardConstants.MONO: return "monopoly";
        default: return "victory_point";
        }
    }

    public boolean isOver()
    {
        return game.getGameState() >= SOCGame.OVER;
    }

    /**
     * The seat that acts now: the lowest one that has to discard; with an open offer the next addressed seat in seat order
     * that hasn't answered, then the offerer (as Lanzenbrecher orders them; under first_accept the others may answer first);
     * else the current player; -1 once the game is over
     */
    public int acting()
    {
        if (isOver())
            return -1;
        if (game.getGameState() == SOCGame.WAITING_FOR_DISCARDS)
            for (int seat = 0; seat < game.maxPlayers; seat++)
                if (game.getPlayer(playerOf(seat)).getNeedToDiscard())
                    return seat;
        Map<String, Object> offer = offer();
        if (offer != null)
        {
            int from = Json.integer(offer.get("from"));
            for (int i = 1; i < game.maxPlayers; i++)
            {
                int seat = (from + i) % game.maxPlayers;
                if (pending(offer, seat))
                    return seat;
            }
            return from;
        }
        return seatOf(game.getCurrentPlayerNumber());
    }

    // The offer JSettlers holds (its object: JSettlers makes one per offer) and the answers the protocol keeps for it: an
    // answer stays what it was first (JSettlers lets a counteroffer be withdrawn after a while, or a decline be followed by
    // an offer; the protocol's answers are final), until the offer closes or another one replaces it. The addressees' own
    // offers that stood when it came are no counteroffers to it (JSettlers keeps an offer until its maker's turn or a
    // trade clears it).
    private SOCTradeOffer answeredOffer;
    private Map<String, Object> answeredTerms;
    private final Map<Integer, String> answers = new HashMap<>();
    private final Map<Integer, Map<String, Object>> counterTerms = new HashMap<>();
    // The counterers' own offers (JSettlers' objects) that are their counteroffers to it, and what the addressees offered
    // before it
    private final Map<Integer, SOCTradeOffer> counterOffers = new HashMap<>(), earlierOffers = new HashMap<>();
    // An offer of the player on turn the protocol has closed though JSettlers still holds it: withdrawn when the player went
    // on (closeOffer), or made before the roll (the protocol has no offers then; a robot's late counteroffer becomes one);
    // and whether the protocol closed the offer answered so far (then its counteroffers don't keep it open)
    private SOCTradeOffer closedOffer;
    private boolean answeredClosed;

    /**
     * The protocol withdrew the open offer (its offerer went on with it standing): it is withdrawn at JSettlers too, so a
     * late accept finds no offer there (JSettlers would otherwise trade on it); called under the game's monitor while the
     * server handles the offerer's request
     */
    public synchronized void closeOffer()
    {
        int current = game.getCurrentPlayerNumber();
        if (current >= 0 && game.getPlayer(current).getCurrentOffer() != null)
        {
            closedOffer = game.getPlayer(current).getCurrentOffer();
            game.getPlayer(current).setCurrentOffer(null);
        }
        answeredClosed = true;
    }

    // No offer is open for the protocol: the answers are forgotten
    private void forget()
    {
        answeredOffer = null;
        answeredTerms = null;
        answeredClosed = false;
        answers.clear();
        counterTerms.clear();
        counterOffers.clear();
        earlierOffers.clear();
    }

    // Whether a counteroffer to the offer answered so far still stands at JSettlers (its maker hasn't withdrawn it)
    private boolean counterStanding()
    {
        for (Map.Entry<Integer, SOCTradeOffer> counter : counterOffers.entrySet())
            if (game.getPlayer(playerOf(counter.getKey())).getCurrentOffer() == counter.getValue())
                return true;
        return false;
    }

    /**
     * The open offer as the protocol has it under player_trading "first_accept": the standing offer of the player on turn
     * (after the roll), each addressed seat's answer (none while pending, "decline", or "counter" with the terms of its own
     * offer to the player on turn, written in the offerer's view), null without one or once every addressed seat declined
     * (the protocol closes it then; JSettlers keeps it until it is withdrawn)
     */
    public synchronized Map<String, Object> offer()
    {
        int current = game.getCurrentPlayerNumber();
        SOCTradeOffer standing = isOver() || current < 0 ? null : game.getPlayer(current).getCurrentOffer();
        // An offer standing before the roll stays closed for the protocol
        if (standing != null && game.getGameState() != SOCGame.PLAY1)
            closedOffer = standing;
        if (isOver() || current < 0 || game.getGameState() != SOCGame.PLAY1)
        {
            forget();
            return null;
        }
        if (standing == null || standing == closedOffer)
        {
            // The offerer withdrew its offer (a JSettlers robot does when a counteroffer comes, then may take that one, which
            // JSettlers keeps standing): for the protocol the counteroffers belong to the offer, which stays open while one
            // of them stands, unless the protocol closed it
            if (answeredOffer != null && ! answeredClosed && Json.integer(answeredTerms.get("from")) == seatOf(current) && counterStanding())
                return written();
            forget();
            return null;
        }
        int from = seatOf(current);
        List<Object> to = new ArrayList<>();
        boolean[] addressed = standing.getTo();
        for (int seat = 0; seat < game.maxPlayers; seat++)
            if (seat != from && addressed[playerOf(seat)])
                to.add((long) seat);
        Map<String, Object> identity = Json.object("from", (long) from, "give", cardSet(standing.getGiveSet()), "get", cardSet(standing.getGetSet()), "to", to);
        // Another offer than the one answered so far: its answers start anew, and what the addressees offered before it is
        // no answer to it
        if (standing != answeredOffer)
        {
            forget();
            answeredOffer = standing;
            answeredTerms = identity;
            for (Object s : to)
                earlierOffers.put(Json.integer(s), game.getPlayer(playerOf(Json.integer(s))).getCurrentOffer());
        }
        boolean[] waiting = standing.getWaitingReply();
        for (Object s : to)
        {
            int seat = Json.integer(s), pn = playerOf(seat);
            if (answers.containsKey(seat))
                continue;
            // An addressee's own offer to the player on turn is its counteroffer: in the offerer's terms the offerer gives
            // what the counterer asks for and gets what it gives
            SOCTradeOffer own = game.getPlayer(pn).getCurrentOffer();
            if (own != null && own != earlierOffers.get(seat) && own.getTo()[current])
            {
                answers.put(seat, "counter");
                counterTerms.put(seat, Json.object("give", cardSet(own.getGetSet()), "get", cardSet(own.getGiveSet())));
                counterOffers.put(seat, own);
            }
            // JSettlers clears the reply flag when the addressee rejects
            else if (! waiting[pn])
                answers.put(seat, "decline");
        }
        // Everybody declined: closed, as the protocol has it
        boolean open = false;
        for (Object s : to)
            open |= ! "decline".equals(answers.get(Json.integer(s)));
        return open ? written() : null;
    }

    // The offer answered so far as the protocol writes it: its terms, the answers, the counteroffers
    private Map<String, Object> written()
    {
        Map<String, Object> answerMap = new TreeMap<>(), counters = new TreeMap<>();
        for (Map.Entry<Integer, String> answer : answers.entrySet())
            answerMap.put(Integer.toString(answer.getKey()), answer.getValue());
        for (Map.Entry<Integer, Map<String, Object>> counter : counterTerms.entrySet())
            counters.put(Integer.toString(counter.getKey()), counter.getValue());
        Map<String, Object> offer = Json.object("from", answeredTerms.get("from"), "give", answeredTerms.get("give"), "get", answeredTerms.get("get"),
            "to", answeredTerms.get("to"), "answers", answerMap);
        if (! counters.isEmpty())
            offer.put("counters", counters);
        return offer;
    }

    // Whether the seat is addressed by the offer and hasn't answered
    @SuppressWarnings("unchecked")
    static boolean pending(Map<String, Object> offer, int seat)
    {
        return ((List<Object>) offer.get("to")).contains((long) seat) && ! ((Map<String, Object>) offer.get("answers")).containsKey(Integer.toString(seat));
    }

    /** The answers the seat may give to the open offer while it is asked: accept if it holds what the offer asks for, decline */
    public List<Object> answerMoves(int seat)
    {
        List<Object> moves = new ArrayList<>();
        Map<String, Object> offer = offer();
        if (offer == null || ! pending(offer, seat))
            return moves;
        if (game.getPlayer(playerOf(seat)).getResources().contains(resourceSet(offer.get("get"))))
            moves.add(move("respond", Json.object("accept", true)));
        moves.add(move("respond", Json.object("accept", false)));
        return moves;
    }

    /** The offerer's picks: trading on a standing counteroffer both sides still hold the cards for, or withdrawing */
    @SuppressWarnings("unchecked")
    public List<Object> pickMoves(int seat)
    {
        List<Object> moves = new ArrayList<>();
        Map<String, Object> offer = offer();
        if (offer == null || Json.integer(offer.get("from")) != seat)
            return moves;
        Map<String, Object> counters = (Map<String, Object>) offer.get("counters");
        if (counters != null)
            for (Map.Entry<String, Object> counter : counters.entrySet())
            {
                int partner = Integer.parseInt(counter.getKey());
                Map<String, Object> terms = (Map<String, Object>) counter.getValue();
                // JSettlers trades on the counterer's standing offer: it may have been withdrawn since
                SOCTradeOffer own = game.getPlayer(playerOf(partner)).getCurrentOffer();
                if (own != null && game.getPlayer(playerOf(seat)).getResources().contains(resourceSet(terms.get("give")))
                    && game.getPlayer(playerOf(partner)).getResources().contains(resourceSet(terms.get("get"))))
                    moves.add(move("confirm", Json.object("partner", (long) partner)));
            }
        moves.add(move("cancel", null));
        return moves;
    }

    /** The protocol's prompt: what the seat that acts has to do */
    public String prompt()
    {
        Map<String, Object> offer = offer();
        if (offer != null)
        {
            for (int seat = 0; seat < game.maxPlayers; seat++)
                if (pending(offer, seat))
                    return "answer_offer";
            return "pick_partner";
        }
        switch (game.getGameState())
        {
        case SOCGame.START1A: case SOCGame.START2A: return "setup_settlement";
        case SOCGame.START1B: case SOCGame.START2B: return "setup_road";
        case SOCGame.ROLL_OR_CARD: return "roll";
        case SOCGame.WAITING_FOR_DISCARDS: return "discard";
        case SOCGame.PLACING_ROBBER: case SOCGame.WAITING_FOR_ROBBER_OR_PIRATE: return "move_robber";
        case SOCGame.PLACING_FREE_ROAD1: case SOCGame.PLACING_FREE_ROAD2: return "road_building";
        default: return isOver() ? "over" : "main";
        }
    }

    /**
     * Whether the player on turn has a free road of a Road Building card to place but no place for it, which JSettlers
     * leaves to the player to give up (road_building forced places it only if there is room)
     */
    public boolean strandedFreeRoad()
    {
        int state = game.getGameState();
        return (state == SOCGame.PLACING_FREE_ROAD1 || state == SOCGame.PLACING_FREE_ROAD2)
            && roadSites(game.getPlayer(game.getCurrentPlayerNumber())).isEmpty();
    }

    // The free roads of a Road Building card still to place, as many as the player has roads and places for
    private int freeRoads()
    {
        int state = game.getGameState();
        if (state != SOCGame.PLACING_FREE_ROAD1 && state != SOCGame.PLACING_FREE_ROAD2)
            return 0;
        SOCPlayer player = game.getPlayer(game.getCurrentPlayerNumber());
        int free = Math.min(state == SOCGame.PLACING_FREE_ROAD1 ? 2 : 1, player.getNumPieces(SOCPlayingPiece.ROAD));
        return roadSites(player).isEmpty() ? 0 : free;
    }

    private Map<String, Object> cards(SOCResourceSet set)
    {
        Map<String, Object> cards = new LinkedHashMap<>();
        for (int type : RESOURCES)
            cards.put(RESOURCE_NAMES[type], (long) set.getAmount(type));
        return cards;
    }

    /**
     * A state's hash as the protocol's decisions carry it: the first 16 hex digits of the SHA-256 of its canonical JSON
     * (keys sorted, no whitespace, UTF-8)
     */
    public static String hash(Object state)
    {
        try
        {
            byte[] digest = java.security.MessageDigest.getInstance("SHA-256").digest(Json.write(state, true).getBytes(java.nio.charset.StandardCharsets.UTF_8));
            StringBuilder hex = new StringBuilder();
            for (int i = 0; i < 8; i++)
                hex.append(String.format("%02x", digest[i]));
            return hex.toString();
        }
        catch (java.security.NoSuchAlgorithmException e)
        {
            throw new IllegalStateException(e);
        }
    }

    /** A card set of only the cards there, for moves, from JSettlers' resources */
    public static Map<String, Object> cardSet(ResourceSet set)
    {
        int[] counts = new int[SOCResourceConstants.WOOD + 1];
        for (int type : RESOURCES)
            counts[type] = set.getAmount(type);
        return cardSet(counts);
    }

    /** A card set of only the cards there, for moves */
    public static Map<String, Object> cardSet(int[] counts)
    {
        Map<String, Object> cards = new LinkedHashMap<>();
        for (int type : RESOURCES)
            if (counts[type] > 0)
                cards.put(RESOURCE_NAMES[type], (long) counts[type]);
        return cards;
    }

    @SuppressWarnings("unchecked")
    public static SOCResourceSet resourceSet(Object set)
    {
        int[] counts = new int[SOCResourceConstants.WOOD + 1];
        for (Map.Entry<String, Object> card : ((Map<String, Object>) set).entrySet())
            counts[resourceType(card.getKey())] += Json.integer(card.getValue());
        return new SOCResourceSet(counts[SOCResourceConstants.CLAY], counts[SOCResourceConstants.ORE], counts[SOCResourceConstants.SHEEP],
            counts[SOCResourceConstants.WHEAT], counts[SOCResourceConstants.WOOD], 0);
    }

    /**
     * The state in the protocol's shape: as the referee sees it (viewer below 0), or as a seat does (other seats'
     * hands, development cards and points with victory point cards null)
     */
    public Map<String, Object> state(int viewer)
    {
        List<Object> settlements = new ArrayList<>(), cities = new ArrayList<>(), roads = new ArrayList<>(), players = new ArrayList<>();
        int[] held = new int[SOCResourceConstants.WOOD + 1];
        SOCPlayer longest = game.getPlayerWithLongestRoad(), largest = game.getPlayerWithLargestArmy();
        for (int seat = 0; seat < game.maxPlayers; seat++)
        {
            SOCPlayer player = game.getPlayer(playerOf(seat));
            for (SOCPlayingPiece piece : player.getSettlements())
                settlements.add(Json.array((long) seat, board.intersection(piece.getCoordinates())));
            for (SOCPlayingPiece piece : player.getCities())
                cities.add(Json.array((long) seat, board.intersection(piece.getCoordinates())));
            for (SOCRoutePiece piece : player.getRoadsAndShips())
                roads.add(Json.array((long) seat, board.path(piece.getCoordinates())));
            SOCResourceSet hand = player.getResources();
            for (int type : RESOURCES)
                held[type] += hand.getAmount(type);
            Map<String, Object> development = new LinkedHashMap<>();
            for (String name : new String[] {"knight", "road_building", "year_of_plenty", "monopoly", "victory_point"})
                development.put(name, 0L);
            SOCInventory inventory = player.getInventory();
            int developmentSize = 0;
            for (int type = SOCDevCardConstants.MIN_KNOWN; type < SOCDevCardConstants.MAXPLUSONE; type++)
            {
                int count = inventory.getAmount(type);
                developmentSize += count;
                development.put(cardName(type), (Long) development.get(cardName(type)) + count);
            }
            Map<String, Object> entry = new LinkedHashMap<>();
            entry.put("seat", (long) seat);
            entry.put("hand_size", (long) hand.getTotal());
            entry.put("development_size", (long) developmentSize);
            entry.put("public_vp", (long) player.getPublicVP());
            entry.put("vp", (long) player.getTotalVP());
            entry.put("hand", cards(hand));
            entry.put("development", development);
            entry.put("played", Json.object("knight", (long) player.getNumKnights()));
            entry.put("longest_road", (long) player.getLongestRoadLength());
            entry.put("has_longest_road", longest == player);
            entry.put("has_largest_army", largest == player);
            entry.put("pieces_left", Json.object("roads", (long) player.getNumPieces(SOCPlayingPiece.ROAD),
                "settlements", (long) player.getNumPieces(SOCPlayingPiece.SETTLEMENT), "cities", (long) player.getNumPieces(SOCPlayingPiece.CITY)));
            players.add(entry);
        }
        // JSettlers has no resource bank (its ruleset says unlimited): the counts stop at 0, though it still pays
        Map<String, Object> bank = new LinkedHashMap<>();
        for (int type : RESOURCES)
            bank.put(RESOURCE_NAMES[type], (long) Math.max(0, 19 - held[type]));
        bank.put("development", (long) game.getNumDevCards());

        Map<String, Object> state = new LinkedHashMap<>();
        state.put("vp_to_win", (long) game.vp_winner);
        state.put("prompt", prompt());
        // Who acts: a player discarding after a 7 may be another than the one on turn
        int acting = acting();
        state.put("current", (long) (acting >= 0 ? acting : seatOf(game.getCurrentPlayerNumber())));
        state.put("on_turn", (long) seatOf(game.getCurrentPlayerNumber()));
        state.put("free_roads", (long) freeRoads());
        state.put("robber", board.hex(game.getBoard().getRobberHex()));
        state.put("buildings", Json.object("settlements", settlements, "cities", cities, "roads", roads));
        state.put("bank", bank);
        state.put("players", players);
        state.put("offer", offer());
        return viewer < 0 ? state : hide(state, viewer);
    }

    // Where the player may build a road now: every land edge JSettlers calls a potential road
    private List<Integer> roadSites(SOCPlayer player)
    {
        List<Integer> sites = new ArrayList<>();
        for (int edge : board.edges())
            if (player.isPotentialRoad(edge))
                sites.add(edge);
        return sites;
    }

    // The seats with a building at the hex who could be robbed there: not the robber, and holding cards
    private List<Integer> victims(int hex, int robberSeat)
    {
        int robber = playerOf(robberSeat);
        List<Integer> victims = new ArrayList<>();
        SOCBoard b = game.getBoard();
        for (int node : b.getAdjacentNodesToHex(hex))
        {
            SOCPlayingPiece piece = b.settlementAtNode(node);
            if (piece == null)
                continue;
            int owner = piece.getPlayerNumber();
            if (owner != robber && ! victims.contains(seatOf(owner)) && game.getPlayer(owner).getResources().getTotal() > 0)
                victims.add(seatOf(owner));
        }
        victims.sort(null);
        return victims;
    }

    private static List<Object> move(String type, Object value)
    {
        return value == null ? Json.array(type) : Json.array(type, value);
    }

    /** The legal moves of the seat (which has to be the one that acts), as JSettlers' rules allow them */
    public List<Object> legal(int seat)
    {
        List<Object> legal = new ArrayList<>();
        if (seat != acting())
            return legal;
        int pn = playerOf(seat);
        SOCPlayer player = game.getPlayer(pn);
        switch (prompt())
        {
        case "answer_offer":
            legal.addAll(answerMoves(seat));
            break;
        case "pick_partner":
            legal.addAll(pickMoves(seat));
            break;
        case "setup_settlement":
            for (int node : board.nodes())
                if (player.canPlaceSettlement(node))
                    legal.add(move("build_settlement", board.intersection(node)));
            break;
        case "setup_road":
        {
            // The road goes next to the settlement just placed
            List<Integer> next = game.getBoard().getAdjacentEdgesToNode(player.getLastSettlementCoord());
            for (int edge : roadSites(player))
                if (next.contains(edge))
                    legal.add(move("build_road", board.path(edge)));
            break;
        }
        case "road_building":
            for (int edge : roadSites(player))
                legal.add(move("build_road", board.path(edge)));
            break;
        case "discard":
            legal.addAll(discardMoves(seat));
            break;
        case "move_robber":
        {
            int robber = game.getBoard().getRobberHex();
            for (int hex : game.getBoard().getLandHexCoords())
            {
                if (hex == robber || ! game.canMoveRobber(pn, hex))
                    continue;
                List<Integer> victims = victims(hex, seat);
                if (victims.isEmpty())
                    legal.add(move("move_robber", Json.object("hex", board.hex(hex), "victim", null)));
                for (int victim : victims)
                    legal.add(move("move_robber", Json.object("hex", board.hex(hex), "victim", (long) victim)));
            }
            break;
        }
        case "roll":
            if (game.canRollDice(pn))
                legal.add(move("roll", null));
            cardMoves(pn, legal);
            break;
        case "main":
            if (game.couldBuildRoad(pn))
                for (int edge : roadSites(player))
                    legal.add(move("build_road", board.path(edge)));
            if (game.couldBuildSettlement(pn))
                for (int node : board.nodes())
                    if (player.canPlaceSettlement(node))
                        legal.add(move("build_settlement", board.intersection(node)));
            if (game.couldBuildCity(pn))
                for (int node : board.nodes())
                    if (player.isPotentialCity(node))
                        legal.add(move("build_city", board.intersection(node)));
            if (game.couldBuyDevCard(pn))
                legal.add(move("buy_development", null));
            cardMoves(pn, legal);
            // Bank trades of one card at the player's best rate: JSettlers' client offers no other, and its bookkeeping
            // of a worse one fails (SOCPlayer.makeBankTrade counts every trade of a resource with its 2:1 harbor at 2:1)
            for (int give : RESOURCES)
            {
                int best = player.getPortFlag(give) ? 2 : player.getPortFlag(SOCBoard.MISC_PORT) ? 3 : 4;
                for (int rate = best; rate <= best; rate++)
                    for (int get : RESOURCES)
                    {
                        if (get == give)
                            continue;
                        int[] gives = new int[SOCResourceConstants.WOOD + 1], gets = new int[SOCResourceConstants.WOOD + 1];
                        gives[give] = rate;
                        gets[get] = 1;
                        // JSettlers' check lets the reverse of the last bank trade through without looking at the hand
                        SOCResourceSet giving = resourceSet(cardSet(gives));
                        if (player.getResources().contains(giving) && game.canMakeBankTrade(giving, resourceSet(cardSet(gets))))
                            legal.add(move("bank_trade", Json.object("give", cardSet(gives), "get", cardSet(gets))));
                    }
            }
            if (game.canEndTurn(pn))
                legal.add(move("end_turn", null));
            break;
        default:
            break;
        }
        return legal;
    }

    // The development cards the player may play now, Year of Plenty and Monopoly with every choice of resources
    private void cardMoves(int pn, List<Object> legal)
    {
        if (game.canPlayKnight(pn))
            legal.add(move("play_knight", null));
        // As road_building forced has it: only with a road left (JSettlers checks that) and a place for it (it doesn't)
        if (game.canPlayRoadBuilding(pn) && ! roadSites(game.getPlayer(pn)).isEmpty())
            legal.add(move("play_road_building", null));
        if (game.canPlayDiscovery(pn))
            for (int i = 0; i < RESOURCES.length; i++)
                for (int j = i; j < RESOURCES.length; j++)
                {
                    List<Object> pick = Json.array(RESOURCE_NAMES[RESOURCES[i]], RESOURCE_NAMES[RESOURCES[j]]);
                    pick.sort(null);
                    legal.add(move("play_year_of_plenty", pick));
                }
        if (game.canPlayMonopoly(pn))
            for (int type : RESOURCES)
                legal.add(move("play_monopoly", RESOURCE_NAMES[type]));
    }

    /** The discards of a seat that has to discard (JSettlers' players discard at the same time, whoever acts) */
    public List<Object> discardMoves(int seat)
    {
        SOCPlayer player = game.getPlayer(playerOf(seat));
        List<Object> legal = new ArrayList<>();
        if (player.getNeedToDiscard())
            discards(player.getResources(), player.getCountToDiscard(), legal);
        return legal;
    }

    /** A state as a seat sees it: the other seats' hands, development cards and points with them hidden (null) */
    @SuppressWarnings("unchecked")
    public static Map<String, Object> hide(Map<String, Object> state, int viewer)
    {
        Map<String, Object> view = copy(state);
        for (Object entry : (List<Object>) view.get("players"))
        {
            Map<String, Object> player = (Map<String, Object>) entry;
            if (Json.integer(player.get("seat")) != viewer)
                for (String field : new String[] {"vp", "hand", "development"})
                    player.put(field, null);
        }
        return view;
    }

    /** A deep copy of a protocol value */
    @SuppressWarnings("unchecked")
    public static Map<String, Object> copy(Map<String, Object> value)
    {
        return (Map<String, Object>) Json.read(Json.write(value));
    }

    // Every card set of the count the hand can give up
    private static void discards(SOCResourceSet hand, int count, List<Object> legal)
    {
        int[] amounts = new int[SOCResourceConstants.WOOD + 1];
        discards(hand, count, 0, amounts, legal);
    }

    private static void discards(SOCResourceSet hand, int left, int index, int[] amounts, List<Object> legal)
    {
        if (index == RESOURCES.length)
        {
            if (left == 0)
                legal.add(move("discard", cardSet(amounts)));
            return;
        }
        int type = RESOURCES[index];
        for (int n = 0; n <= Math.min(left, hand.getAmount(type)); n++)
        {
            amounts[type] = n;
            discards(hand, left - n, index + 1, amounts, legal);
        }
        amounts[type] = 0;
    }

    // An accepted offer, as SOCGameMessageHandler.executeTrade makes it: the trade if JSettlers allows it, then every offer
    // cleared
    private void trade(int offering, int accepting)
    {
        if (! game.canMakeTrade(offering, accepting))
            return;
        game.makeTrade(offering, accepting);
        for (int pn = 0; pn < game.maxPlayers; pn++)
            game.getPlayer(pn).setCurrentOffer(null);
    }

    /**
     * Plays a legal move of the seat by JSettlers' rules and returns its result as the referee knows it: the dice, the
     * card bought, the card stolen (null if none), the cards a monopoly took from each other seat; null for other moves
     */
    @SuppressWarnings("unchecked")
    public Map<String, Object> apply(List<Object> move, int seat)
    {
        String type = (String) move.get(0);
        Object value = move.size() > 1 ? move.get(1) : null;
        int pn = playerOf(seat);
        SOCPlayer player = game.getPlayer(pn);
        SOCBoard b = game.getBoard();
        int state = game.getGameState();
        boolean initial = state < SOCGame.ROLL_OR_CARD;
        switch (type)
        {
        case "roll":
        {
            SOCGame.RollResult roll = game.rollDice();
            return Json.object("dice", Json.array((long) roll.diceA, (long) roll.diceB));
        }
        case "build_settlement":
        {
            int node = board.node(value);
            if (! initial)
                game.buySettlement(pn);
            game.putPiece(new SOCSettlement(player, node, b));
            return null;
        }
        case "build_road":
        {
            int edge = board.edge(value);
            if (! initial && state != SOCGame.PLACING_FREE_ROAD1 && state != SOCGame.PLACING_FREE_ROAD2)
                game.buyRoad(pn);
            game.putPiece(new SOCRoad(player, edge, b));
            // A first free road that leaves no place for the second gives the second up (JSettlers would wait for it)
            if (strandedFreeRoad())
                game.cancelBuildRoad(pn);
            return null;
        }
        case "build_city":
            game.buyCity(pn);
            game.putPiece(new SOCCity(player, board.node(value), b));
            return null;
        case "buy_development":
            return Json.object("card", cardName(game.buyDevCard()));
        case "play_knight":
            game.playKnight();
            return null;
        case "play_road_building":
            game.playRoadBuilding();
            return null;
        case "play_year_of_plenty":
        {
            game.playDiscovery();
            int[] counts = new int[SOCResourceConstants.WOOD + 1];
            for (Object name : (List<Object>) value)
                counts[resourceType((String) name)]++;
            game.doDiscoveryAction(resourceSet(cardSet(counts)));
            return null;
        }
        case "play_monopoly":
        {
            game.playMonopoly();
            int[] taken = game.doMonopolyAction(resourceType((String) value));
            Map<String, Object> result = new LinkedHashMap<>();
            for (int other = 0; other < game.maxPlayers; other++)
                if (other != seat)
                    result.put(Integer.toString(other), (long) taken[playerOf(other)]);
            return Json.object("taken", result);
        }
        case "move_robber":
        {
            Map<String, Object> robbery = (Map<String, Object>) value;
            int hex = board.hexCoord((List<?>) robbery.get("hex"));
            SOCMoveRobberResult result = game.moveRobber(pn, hex);
            int loot = result.getLoot();
            if (game.getGameState() == SOCGame.WAITING_FOR_ROB_CHOOSE_PLAYER)
                loot = game.choosePlayerForRobbery(playerOf(Json.integer(robbery.get("victim"))));
            return Json.object("stolen", loot > 0 ? resourceName(loot) : null);
        }
        case "discard":
            game.discard(pn, resourceSet(value));
            return null;
        case "bank_trade":
        {
            Map<String, Object> trade = (Map<String, Object>) value;
            game.makeBankTrade(resourceSet(trade.get("give")), resourceSet(trade.get("get")));
            return null;
        }
        case "end_turn":
            game.endTurn();
            return null;
        // Trades between players as SOCGameMessageHandler handles their requests (player_trading "first_accept")
        case "offer":
        {
            Map<String, Object> offer = (Map<String, Object>) value;
            boolean[] to = new boolean[game.maxPlayers];
            for (Object s : (List<Object>) offer.get("to"))
                to[playerOf(Json.integer(s))] = true;
            player.setCurrentOffer(new SOCTradeOffer(game.getName(), pn, to, resourceSet(offer.get("give")), resourceSet(offer.get("get"))));
            return null;
        }
        case "respond":
            if (Boolean.TRUE.equals(((Map<String, Object>) value).get("accept")))
                trade(game.getCurrentPlayerNumber(), pn);
            else
                game.rejectTradeOffersTo(pn);
            return null;
        case "counter":
        {
            // The counterer's own offer to the player on turn: in its own terms it gives the offerer's get
            Map<String, Object> terms = (Map<String, Object>) value;
            boolean[] to = new boolean[game.maxPlayers];
            to[game.getCurrentPlayerNumber()] = true;
            player.setCurrentOffer(new SOCTradeOffer(game.getName(), pn, to, resourceSet(terms.get("get")), resourceSet(terms.get("give"))));
            return null;
        }
        case "confirm":
            trade(playerOf(Json.integer(((Map<String, Object>) value).get("partner"))), pn);
            return null;
        case "cancel":
            player.setCurrentOffer(null);
            return null;
        default:
            throw new IllegalArgumentException("No JSettlers move for " + type);
        }
    }
}
