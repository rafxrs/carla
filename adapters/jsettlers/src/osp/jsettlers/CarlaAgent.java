package osp.jsettlers;

import java.io.*;
import java.lang.reflect.Field;
import java.util.*;
import soc.game.*;
import soc.message.*;
import soc.robot.*;
import soc.server.SOCServer;
import soc.util.*;

/**
 * CARLA v1 sidecar using the original JSettlers brain and strategies. The referee controls
 * transitions. Negotiation/phase orchestration is adapted. No socket, server, GUI, pinger or robot
 * thread is started.
 */
@SuppressWarnings({"unchecked", "deprecation"})
public final class CarlaAgent {
  static final int[] RES = {5, 1, 3, 4, 2};
  static final int[] DEV = {
    SOCDevCardConstants.KNIGHT,
    SOCDevCardConstants.ROADS,
    SOCDevCardConstants.DISC,
    SOCDevCardConstants.MONO,
    SOCDevCardConstants.CAP
  };

  static int n(Object value) {
    return Json.integer(value);
  }

  static Map<String, Object> obj(Object value) {
    return (Map<String, Object>) value;
  }

  static List<Object> arr(Object value) {
    return (List<Object>) value;
  }

  static List<Object> list(int[] values) {
    List<Object> a = new ArrayList<>();
    for (int v : values) a.add(v);
    return a;
  }

  static Map<String, Object> action(String type) {
    return Json.object("type", type);
  }

  static class Selected extends RuntimeException {
    final SOCMessage message;

    Selected(SOCMessage message) {
      this.message = message;
    }
  }

  static class Client extends SOCRobotClient {
    Client(int seat) {
      super("localhost", 0, "seat" + seat, null, null);
      sVersion = 2610;
    }

    @Override
    public synchronized boolean put(String command) {
      SOCMessage message = SOCMessage.toMsg(command);
      if (message instanceof SOCGameTextMsg) return true;
      throw new Selected(message);
    }
  }

  static class Brain extends SOCRobotBrain {
    Brain(Client client, SOCGame game, boolean smart, long seed) {
      super(
          client,
          smart ? SOCServer.ROBOT_PARAMS_SMARTER : SOCServer.ROBOT_PARAMS_DEFAULT,
          game,
          new CappedQueue<SOCMessage>());
      rand.setSeed(seed);
      setOurPlayerData();
    }

    @Override
    public void pause(int msec) {}

    /** planBuilding calls, a diagnostic for when the original brain plans (and so refreshes). */
    int plans;

    @Override
    protected void planBuilding() {
      plans++;
      super.planBuilding();
    }

    void piece(SOCPlayingPiece piece) {
      if (piece.getPlayerNumber() == ourPlayerNumber) {
        if (!buildingPlan.isEmpty()) {
          SOCPossiblePiece planned = buildingPlan.getFirstPiece();
          if (planned.getCoordinates() == piece.getCoordinates()
              && planned.getType() == piece.getType()) buildingPlan.advancePlan();
        }
        whatWeWantToBuild = null;
      }
      if (piece instanceof SOCSettlement) trackNewSettlement((SOCSettlement) piece, false);
      else if (piece instanceof SOCCity) trackNewCity((SOCCity) piece, false);
      else trackNewRoadOrShip((SOCRoutePiece) piece, false);
    }

    void clearWaiting() {
      waitingForGameState =
          waitingForTradeMsg = waitingForTradeResponse = waitingForDevCard = false;
      expectPLACING_ROAD =
          expectPLACING_SETTLEMENT = expectPLACING_CITY = expectPLACING_SHIP = false;
      expectPLACING_FREE_ROAD1 = expectPLACING_FREE_ROAD2 = expectPLACING_ROBBER = false;
      expectWAITING_FOR_DISCOVERY = expectWAITING_FOR_MONOPOLY = false;
      ourTurn = true;
      waitingForOurTurn = false;
      expectDICERESULT = false;
    }

    void turn() {
      resetFieldsAtEndTurn();
      clearWaiting();
    }

    void main() {
      clearWaiting();
      // Revalidate retained plans after CARLA collapses request/placement into one action.
      for (SOCPlayerTracker tracker : playerTrackers)
        if (tracker != null)
          tracker
              .getPossibleRoads()
              .keySet()
              .removeIf(edge -> game.getBoard().roadOrShipAtEdge(edge) != null);
      if (!buildingPlan.isEmpty()) {
        SOCPossiblePiece planned = buildingPlan.getFirstPiece();
        if (planned.getType() == SOCPlayingPiece.ROAD
            && game.getBoard().roadOrShipAtEdge(planned.getCoordinates()) != null)
          buildingPlan.clear();
      }
      planAndDoActionForPLAY1();
    }

    void roll() {
      clearWaiting();
      rollOrPlayKnightOrExpectDice();
    }

    SOCPlayingPiece wanted() {
      return whatWeWantToBuild;
    }

    int opening(boolean second) {
      return second
          ? openingBuildStrategy.planSecondSettlement()
          : openingBuildStrategy.planInitialSettlements();
    }

    int road() {
      return openingBuildStrategy.planInitRoad();
    }

    SOCResourceSet discardOne() {
      return discardStrategy.discard(1, buildingPlan);
    }

    int robber() {
      return robberStrategy.getBestRobberHex();
    }

    int victim(boolean[] candidates) {
      return robberStrategy.chooseRobberVictim(candidates, false);
    }

    SOCResourceSet resources() {
      return decisionMaker.getResourceChoices();
    }

    int monopoly() {
      return monopolyStrategy.getMonopolyChoice();
    }

    SOCTradeOffer offer() {
      if (buildingPlan.isEmpty()) planBuilding();
      return buildingPlan.isEmpty() ? null : negotiator.makeOffer(buildingPlan);
    }

    int consider(SOCTradeOffer offer) {
      return negotiator.considerOffer2(offer, ourPlayerNumber);
    }

    SOCTradeOffer counter(SOCTradeOffer offer) {
      return negotiator.makeCounterOffer(offer);
    }

    void offered(SOCTradeOffer offer) {
      negotiator.addToOffersMade(offer);
    }

    void limitOffers() {
      setDoneTrading(true);
    }

    /** handlePLAYERELEMENT during ROLL_OR_CARD: after our own production on a roll. */
    void productionReceived() {
      negotiator.resetIsSelling();
    }

    void plan() {
      buildingPlan.clear();
      planBuilding();
    }

    SOCBuildPlan planState() {
      return buildingPlan;
    }

    SOCPlayerTracker[] trackers() {
      return playerTrackers;
    }

    /** SOCRobotDM.getWinGameETABonusForRoad's hypothetical, without scoring (diagnostic). */
    List<Object> tryRoad(int edge, java.util.function.Function<SOCPossiblePiece, Object> piece) {
      SOCRoad road = new SOCRoad(ourPlayerData, edge, null);
      SOCPlayerTracker[] copies = SOCPlayerTracker.tryPutPiece(road, game, playerTrackers);
      List<Object> players = new ArrayList<>();
      try {
        SOCPlayerTracker.updateWinGameETAs(copies);
        for (int p = 0; p < 4; p++) {
          List<Object> settlements = new ArrayList<>();
          for (SOCPossibleSettlement s : copies[p].getPossibleSettlements().values())
            settlements.add(piece.apply(s));
          players.add(
              Json.object(
                  "win_eta", copies[p].getWinGameETA(),
                  "road_eta", copies[p].getLongestRoadETA(),
                  "roads_to_go", copies[p].getRoadsToGo(),
                  "army_eta", copies[p].getLargestArmyETA(),
                  "road_length", game.getPlayer(p).getLongestRoadLength(),
                  "settlements", settlements));
        }
      } finally {
        SOCPlayerTracker.undoTryPutPiece(road, game);
      }
      return players;
    }

    Object negotiationField(String name) throws Exception {
      Field field = SOCRobotNegotiator.class.getDeclaredField(name);
      field.setAccessible(true);
      return field.get(negotiator);
    }

    void observeOffer(SOCTradeOffer offer) {
      try {
        java.lang.reflect.Method method =
            SOCRobotNegotiator.class.getDeclaredMethod(
                "recordResourcesFromOffer", SOCTradeOffer.class);
        method.setAccessible(true);
        method.invoke(negotiator, offer);
      } catch (ReflectiveOperationException error) {
        throw new IllegalStateException(error);
      }
    }

    void observeReject(int rejector, SOCTradeOffer offer) {
      SOCTradeOffer saved = ourPlayerData.getCurrentOffer();
      try {
        ourPlayerData.setCurrentOffer(offer);
        java.lang.reflect.Method method =
            SOCRobotNegotiator.class.getDeclaredMethod("recordResourcesFromReject", int.class);
        method.setAccessible(true);
        method.invoke(negotiator, rejector);
      } catch (ReflectiveOperationException error) {
        throw new IllegalStateException(error);
      } finally {
        ourPlayerData.setCurrentOffer(saved);
      }
    }

    SOCTradeOffer bank(SOCResourceSet target) {
      return negotiator.getOfferToBank(target);
    }

    SOCResourceSet discardCards(int count) {
      return discardStrategy.discard(count, buildingPlan);
    }

    boolean chooseMonopoly() {
      return monopolyStrategy.decidePlayMonopoly();
    }

    boolean knightForArmy() {
      return decisionMaker.shouldPlayKnightForLA();
    }

    float smartScore(int[] before, int[] after, int eta) throws Exception {
      Field field = SOCPlayerTracker.class.getDeclaredField("winGameETA");
      field.setAccessible(true);
      int[] saved = new int[4];
      Vector<SOCPlayerTracker> leaders = new Vector<>();
      int best = 1000;
      for (int value : before) best = Math.min(best, value);
      for (int p = 0; p < 4; p++) {
        saved[p] = field.getInt(playerTrackers[p]);
        if (before[p] == best) leaders.add(playerTrackers[p]);
      }
      try {
        for (int p = 0; p < 4; p++) field.setInt(playerTrackers[p], after[p]);
        java.lang.reflect.Method bonusMethod =
            SOCRobotDM.class.getDeclaredMethod(
                "calcWGETABonusAux", int[].class, SOCPlayerTracker[].class, Vector.class);
        bonusMethod.setAccessible(true);
        float bonus =
            ((Number) bonusMethod.invoke(decisionMaker, before, playerTrackers, leaders))
                .floatValue();
        java.lang.reflect.Method etaMethod =
            SOCRobotDM.class.getDeclaredMethod("getETABonus", int.class, int.class, float.class);
        etaMethod.setAccessible(true);
        return ((Number) etaMethod.invoke(decisionMaker, eta, best, bonus)).floatValue();
      } finally {
        for (int p = 0; p < 4; p++) field.setInt(playerTrackers[p], saved[p]);
      }
    }

    SOCResourceSet chooseResources(SOCResourceSet target, int count) throws Exception {
      java.lang.reflect.Method method =
          SOCRobotDM.class.getDeclaredMethod(
              "chooseFreeResources", SOCResourceSet.class, int.class, boolean.class);
      method.setAccessible(true);
      method.invoke(decisionMaker, target, count, true);
      return decisionMaker.getResourceChoices();
    }
  }

  SOCGame game;
  OSPBoard board;
  Brain brain;
  int seat, turn = -1;
  int[] nodes, edges, hexes;
  Map<Integer, Integer> nodeInverse = new HashMap<>(),
      edgeInverse = new HashMap<>(),
      hexInverse = new HashMap<>();
  int[] owners = new int[54], buildings = new int[54], roadOwners = new int[72];
  boolean smart;
  int setupBuildings = 0, setupRoads = 0, freeRoads = 0;

  CarlaAgent(boolean smart) {
    this.smart = smart;
  }

  void before(Map<String, Object> message) throws Exception {
    Map<String, Object> view = obj(message.get("view"));
    seat = n(view.get("seat"));
    game = OSPBoard.newGame("carla");
    game.setFirstPlayer(0);
    board = new OSPBoard(game.getBoard());
    Map<String, Object> topology = obj(message.get("topology"));
    List<Object> keys = arr(topology.get("corners"));
    nodes = new int[54];
    for (int i = 0; i < 54; i++) {
      nodes[i] = board.node(keys.get(i));
      nodeInverse.put(nodes[i], i);
    }
    List<Object> paths = arr(topology.get("edges"));
    edges = new int[72];
    for (int i = 0; i < 72; i++) {
      edges[i] = board.edge(paths.get(i));
      edgeInverse.put(edges[i], i);
    }
    List<Object> tiles = arr(topology.get("tiles"));
    hexes = new int[19];
    for (int i = 0; i < 19; i++) {
      hexes[i] = board.hexCoord(arr(tiles.get(i)));
      hexInverse.put(hexes[i], i);
    }
    Map<String, Object> state = obj(view.get("state")), layout = obj(state.get("board"));
    int[] hl = game.getBoard().getHexLayout().clone(),
        nl = game.getBoard().getNumberLayout().clone();
    List<Object> terrain = arr(layout.get("tile_resource")),
        numbers = arr(layout.get("tile_number"));
    for (int i = 0; i < 19; i++) {
      int pos = game.getBoard().getHexNumFromCoord(hexes[i]);
      int r = n(terrain.get(i));
      hl[pos] = r < 0 ? SOCBoard.DESERT_HEX : RES[r];
      nl[pos] = n(numbers.get(i));
    }
    game.getBoard().setHexLayout(hl);
    game.getBoard().setNumberLayout(nl);
    Field portField = SOCBoard.class.getDeclaredField("nodeIDtoPortType");
    portField.setAccessible(true);
    Map<Integer, Integer> ports = (Map<Integer, Integer>) portField.get(game.getBoard());
    ports.clear();
    for (int i = 0; i < 6; i++) game.getBoard().getPortCoordinates(i).clear();
    List<Object> portNodes = arr(topology.get("ports")), portTypes = arr(layout.get("port_types"));
    for (int i = 0; i < 9; i++) {
      int r = n(portTypes.get(i));
      int type = r == 5 ? 0 : RES[r];
      for (Object c : arr(portNodes.get(i))) {
        ports.put(nodes[n(c)], type);
        game.getBoard().getPortCoordinates(type).add(nodes[n(c)]);
      }
    }
    Arrays.fill(owners, -1);
    Arrays.fill(roadOwners, -1);
    brain = new Brain(new Client(seat), game, smart, ((Number) message.get("seed")).longValue());
    synchronize(state);
  }

  void synchronize(Map<String, Object> s) {
    int current = n(s.get("current_player"));
    game.setCurrentPlayerNumber(current);
    int phase = n(s.get("phase")), setup = n(s.get("setup_step"));
    int gs =
        phase == 0
            ? (setup < 4 ? SOCGame.START1A : SOCGame.START2A)
            : phase == 1
                ? (setup < 4 ? SOCGame.START1B : SOCGame.START2B)
                : phase == 2
                    ? SOCGame.ROLL_OR_CARD
                    : phase == 3
                        ? SOCGame.WAITING_FOR_DISCARDS
                        : phase == 4
                            ? SOCGame.PLACING_ROBBER
                            : phase == 6 ? SOCGame.PLACING_FREE_ROAD1 : SOCGame.PLAY1;
    game.setGameState(gs);
    List<Object> co = arr(s.get("corner_owner")),
        cb = arr(s.get("corner_building")),
        eo = arr(s.get("edge_owner"));
    // Normally step messages install pieces chronologically. This also supports start snapshots.
    for (int i = 0; i < 54; i++)
      if (n(co.get(i)) >= 0 && owners[i] < 0) putPiece(n(co.get(i)), i, 1);
    for (int i = 0; i < 72; i++)
      if (n(eo.get(i)) >= 0 && roadOwners[i] < 0) putPiece(n(eo.get(i)), i, 0);
    for (int i = 0; i < 54; i++)
      if (n(cb.get(i)) == 2 && buildings[i] != 2) putPiece(n(co.get(i)), i, 2);
    game.setCurrentPlayerNumber(current);
    game.setGameState(gs);
    game.getBoard().setRobberHex(hexes[n(s.get("robber_tile"))], false);
    List<Object> hands = arr(s.get("hands")),
        sizes = arr(s.get("hand_sizes")),
        devs = arr(s.get("dev_hands")),
        devsizes = arr(s.get("development_sizes"));
    for (int p = 0; p < 4; p++) {
      SOCPlayer player = game.getPlayer(p);
      SOCResourceSet hand = player.getResources();
      if (hands.get(p) == null) {
        int count = n(sizes.get(p));
        if (hand.getAmount(SOCResourceConstants.UNKNOWN) < 0 || hand.getKnownTotal() > count) {
          hand.clear();
          hand.add(count, SOCResourceConstants.UNKNOWN);
        } else hand.add(count - hand.getTotal(), SOCResourceConstants.UNKNOWN);
      } else {
        hand.clear();
        for (int r = 0; r < 5; r++) hand.add(n(arr(hands.get(p)).get(r)), RES[r]);
      }
      player.getInventory().clear();
      if (devs.get(p) == null)
        player
            .getInventory()
            .addDevCard(n(devsizes.get(p)), SOCInventory.OLD, SOCDevCardConstants.UNKNOWN);
      else
        for (int c = 0; c < 5; c++) {
          int amount = n(arr(devs.get(p)).get(c)), fresh = 0;
          if (p == current && s.get("dev_bought_this_turn") != null)
            fresh = n(arr(s.get("dev_bought_this_turn")).get(c));
          player.getInventory().addDevCard(amount - fresh, SOCInventory.OLD, DEV[c]);
          player.getInventory().addDevCard(fresh, SOCInventory.NEW, DEV[c]);
        }
      player.setNumKnights(n(arr(arr(s.get("dev_played")).get(p)).get(0)));
      player.setPlayedDevCard(p == current && Boolean.TRUE.equals(s.get("dev_played_this_turn")));
    }
    int lr = n(s.get("longest_road_owner")), la = n(s.get("largest_army_owner"));
    game.setPlayerWithLongestRoad(lr < 0 ? null : game.getPlayer(lr));
    game.setPlayerWithLargestArmy(la < 0 ? null : game.getPlayer(la));
    game.setNumDevCards(n(s.get("development_remaining")));
    game.setRoundCount(n(s.get("turn")) / 4 + 1);
    int nextTurn = n(s.get("turn"));
    if (turn != nextTurn) {
      brain.turn();
      turn = nextTurn;
    }
    Object limit = obj(s.get("config")).get("max_trade_proposals_per_turn");
    if (!Boolean.TRUE.equals(obj(s.get("config")).get("allow_domestic_trade"))
        || (limit != null && n(s.getOrDefault("trade_proposals_this_turn", 0)) >= n(limit)))
      brain.limitOffers();
  }

  void putPiece(int owner, int location, int type) {
    SOCPlayer p = game.getPlayer(owner);
    SOCPlayingPiece piece =
        type == 0
            ? new SOCRoad(p, edges[location], null)
            : type == 1
                ? new SOCSettlement(p, nodes[location], null)
                : new SOCCity(p, nodes[location], null);
    // Native putPiece grants setup resources itself. CARLA's subsequent
    // STARTING_RESOURCES event is authoritative, so retain the pre-placement
    // hand here rather than applying that gain twice and losing known types
    // during the next snapshot reconciliation.
    SOCResourceSet resourcesBeforePlacement = p.getResources().copy();
    game.putPiece(piece);
    p.getResources().setAmounts(resourcesBeforePlacement);
    brain.piece(piece);
    if (type == 0) roadOwners[location] = owner;
    else {
      owners[location] = owner;
      buildings[location] = type;
    }
  }

  void change(int player, int resource, int amount) {
    SOCResourceSet h = game.getPlayer(player).getResources();
    if (amount >= 0) {
      // The original never marks single gains as selling; see productionReceived.
      h.add(amount, resource);
    } else h.subtract(-amount, resource, resource == SOCResourceConstants.UNKNOWN);
  }

  void cost(int player, int[] cost) {
    for (int r = 0; r < 5; r++) change(player, RES[r], -cost[r]);
  }

  void step(Map<String, Object> message) {
    boolean produced = false;
    for (Object value : arr(message.get("events"))) {
      Map<String, Object> e = obj(value);
      int kind = n(e.get("kind")), p = n(e.get("player"));
      if (kind == 16) {
        // TURN: the original resets its turn fields (selling flags, offers made, plan) when
        // the next turn starts, before that player's first offer, not at our next decision.
        brain.turn();
        turn++;
      }
      if (kind == 1 && p == seat)
        for (Object count : arr(e.get("counts"))) produced |= n(count) > 0;
      if (kind == 18 || kind == 23) {
        boolean[] targets = new boolean[4];
        if (kind == 23) targets[n(e.get("target"))] = true;
        else for (int other = 0; other < 4; other++) targets[other] = other != p;
        SOCTradeOffer offer =
            new SOCTradeOffer(
                game.getName(),
                p,
                targets,
                resourceSet(e.get("give")),
                resourceSet(e.get("receive")));
        game.getPlayer(p).setCurrentOffer(offer);
        if (p != seat) brain.observeOffer(offer);
      }
      if (kind == 19 && n(e.get("value")) == 0) {
        SOCTradeOffer offer = game.getPlayer(n(e.get("target"))).getCurrentOffer();
        if (offer != null) brain.observeReject(p, offer);
      }
      if (kind == 20 || kind == 21)
        for (int other = 0; other < 4; other++) game.getPlayer(other).setCurrentOffer(null);
      if (kind == 1 || kind == 2 || kind == 11 || kind == 12) {
        List<Object> counts = arr(e.get("counts"));
        for (int r = 0; r < 5; r++) change(p, RES[r], n(counts.get(r)));
      }
      if (kind == 3) {
        if (setupBuildings >= 8) cost(p, new int[] {1, 1, 1, 1, 0});
        setupBuildings++;
      }
      if (kind == 4) cost(p, new int[] {0, 0, 0, 2, 3});
      if (kind == 5) {
        if (setupRoads >= 8) {
          if (freeRoads > 0) freeRoads--;
          else cost(p, new int[] {1, 1, 0, 0, 0});
        }
        setupRoads++;
      }
      if (kind == 6) cost(p, new int[] {0, 0, 1, 1, 1});
      if (kind == 7 && n(e.get("value")) == 1) freeRoads = 2;
      if (kind == 9) {
        int r = n(e.get("value")), resource = r < 0 ? SOCResourceConstants.UNKNOWN : RES[r];
        change(n(e.get("target")), resource, -1);
        change(p, resource, 1);
      }
      if (kind == 10) change(p, RES[n(e.get("value"))], -1);
      if (kind == 13) {
        int r = RES[n(e.get("value"))];
        List<Object> counts = arr(e.get("counts"));
        for (int other = 0; other < 4; other++)
          if (other != p) {
            int count = n(counts.get(other));
            change(other, r, -count);
            change(p, r, count);
          }
      }
      if (kind == 20) {
        int other = n(e.get("target"));
        List<Object> give = arr(e.get("give")), get = arr(e.get("receive"));
        for (int r = 0; r < 5; r++) {
          int delta = n(get.get(r)) - n(give.get(r));
          change(p, RES[r], delta);
          change(other, RES[r], -delta);
        }
      }
      if (kind >= 3 && kind <= 5) {
        int loc = n(e.get("location")), type = kind == 3 ? 1 : kind == 4 ? 2 : 0;
        if (type == 0 ? roadOwners[loc] < 0 : buildings[loc] != type) putPiece(p, loc, type);
      }
    }
    // The server sends a gaining player its full hand after a roll, while the robot's game is
    // still in ROLL_OR_CARD; handlePLAYERELEMENT then resets every player's selling flags.
    if (produced) brain.productionReceived();
  }

  Map<String, Object> pieceAction(SOCPlayingPiece piece) {
    Map<String, Object> a =
        action(
            piece instanceof SOCRoad
                ? "BUILD_ROAD"
                : piece instanceof SOCCity ? "BUILD_CITY" : "BUILD_SETTLEMENT");
    a.put(
        piece instanceof SOCRoad ? "edge" : "corner",
        piece instanceof SOCRoad
            ? edgeInverse.get(piece.getCoordinates())
            : nodeInverse.get(piece.getCoordinates()));
    return a;
  }

  Map<String, Object> offerAction(SOCTradeOffer offer) {
    List<Object> give = new ArrayList<>(), get = new ArrayList<>();
    for (int r : RES) {
      give.add(offer.getGiveSet().getAmount(r));
      get.add(offer.getGetSet().getAmount(r));
    }
    Map<String, Object> a = action("PROPOSE_TRADE");
    a.put("trade", Json.object("give", give, "receive", get));
    return a;
  }

  Map<String, Object> selected(SOCMessage message) {
    if (message instanceof SOCBuildRequest) return pieceAction(brain.wanted());
    if (message instanceof SOCPutPiece) {
      SOCPutPiece p = (SOCPutPiece) message;
      return pieceAction(
          p.getPieceType() == 0
              ? new SOCRoad(game.getPlayer(seat), p.getCoordinates(), null)
              : p.getPieceType() == 1
                  ? new SOCSettlement(game.getPlayer(seat), p.getCoordinates(), null)
                  : new SOCCity(game.getPlayer(seat), p.getCoordinates(), null));
    }
    if (message instanceof SOCRollDice) return action("ROLL_DICE");
    if (message instanceof SOCEndTurn) return action("END_TURN");
    if (message instanceof SOCBuyDevCardRequest) return action("BUY_DEV_CARD");
    if (message instanceof SOCMakeOffer) {
      SOCTradeOffer o = ((SOCMakeOffer) message).getOffer();
      brain.offered(o);
      return offerAction(o);
    }
    if (message instanceof SOCBankTrade) {
      SOCBankTrade t = (SOCBankTrade) message;
      Map<String, Object> a = action("MARITIME_TRADE");
      for (int r = 0; r < 5; r++) {
        if (t.getGiveSet().getAmount(RES[r]) > 0) a.put("resource", r);
        if (t.getGetSet().getAmount(RES[r]) > 0) a.put("resource2", r);
      }
      return a;
    }
    if (message instanceof SOCPlayDevCardRequest) {
      int c = ((SOCPlayDevCardRequest) message).getDevCard();
      Map<String, Object> a = action("PLAY_DEV_CARD");
      for (int i = 0; i < 4; i++) if (c == DEV[i]) a.put("dev", i);
      if (c == DEV[2]) {
        List<Integer> picks = new ArrayList<>();
        for (int r = 0; r < 5; r++)
          for (int j = 0; j < brain.resources().getAmount(RES[r]); j++) picks.add(r);
        Collections.sort(picks);
        a.put("resource", picks.get(0));
        a.put("resource2", picks.get(1));
      }
      if (c == DEV[3])
        for (int r = 0; r < 5; r++) if (RES[r] == brain.monopoly()) a.put("resource", r);
      return a;
    }
    throw new IllegalStateException("unsupported robot command: " + message);
  }

  static boolean bankTradeLegal(Map<String, Object> trade, List<Object> legal) {
    for (Object value : legal) {
      Map<String, Object> a = obj(value);
      if ("MARITIME_TRADE".equals(a.get("type"))
          && n(a.get("resource")) == n(trade.get("resource"))
          && n(a.get("resource2")) == n(trade.get("resource2"))) return true;
    }
    return false;
  }

  Map<String, Object> decide(Map<String, Object> view) {
    Map<String, Object> s = obj(view.get("state"));
    synchronize(s);
    int phase = n(s.get("phase"));
    List<Object> legal = arr(view.get("legal"));
    // No single-option shortcut: the original brain plans (refreshing its tracker ETAs) on every
    // main-phase turn and discard, and recalculates win ETAs for every robber move, even when
    // only one move is legal. Skipping it here diverged from robots on their own server.
    if (phase == 0) {
      int node = brain.opening(n(s.get("setup_step")) >= 4);
      Map<String, Object> a = action("BUILD_SETTLEMENT");
      a.put("corner", nodeInverse.get(node));
      return a;
    }
    if (phase == 1) {
      Map<String, Object> a = action("BUILD_ROAD");
      a.put("edge", edgeInverse.get(brain.road()));
      return a;
    }
    if (phase == 3) {
      SOCResourceSet discard = brain.discardOne();
      for (int r = 0; r < 5; r++)
        if (discard.getAmount(RES[r]) > 0) {
          Map<String, Object> a = action("DISCARD");
          a.put("resource", r);
          return a;
        }
    }
    if (phase == 4) {
      int hex = brain.robber();
      List<Map<String, Object>> options = new ArrayList<>();
      boolean[] victims = new boolean[4];
      for (Object value : legal) {
        Map<String, Object> a = obj(value);
        if (n(a.get("tile")) == hexInverse.get(hex)) {
          options.add(a);
          if (a.containsKey("player") && n(a.get("player")) >= 0)
            victims[n(a.get("player"))] = true;
        }
      }
      if (options.size() == 1) return options.get(0);
      int victim = brain.victim(victims);
      for (Map<String, Object> a : options)
        if (a.containsKey("player") && n(a.get("player")) == victim) return a;
      throw new IllegalStateException("robot chose no legal robber victim");
    }
    if (phase == 7) {
      Object pending = s.get("pending_trade");
      if (pending == null) {
        SOCTradeOffer o = brain.offer();
        return o == null ? action("REJECT_TRADE") : offerAction(o);
      }
      Map<String, Object> terms =
          pending instanceof List
              ? Json.object("give", arr(pending).get(0), "receive", arr(pending).get(1))
              : obj(pending);
      SOCResourceSet give = new SOCResourceSet(), get = new SOCResourceSet();
      for (int r = 0; r < 5; r++) {
        give.add(n(arr(terms.get("give")).get(r)), RES[r]);
        get.add(n(arr(terms.get("receive")).get(r)), RES[r]);
      }
      boolean[] targets = new boolean[4];
      targets[seat] = true;
      SOCTradeOffer offer =
          new SOCTradeOffer(game.getName(), n(s.get("current_player")), targets, give, get);
      int response = brain.consider(offer);
      if (response == SOCRobotNegotiator.ACCEPT_OFFER) return action("ACCEPT_TRADE");
      // handleMAKEOFFER only counteroffers on COUNTER_OFFER; REJECT_OFFER rejects outright.
      if (response != SOCRobotNegotiator.COUNTER_OFFER) return action("REJECT_TRADE");
      SOCTradeOffer counter = brain.counter(offer);
      return counter == null ? action("REJECT_TRADE") : offerAction(counter);
    }
    if (phase == 8) {
      for (Object value : legal) {
        Map<String, Object> a = obj(value);
        if ("CONFIRM_TRADE".equals(a.get("type"))) return a;
      }
      return action("CANCEL_TRADE");
    }
    if (phase == 6) {
      SOCPlayingPiece wanted = brain.wanted();
      if (wanted instanceof SOCRoad)
        for (Object v : legal)
          if (n(obj(v).get("edge")) == edgeInverse.get(wanted.getCoordinates())) return obj(v);
      brain
          .getDecisionMaker()
          .planStuff(smart ? SOCRobotDM.SMART_STRATEGY : SOCRobotDM.FAST_STRATEGY);
      Map<String, Object> best = null;
      float score = -Float.MAX_VALUE;
      for (Object v : legal) {
        Map<String, Object> a = obj(v);
        SOCPossibleRoad road =
            brain.getOurPlayerTracker().getPossibleRoads().get(edges[n(a.get("edge"))]);
        if (road != null && (best == null || road.getScore() > score)) {
          best = a;
          score = road.getScore();
        }
      }
      if (best != null) return best;
      throw new IllegalStateException("no legal Road Building candidate in robot tracker");
    }
    try {
      if (phase == 2) brain.roll();
      else brain.main();
    } catch (Selected selected) {
      Map<String, Object> a = selected(selected.message);
      // JSettlers' bank is unlimited; CARLA's holds 19 of each. A trade the bank cannot pay is
      // skipped, and since getOfferToBank only trades for an unaffordable plan, the turn ends,
      // as in the native port's boundary policy.
      if (selected.message instanceof SOCBankTrade && !bankTradeLegal(a, legal))
        return action("END_TURN");
      return a;
    }
    throw new IllegalStateException(
        "robot emitted no command phase="
            + phase
            + " turn="
            + turn
            + " "
            + brain.debugPrintBrainStatus(false));
  }

  List<Object> resourceList(SOCResourceSet resources) {
    List<Object> result = new ArrayList<>();
    for (int r : RES) result.add(resources.getAmount(r));
    return result;
  }

  SOCResourceSet resourceSet(Object values) {
    SOCResourceSet result = new SOCResourceSet();
    List<Object> counts = arr(values);
    for (int i = 0; i < 5; i++) result.add(n(counts.get(i)), RES[i]);
    return result;
  }

  Map<String, Object> possiblePiece(SOCPossiblePiece piece) {
    int type = piece.getType();
    Map<String, Object> result =
        Json.object("type", type, "eta", piece.getETA(), "score", piece.getScore());
    if (type == SOCPlayingPiece.ROAD)
      result.put("location", edgeInverse.get(piece.getCoordinates()));
    else if (type == SOCPlayingPiece.SETTLEMENT || type == SOCPlayingPiece.CITY)
      result.put("location", nodeInverse.get(piece.getCoordinates()));
    else result.put("location", -1);
    SOCResourceSet cost = piece.getResourcesToBuild();
    result.put("cost", cost == null ? null : resourceList(cost));
    List<Object> necessary = new ArrayList<>();
    if (piece instanceof SOCPossibleRoad)
      for (SOCPossibleRoad road : ((SOCPossibleRoad) piece).getNecessaryRoads())
        necessary.add(edgeInverse.get(road.getCoordinates()));
    if (piece instanceof SOCPossibleSettlement)
      for (SOCPossibleRoad road : ((SOCPossibleSettlement) piece).getNecessaryRoads())
        necessary.add(edgeInverse.get(road.getCoordinates()));
    result.put("necessary_roads", necessary);
    List<Object> threats = new ArrayList<>();
    for (SOCPossiblePiece threat : piece.getThreats()) {
      int kind = threat.getType();
      Integer location =
          kind == SOCPlayingPiece.ROAD
              ? edgeInverse.get(threat.getCoordinates())
              : nodeInverse.get(threat.getCoordinates());
      threats.add(Json.array(threat.getPlayer().getPlayerNumber(), kind, location));
    }
    result.put("threats", threats);
    if (piece instanceof SOCPossibleSettlement) {
      List<Object> conflicts = new ArrayList<>();
      for (SOCPossibleSettlement conflict : ((SOCPossibleSettlement) piece).getConflicts())
        conflicts.add(
            Json.array(
                conflict.getPlayer().getPlayerNumber(),
                nodeInverse.get(conflict.getCoordinates())));
      result.put("conflicts", conflicts);
    }
    if (piece instanceof SOCPossibleCity)
      result.put("speedup", list(((SOCPossibleCity) piece).getSpeedup()));
    if (piece instanceof SOCPossibleSettlement)
      result.put("speedup", list(((SOCPossibleSettlement) piece).getSpeedup()));
    if (piece instanceof SOCPossibleSettlement) {
      Stack<SOCPossibleRoad> path = ((SOCPossibleSettlement) piece).getRoadPath();
      List<Object> execution = new ArrayList<>();
      if (path != null)
        for (int i = path.size() - 1; i >= 0; i--)
          execution.add(edgeInverse.get(path.get(i).getCoordinates()));
      result.put("road_path", execution);
      result.put(
          "number_of_necessary_roads", ((SOCPossibleSettlement) piece).getNumberOfNecessaryRoads());
    }
    return result;
  }

  // Diagnostics are stateful and must run in a dedicated reference process,
  // never in the process used to supply a scored tournament decision.
  Map<String, Object> inspect(Map<String, Object> request) throws Exception {
    synchronize(obj(obj(request.get("view")).get("state")));
    String operation = (String) request.get("operation");
    SOCPlayer player = game.getPlayer(seat);
    Map<String, Object> result = Json.object("operation", operation, "version", "2.6.10");
    if (operation.equals("opening")) {
      int node = brain.opening(Boolean.TRUE.equals(request.get("second")));
      result.put("corner", nodeInverse.get(node));
    } else if (operation.equals("initial_road")) {
      result.put("edge", edgeInverse.get(brain.road()));
    } else if (operation.equals("plan")) {
      brain.plan();
    } else if (operation.equals("win_times")) {
      if (Boolean.TRUE.equals(request.get("trace"))) {
        List<Object> traces = new ArrayList<>();
        DebugRecorder recorder = brain.getDRecorder();
        recorder.turnOn();
        try {
          for (SOCPlayerTracker tracker : brain.trackers()) {
            tracker.recalcLongestRoadETA();
            tracker.recalcLargestArmyETA();
            recorder.startRecording("projection");
            tracker.recalcWinGameETA();
            recorder.stopRecording();
            traces.add(new ArrayList<>(recorder.getRecord("projection")));
          }
        } finally {
          recorder.turnOff();
        }
        result.put("traces", traces);
      } else SOCPlayerTracker.updateWinGameETAs(brain.trackers());
    } else if (operation.equals("consider")) {
      // considerOffer2 on the pending offer (diagnostic; may cache predicted target pieces).
      Map<String, Object> s = obj(obj(request.get("view")).get("state"));
      Object pending = s.get("pending_trade");
      Map<String, Object> terms =
          pending instanceof List
              ? Json.object("give", arr(pending).get(0), "receive", arr(pending).get(1))
              : obj(pending);
      SOCResourceSet give = new SOCResourceSet(), get = new SOCResourceSet();
      for (int r = 0; r < 5; r++) {
        give.add(n(arr(terms.get("give")).get(r)), RES[r]);
        get.add(n(arr(terms.get("receive")).get(r)), RES[r]);
      }
      boolean[] to = new boolean[4];
      to[seat] = true;
      SOCTradeOffer offer =
          new SOCTradeOffer(game.getName(), n(s.get("current_player")), to, give, get);
      result.put("response", brain.consider(offer));
      result.put("receiver_resources", resourceList(game.getPlayer(seat).getResources()));
    } else if (operation.equals("try_road")) {
      result.put("after", brain.tryRoad(edges[n(request.get("edge"))], this::possiblePiece));
    } else if (operation.equals("bank")) {
      SOCTradeOffer offer = brain.bank(resourceSet(request.get("target")));
      result.put("offer", offer == null ? null : offerAction(offer).get("trade"));
    } else if (operation.equals("discard")) {
      result.put("discard", resourceList(brain.discardCards(n(request.get("count")))));
    } else if (operation.equals("free_resources")) {
      result.put(
          "resources",
          resourceList(
              brain.chooseResources(resourceSet(request.get("target")), n(request.get("count")))));
    } else if (operation.equals("monopoly")) {
      result.put("play", brain.chooseMonopoly());
      result.put("resource", indexOf(RES, brain.monopoly()));
    } else if (operation.equals("knight")) {
      result.put("play", brain.knightForArmy());
    } else if (operation.equals("smart_score")) {
      int[] before = new int[4], after = new int[4];
      for (int p = 0; p < 4; p++) {
        before[p] = n(arr(request.get("before")).get(p));
        after[p] = n(arr(request.get("after")).get(p));
      }
      result.put("score", brain.smartScore(before, after, n(request.get("eta"))));
    } else if (!operation.equals("snapshot")) {
      throw new IllegalArgumentException("unknown diagnostic operation " + operation);
    }
    List<Object> plan = new ArrayList<>();
    SOCBuildPlan currentPlan = brain.planState();
    for (int i = 0; i < currentPlan.getPlanDepth(); i++)
      plan.add(possiblePiece(currentPlan.getPlannedPiece(i)));
    result.put("plan", plan);
    result.put("plans", brain.plans);
    List<Object> players = new ArrayList<>();
    for (int p = 0; p < 4; p++) {
      SOCPlayer data = game.getPlayer(p);
      SOCPlayerTracker tracker = brain.trackers()[p];
      Map<String, Object> entry =
          Json.object(
              "known",
              resourceList(data.getResources()),
              "unknown",
              data.getResources().getAmount(SOCResourceConstants.UNKNOWN));
      List<Object> candidates = new ArrayList<>();
      if (tracker != null) {
        for (SOCPossiblePiece piece : tracker.getPossibleRoads().values())
          candidates.add(possiblePiece(piece));
        for (SOCPossiblePiece piece : tracker.getPossibleSettlements().values())
          candidates.add(possiblePiece(piece));
        for (SOCPossiblePiece piece : tracker.getPossibleCities().values())
          candidates.add(possiblePiece(piece));
        entry.put("win_eta", tracker.getWinGameETA());
        entry.put("road_eta", tracker.getLongestRoadETA());
        entry.put("army_eta", tracker.getLargestArmyETA());
      }
      entry.put("candidates", candidates);
      List<Object> roadNodes = new ArrayList<>();
      for (int node : data.getRoadNodes()) roadNodes.add(nodeInverse.get(node));
      entry.put("road_nodes", roadNodes);
      List<Object> longestPaths = new ArrayList<>();
      for (SOCLRPathData path : data.getLRPaths())
        longestPaths.add(
            Json.object(
                "beginning",
                nodeInverse.get(path.getBeginning()),
                "end",
                nodeInverse.get(path.getEnd()),
                "length",
                path.getLength()));
      entry.put("longest_paths", longestPaths);
      int[] potentials = data.getPotentialSettlements_arr();
      List<Object> order = new ArrayList<>();
      if (potentials != null) for (int node : potentials) order.add(nodeInverse.get(node));
      entry.put("potential_order", order);
      players.add(entry);
    }
    result.put("players", players);
    List<Object> targets = new ArrayList<>(),
        selling = new ArrayList<>(),
        wants = new ArrayList<>();
    for (SOCPossiblePiece target : (SOCPossiblePiece[]) brain.negotiationField("targetPieces"))
      targets.add(target == null ? null : possiblePiece(target));
    for (boolean[] flags : (boolean[][]) brain.negotiationField("isSellingResource")) {
      List<Object> row = new ArrayList<>();
      for (int r : RES) row.add(flags[r]);
      selling.add(row);
    }
    for (boolean[] flags : (boolean[][]) brain.negotiationField("wantsAnotherOffer")) {
      List<Object> row = new ArrayList<>();
      for (int r : RES) row.add(flags[r]);
      wants.add(row);
    }
    result.put(
        "negotiation", Json.object("targets", targets, "selling", selling, "wants_another", wants));
    result.put("nodes", list(nodes));
    result.put("edges", list(edges));
    return result;
  }

  static int indexOf(int[] values, int value) {
    for (int i = 0; i < values.length; i++) if (values[i] == value) return i;
    return -1;
  }

  public static void main(String[] args) throws Exception {
    CarlaAgent adapter = new CarlaAgent(args.length == 0 || !args[0].equals("fast"));
    BufferedReader in = new BufferedReader(new InputStreamReader(System.in, "UTF-8"));
    String line;
    while ((line = in.readLine()) != null) {
      Map<String, Object> m = obj(Json.read(line));
      String type = (String) m.get("type");
      if (type.equals("step")) {
        adapter.step(m);
        continue;
      }
      Map<String, Object> r = Json.object("type", type, "request", m.get("request"));
      try {
        if (type.equals("handshake")) {
          r.put("protocol", "carla-agent");
          r.put("version", 1);
          r.put(
              "capabilities",
              Json.object(
                  "players",
                  Json.array(4),
                  "victory_points",
                  Json.array(10),
                  "domestic_trading",
                  "adapted"));
        } else if (type.equals("before")) adapter.before(m);
        else if (type.equals("decide")) r.put("action", adapter.decide(obj(m.get("view"))));
        else if (type.equals("inspect")) r.put("diagnostic", adapter.inspect(m));
        else if (type.equals("eta")) {
          adapter.synchronize(obj(obj(m.get("view")).get("state")));
          SOCPlayer player = adapter.game.getPlayer(adapter.seat);
          SOCBuildingSpeedEstimate estimate = new SOCBuildingSpeedEstimate(player.getNumbers());
          r.put("periods", list(estimate.getRollsPerResource()));
          r.put(
              "estimates",
              list(estimate.getEstimatesFromNowFast(player.getResources(), player.getPortFlags())));
        }
      } catch (Throwable error) {
        error.printStackTrace(System.err);
        r.put("error", error.toString());
      }
      System.out.println(Json.write(r));
      System.out.flush();
    }
  }
}
