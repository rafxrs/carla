# How the agents play

A short guide to each bot's decision loop. Registry names are in brackets; benchmark setup is
in [benchmarks/README.md](benchmarks/README.md).

## Random (`random`)

The weakest baseline. Uniform over action *types* first, so rare moves (trades, dev cards)
are not drowned out by the many build locations.

```mermaid
flowchart LR
    A[Legal actions] --> B[Pick an action type uniformly] --> C[Pick one action of that type uniformly]
    C -->|domestic offer| D[Random 1-for-1 cards]
```

## CARLA heuristic (`heuristic`)

Greedy rules with no lookahead. Sites are scored by `corner_value`: pips weighted by resource,
plus resource diversity, plus a port bonus. The current **target** is the first goal it has a
place for, in the order city, settlement, road, dev card.

```mermaid
flowchart TD
    S[Main phase] --> C{City buildable?}
    C -->|yes| C1[Best-valued city]
    C -->|no| K{Robber on my tile and knight?}
    K -->|yes| K1[Play knight]
    K -->|no| D{Useful dev card?<br/>Road Building / Year of Plenty}
    D -->|yes| D1[Play it]
    D -->|no| R{No settlement site?}
    R -->|yes| R1[Road toward the best site]
    R -->|no| B{Target is dev card or road?}
    B -->|yes| B1[Buy dev card]
    B -->|no| T{Spare card for a needed one?}
    T -->|yes| T1[Offer 1-for-1 to players, else bank trade]
    T -->|no| E[End turn]
```

Other phases: setup places the best-valued corner and a road toward the best site; the robber
goes where it hurts high-VP opponents most, never on its own tiles; it discards the resource
it holds most; it accepts an offer only if the offer reduces its shortfall for the target.

## Catanatron ValueFunction (`native-f-upstream`, original `catanatron-f`)

One-ply greedy search with a hand-tuned value function: VP, production (own minus the next
player's), reachable production, buildable nodes, hand synergy, dev cards, army size.

```mermaid
flowchart LR
    A[Legal actions] --> B[For each: copy the game, apply the action<br/>chance resolved by ONE random sample]
    B --> C[Score the result with the value function] --> D[Play the highest score]
```

The single sample is the upstream behaviour. The plain `native-f` averages over all chance
outcomes instead.

## Catanatron AlphaBeta (`native-ab2-upstream`, original `catanatron-ab2`)

The same value function at the leaves of a depth-2 search: our move, then the next decision.
That decision is maximised if it is ours and minimised if it is an opponent's.

```mermaid
flowchart TD
    A[Legal actions] --> B[Expand each action's chance outcomes<br/>dice, steals, dev-card draws]
    B --> C[Search 2 decisions deep with alpha-beta pruning]
    C --> D[Leaves scored by the value function]
    D --> E[Play the action with the best expected value]
```

Never offers domestic trades and rejects all offers.

## JSettlers FAST (`jsettlers-fast-native`, original `jsettlers-fast-adapted`)

Plans around **building speed**: how many rolls until it can afford each piece, given its
production and ports.

```mermaid
flowchart TD
    A[Turn] --> P{Building plan empty?}
    P -->|yes| Q[Plan: the fastest-to-afford of city,<br/>settlement plus its roads, dev card,<br/>or roads toward Longest Road]
    P -->|no| T
    Q --> T{Can afford the plan?}
    T -->|yes| U[Build it]
    T -->|no| V[Offer a player trade toward it,<br/>else bank or port trade, else end turn]
```

## JSettlers SMART (`jsettlers-smart-native`, original `jsettlers-smart-adapted`)

FAST's machinery plus **win-time projection**: for every player, an estimate of how many rolls
they need to win. Each candidate piece is scored by how much it speeds up its own win and slows
the leader.

```mermaid
flowchart TD
    A[Turn] --> B[Project every player's win time]
    B --> C[For each candidate piece and dev card:<br/>re-project with it placed]
    C --> D[Score = own speed-up + slowing the leader,<br/>discounted by time to afford]
    D --> E[Plan the best, then build, trade or end turn as in FAST]
```

Shared by FAST and SMART:
- **Opening:** pick settlement pairs by number and port scores, point the road at the next site.
- **Robber:** target the player projected to win soonest, on the hex whose loss slows their
  building most.
- **Negotiation:** offer trades toward the plan; accept only if it shortens their own build
  time, the partner is not close to winning, and the partner is not racing them for the same
  spot.
