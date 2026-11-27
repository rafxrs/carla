"""Figures that explain the heuristic bot (src/carla/agents/heuristic.py).

Plays games between four heuristic bots, counts every branch of their decisions, then draws:
  heuristic_setup.png  how it places its starting settlements and roads, on an example board
  heuristic_turn.png   its turn as a state machine with measured branch frequencies (and .svg)

Needs Graphviz (`dot`; set CARLA_DOT to its path if it is not found). Run from the repo root:
  python scripts/heuristic_figures.py --games 1000 --workers 4 --output docs/img
"""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from collections import Counter
from math import sqrt
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon

from carla.agents.heuristic import RESOURCE_WEIGHTS, HeuristicAgent, corner_value
from carla.config.loader import load_config
from carla.engine.actions import ActionType as A
from carla.engine.config import EngineConfig
from carla.engine.constants import DESERT, GENERIC_PORT, PIPS, DevCard
from carla.engine.events import EventKind as E
from carla.engine.game import Game
from carla.engine.rng import derive_seed
from carla.engine.rules import settlement_corners
from carla.engine.state import Phase
from carla.engine.topology import (
    CORNER_TILES,
    CORNER_XY,
    EDGE_CORNERS,
    EDGE_XY,
    PORT_EDGES,
    TILE_CORNERS,
    TILE_XY,
)

DEFAULT_CONFIG = "configs/experiments/full_trade_own_heuristic.yaml"
"""Its engine rules are the ones the full model trains under (domestic trade on)."""
STATS_SEED = 40_000

FONT = "Segoe UI"
SURFACE, INK, INK_2, MUTED, BORDER = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
PANEL, PANEL_2 = "#f0efec", "#e4e2dc"
SEATS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # categorical slots 1-4
RESOURCE_FILL = ("#5f8f4e", "#c26a4a", "#b7d39a", "#e8cf6e", "#9a9a9e")  # lumber..ore
DESERT_FILL = "#e9dcc0"
RESOURCE_NAMES = ("lumber", "brick", "wool", "grain", "ore")
HEAT = ("#ece9fb", "#c6bff2", "#9c90e6", "#6250d6", "#3a2c99")  # violet, light to dark


def find_dot() -> str:
    candidates = [os.environ.get("CARLA_DOT"), r"C:\Program Files\Graphviz\bin\dot.exe"]
    for path in candidates:
        if path and Path(path).exists():
            return path
    found = shutil.which("dot")
    if found is None:
        raise SystemExit("Graphviz `dot` not found; install Graphviz or set CARLA_DOT")
    return found


def pct(x: float) -> str:
    if 0 < x < 0.0005:
        return "&lt;0.1%"
    return f"{100 * x:.1f}%" if x < 0.995 else f"{100 * x:.0f}%"


def render(dot: str, out: Path, fmt: str = "png") -> Path:
    """Render Graphviz source to `out` with the format's suffix."""
    target = out.with_suffix(f".{fmt}")
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "figure.dot"
        source.write_text(dot, encoding="utf-8")
        args = [find_dot(), f"-T{fmt}", str(source), "-o", str(target)]
        if fmt == "png":
            args.insert(1, "-Gdpi=200")
        subprocess.run(args, check=True)
    return target


# ---------------------------------------------------------------------------- statistics


def _play(job: tuple[EngineConfig, int]) -> tuple[dict, list[dict]]:
    """One game of four heuristic bots: branch counts and every setup pick."""
    engine, index = job
    seed = derive_seed(STATS_SEED, "heuristic-stats", index)
    game = Game(engine, seed)
    agents = [HeuristicAgent() for _ in range(4)]
    for seat, agent in enumerate(agents):
        agent.reset(game, seat, derive_seed(seed, "seat", seat))
    c: Counter = Counter()
    setups = []
    last_main = None  # (seat, label) of the previous main-phase decision
    while not game.is_over:
        s = game.state
        seat, phase = game.decision_player, game.phase
        legal = game.legal_actions()
        action = agents[seat].act(game, seat, legal)
        t = action.type
        label = t.name if t != A.PLAY_DEV_CARD else f"PLAY_{DevCard(action.dev).name}"
        c[f"{phase.name}|{label}"] += 1
        c[f"phase|{phase.name}"] += 1
        if phase == Phase.SETUP_SETTLEMENT:
            values = sorted(corner_value(s.board, a.corner) for a in legal)
            corner = action.corner
            tiles = [x for x in CORNER_TILES[corner] if s.board.tile_resource[x] != DESERT]
            port = s.board.corner_port[corner]
            setups.append(
                {
                    "round": 1 if s.setup_step < 4 else 2,
                    "value": corner_value(s.board, corner),
                    "options": len(values),
                    "kinds": len({s.board.tile_resource[x] for x in tiles}),
                    "port": "none" if port < 0 else "3:1" if port == GENERIC_PORT else "2:1",
                    "top_share": values.count(values[-1]),
                }
            )
        if phase == Phase.MAIN:
            c["main|decisions"] += 1
            if not settlement_corners(s, seat):
                c["main|no_site"] += 1
            if last_main is not None and last_main[0] == seat:
                c[f"trans|{last_main[1]}|{label}"] += 1
            last_main = (seat, label) if t != A.END_TURN else None
        for event in game.step(action):
            if event.kind == E.DICE_ROLLED:
                c["rolls"] += 1
                c[f"roll|{event.value}"] += 1
        if phase == Phase.PRE_ROLL and t == A.ROLL_DICE:
            c[f"after_roll|{game.phase.name}"] += 1
        if phase == Phase.MOVE_ROBBER:
            c["robber|steal" if action.player >= 0 else "robber|nosteal"] += 1
    c["games"] += 1
    return dict(c), setups


def collect_stats(engine: EngineConfig, games: int, workers: int) -> dict:
    total: Counter = Counter()
    setups: list[dict] = []
    with Pool(workers) as pool:
        for counts, picks in pool.imap_unordered(_play, [(engine, i) for i in range(games)]):
            total.update(counts)
            setups.extend(picks)
    return {"counts": dict(total), "setups": setups}


# ---------------------------------------------------------------------------- figure 1


def setup_flow(stats: dict) -> str:
    setups = stats["setups"]

    def share(round_, test):
        rows = [s for s in setups if s["round"] == round_]
        return sum(test(s) for s in rows) / len(rows)

    def mean(round_, key):
        rows = [s for s in setups if s["round"] == round_]
        return sum(s[key] for s in rows) / len(rows)

    weights = ", ".join(f"{n} {w:g}" for n, w in zip(RESOURCE_NAMES, RESOURCE_WEIGHTS, strict=True))
    r1 = (
        f"mean score {mean(1, 'value'):.1f} of {mean(1, 'options'):.0f} legal corners<br/>"
        f"3 resource types {pct(share(1, lambda s: s['kinds'] == 3))}, "
        f"port {pct(share(1, lambda s: s['port'] != 'none'))}"
    )
    r2 = (
        f"mean score {mean(2, 'value'):.1f} of {mean(2, 'options'):.0f} legal corners<br/>"
        f"3 resource types {pct(share(2, lambda s: s['kinds'] == 3))}, "
        f"port {pct(share(2, lambda s: s['port'] != 'none'))}"
    )
    ties = (
        share(1, lambda s: s["top_share"] > 1) * 0.5 + share(2, lambda s: s["top_share"] > 1) * 0.5
    )
    return f"""digraph setup {{
  graph [rankdir=TB, bgcolor="{SURFACE}", nodesep=0.35, ranksep=0.32, pad=0.2,
         fontname="{FONT}"];
  node  [shape=box, style="rounded,filled", fillcolor="{PANEL}", color="{BORDER}",
         fontname="{FONT}", fontsize=12, fontcolor="{INK}", margin="0.18,0.10", penwidth=1.2];
  edge  [color="{INK_2}", fontname="{FONT}", fontsize=11, fontcolor="{INK_2}", penwidth=1.3,
         arrowsize=0.8];

  order [label=<<b>Setup order</b><br/>snake draft: seats 0, 1, 2, 3, then 3, 2, 1, 0<br/>
each turn = one settlement, then one road next to it>, fillcolor="{PANEL_2}"];
  legal [label=<<b>1. List legal corners</b><br/>empty, and no building on a neighbouring corner<br/>
(distance rule; nothing else is filtered out)>];
  score [label=<<b>2. Score every legal corner</b><br align="left"/>
<font point-size="11">score = <b>Σ</b> over adjacent tiles (desert skipped) of pips(number) × weight(resource)<br align="left"/>
&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;+ 1.5 × number of different resources<br align="left"/>
&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;+ port bonus: 3:1 port +1.0, 2:1 port +1.5<br align="left"/>
pips: 2,12 = 1 · 3,11 = 2 · 4,10 = 3 · 5,9 = 4 · 6,8 = 5<br align="left"/>
weights: {weights}<br align="left"/></font>>, fillcolor="#ffffff"];
  pick [label=<<b>3. Take the highest score</b><br/>ties go to the first corner in the legal list<br/>({pct(ties)} of picks are ties)>];
  road [label=<<b>4. Place the road</b><br align="left"/>
<font point-size="11">for each edge touching the new settlement:<br align="left"/>
&#160;&#160;road score = best free site at the edge's far corner (full score)<br align="left"/>
&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;or one corner further (half score)<br align="left"/>
the far corner always touches the new settlement, so in practice<br align="left"/>
the road points at the best site two corners away, at half value<br align="left"/></font>>,
        fillcolor="#ffffff"];
  out1 [label=<<b>First settlement</b> (round 1)<br/><font point-size="11">{r1}</font>>,
        fillcolor="{PANEL_2}"];
  out2 [label=<<b>Second settlement</b> (round 2)<br/><font point-size="11">{r2}</font>>,
        fillcolor="{PANEL_2}"];
  ignores [shape=note, fillcolor="#fff6e0", color="#e2c98a", label=<<b>What it does not consider</b><br align="left"/>
<font point-size="11">• which resources its first settlement already covers<br align="left"/>
• the starting cards its second settlement pays out<br align="left"/>
• where opponents are likely to build next<br align="left"/>
• spreading its numbers (two 6s count the same as a 6 and an 8)<br align="left"/>
The same scoring is reused for every settlement later in the game.<br align="left"/></font>>];

  order -> legal -> score -> pick -> road;
  road -> out1 [label=" round 1 "];
  road -> out2 [label=" round 2 "];
  out1 -> ignores [style=invis];
  {{rank=same; out1; out2}}
}}
"""


def example_setup(engine: EngineConfig, seed: int):
    game = Game(engine, seed)
    agents = [HeuristicAgent() for _ in range(4)]
    for seat, agent in enumerate(agents):
        agent.reset(game, seat, derive_seed(seed, "seat", seat))
    board = game.state.board
    start_values = {c: corner_value(board, c) for c in range(len(CORNER_XY))}
    picks, roads = [], []
    while game.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
        seat = game.decision_player
        legal = game.legal_actions()
        action = agents[seat].act(game, seat, legal)
        if action.type == A.BUILD_SETTLEMENT:
            values = sorted((corner_value(board, a.corner) for a in legal), reverse=True)
            picks.append((seat, action.corner, values[0], len(legal)))
        else:
            roads.append((seat, action.edge))
        game.step(action)
    return board, start_values, picks, roads


def flip(xy):
    return xy[0], -xy[1]


def draw_board(ax, board, start_values, picks, roads) -> None:
    ax.set_facecolor(SURFACE)
    for t, corners in enumerate(TILE_CORNERS):
        r = board.tile_resource[t]
        fill = DESERT_FILL if r == DESERT else RESOURCE_FILL[r]
        ax.add_patch(
            Polygon(
                [flip(CORNER_XY[c]) for c in corners],
                closed=True,
                facecolor=fill,
                edgecolor=SURFACE,
                linewidth=2.5,
                zorder=1,
            )
        )
        x, y = flip(TILE_XY[t])
        n = board.tile_number[t]
        if r == DESERT:
            ax.text(x, y, "desert", ha="center", va="center", fontsize=8, color=INK_2, zorder=3)
            continue
        ax.add_patch(Circle((x, y), 0.36, facecolor="#fffdf7", edgecolor=BORDER, lw=1, zorder=2))
        hot = n in (6, 8)
        ax.text(
            x,
            y + 0.06,
            str(n),
            ha="center",
            va="center",
            fontsize=12,
            weight="bold",
            color="#b3261e" if hot else INK,
            zorder=3,
        )
        ax.text(
            x,
            y - 0.2,
            "•" * PIPS[n],
            ha="center",
            va="center",
            fontsize=6,
            color="#b3261e" if hot else INK_2,
            zorder=3,
        )
        ax.text(
            x,
            y - 0.6,
            RESOURCE_NAMES[r],
            ha="center",
            va="center",
            fontsize=7,
            color=INK if r in (2, 3) else "#ffffff",
            zorder=3,
        )

    for slot, e in enumerate(PORT_EDGES):
        kind = board.port_types[slot]
        (x1, y1), (x2, y2) = (flip(CORNER_XY[c]) for c in EDGE_CORNERS[e])
        ax.plot([x1, x2], [y1, y2], color=INK_2, lw=4, solid_capstyle="round", zorder=2)
        mx, my = flip(EDGE_XY[e])
        label = "3:1" if kind == GENERIC_PORT else f"2:1\n{RESOURCE_NAMES[kind]}"
        ax.text(mx * 1.2, my * 1.2, label, ha="center", va="center", fontsize=7.5, color=INK_2)

    values = sorted(start_values.values())
    lo, hi = values[0], values[-1]
    top = sorted(start_values, key=start_values.get, reverse=True)[:10]
    for c, xy in enumerate(CORNER_XY):
        x, y = flip(xy)
        v = start_values[c]
        step = min(len(HEAT) - 1, int((v - lo) / (hi - lo + 1e-9) * len(HEAT)))
        ax.add_patch(Circle((x, y), 0.13, facecolor=HEAT[step], edgecolor=SURFACE, lw=1, zorder=4))
        if c in top:
            ax.text(
                x + 0.17,
                y + 0.17,
                f"{v:.1f}",
                fontsize=7.5,
                color=INK,
                zorder=7,
                bbox={"boxstyle": "round,pad=0.12", "fc": "#ffffff", "ec": BORDER, "lw": 0.6},
            )

    for seat, e in roads:
        (x1, y1), (x2, y2) = (flip(CORNER_XY[c]) for c in EDGE_CORNERS[e])
        ax.plot([x1, x2], [y1, y2], color=SEATS[seat], lw=6, solid_capstyle="round", zorder=5)
    for order, (seat, corner, _, _) in enumerate(picks, start=1):
        x, y = flip(CORNER_XY[corner])
        ax.add_patch(
            Circle((x, y), 0.27, facecolor=SEATS[seat], edgecolor=SURFACE, lw=2.5, zorder=6)
        )
        ax.text(
            x,
            y,
            str(order),
            ha="center",
            va="center",
            fontsize=10,
            weight="bold",
            color="#ffffff",
            zorder=7,
        )

    handles = [
        Line2D([], [], marker="o", ls="", ms=11, mfc=SEATS[s], mec=SURFACE, label=f"seat {s}")
        for s in range(4)
    ]
    handles += [
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            ms=8,
            mfc=HEAT[i],
            mec=SURFACE,
            label=("lowest" if i == 0 else "highest" if i == len(HEAT) - 1 else " ") + " score",
        )
        for i in (0, len(HEAT) - 1)
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        ncol=6,
        frameon=False,
        fontsize=8.5,
        bbox_to_anchor=(0.5, -0.06),
        handletextpad=0.3,
        columnspacing=1.0,
    )
    ax.set_xlim(-5.6 * sqrt(3) / 1.6, 5.6 * sqrt(3) / 1.6)
    ax.set_ylim(-5.3, 5.0)
    ax.set_aspect("equal")
    ax.axis("off")


def breakdown(board, corner) -> list[str]:
    lines = []
    kinds = set()
    for t in CORNER_TILES[corner]:
        r = board.tile_resource[t]
        if r == DESERT:
            lines.append("desert: 0")
            continue
        n = board.tile_number[t]
        kinds.add(r)
        lines.append(
            f"{RESOURCE_NAMES[r]} {n}: {PIPS[n]} pips × {RESOURCE_WEIGHTS[r]:g} = {PIPS[n] * RESOURCE_WEIGHTS[r]:.1f}"
        )
    lines.append(f"{len(kinds)} resource types: + {1.5 * len(kinds):.1f}")
    port = board.corner_port[corner]
    if port >= 0:
        lines.append(f"port: + {1.0 if port == GENERIC_PORT else 1.5:.1f}")
    lines.append(f"score = {corner_value(board, corner):.1f}")
    return lines


def figure_setup(stats: dict, engine: EngineConfig, out_dir: Path, seed: int) -> Path:
    with tempfile.TemporaryDirectory() as tmp:
        flow = mpimg.imread(render(setup_flow(stats), Path(tmp) / "flow"))
    board, start_values, picks, roads = example_setup(engine, seed)
    fig = plt.figure(figsize=(19, 10.5), dpi=150, facecolor=SURFACE)
    left = fig.add_axes((0.0, 0.0, 0.39, 0.92))
    left.imshow(flow)
    left.axis("off")
    right = fig.add_axes((0.40, 0.05, 0.45, 0.84))
    draw_board(right, board, start_values, picks, roads)
    fig.text(
        0.015,
        0.965,
        "How the heuristic bot places its starting settlements and roads",
        fontsize=17,
        weight="bold",
        color=INK,
        ha="left",
    )
    fig.text(
        0.42,
        0.905,
        f"Example board (seed {seed}). Big circles: the 8 setup settlements in the order "
        "they were placed, coloured by seat.\nSmall dots: corner score on the empty board "
        "(darker = higher); the 10 best scores are labelled.",
        fontsize=10.5,
        color=INK_2,
        ha="left",
        va="bottom",
        linespacing=1.5,
    )
    first = picks[0]
    text = "\n".join(["Settlement 1, worked example", "", *breakdown(board, first[1])])
    fig.text(
        0.855,
        0.70,
        text,
        fontsize=9,
        color=INK,
        ha="left",
        va="top",
        family="monospace",
        linespacing=1.45,
        bbox={"boxstyle": "round,pad=0.6", "fc": "#ffffff", "ec": BORDER, "lw": 0.8},
    )
    second = [p for p in picks if p[0] == picks[0][0]][1]
    fig.text(
        0.855,
        0.42,
        "Seat 0's picks\n\n"
        f"round 1: score {picks[0][2]:.1f}\n  (best of {picks[0][3]} corners)\n"
        f"round 2: score {second[2]:.1f}\n  (best of {second[3]} corners)\n\n"
        "Later picks score lower:\nthe best corners and their\nneighbours are taken.",
        fontsize=9,
        color=INK,
        ha="left",
        va="top",
        family="monospace",
        linespacing=1.45,
        bbox={"boxstyle": "round,pad=0.6", "fc": "#ffffff", "ec": BORDER, "lw": 0.8},
    )
    fig.text(
        0.015,
        0.012,
        f"Statistics from {stats['counts']['games']} games between four heuristic bots "
        "(training rules, domestic trade on). Source: src/carla/agents/heuristic.py",
        fontsize=8.5,
        color=MUTED,
        ha="left",
    )
    path = out_dir / "heuristic_setup.png"
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------- figure 2


def turn_machine(stats: dict) -> str:
    c = stats["counts"]
    rolls = c["rolls"]
    pre = c["phase|PRE_ROLL"]
    main = c["main|decisions"]
    turns = rolls
    m = {k.split("|", 1)[1]: v / main for k, v in c.items() if k.startswith("MAIN|")}
    responses = c["phase|TRADE_RESPONSE"]
    confirms = c["phase|TRADE_CONFIRM"]
    after = {k.split("|")[1]: v / rolls for k, v in c.items() if k.startswith("after_roll|")}
    knight_pre = c["PRE_ROLL|PLAY_KNIGHT"] / pre
    steal = c["robber|steal"] / c["phase|MOVE_ROBBER"]
    acted = 1 - m.get("PROPOSE_TRADE", 0) - m.get("END_TURN", 0)

    # what follows each MAIN action within the same turn
    follow = {}
    for key, v in c.items():
        if key.startswith("trans|"):
            _, a, b = key.split("|")
            follow.setdefault(a, {})[b] = follow.setdefault(a, {}).get(b, 0) + v
    short = {
        "BUILD_SETTLEMENT": "settlement",
        "BUILD_CITY": "city",
        "BUILD_ROAD": "road",
        "BUY_DEV_CARD": "dev card",
        "PROPOSE_TRADE": "propose",
        "MARITIME_TRADE": "bank trade",
        "END_TURN": "end turn",
        "PLAY_ROAD_BUILDING": "road building",
        "PLAY_YEAR_OF_PLENTY": "year of plenty",
        "PLAY_KNIGHT": "knight",
    }
    follow_rows = []
    for a in (
        "PROPOSE_TRADE",
        "MARITIME_TRADE",
        "BUILD_ROAD",
        "BUILD_CITY",
        "BUILD_SETTLEMENT",
        "BUY_DEV_CARD",
    ):
        nxt = follow.get(a, {})
        total = sum(nxt.values())
        top = sorted(nxt.items(), key=lambda kv: -kv[1])[:3]
        cells = ", ".join(f"{short[b]} {pct(v / total)}" for b, v in top)
        follow_rows.append(
            f'<tr><td align="left"><b>{short[a]}</b></td><td align="left">{cells}</td></tr>'
        )

    rules = [
        (
            "1",
            "Build a settlement",
            "any legal site; highest corner score",
            m.get("BUILD_SETTLEMENT", 0),
        ),
        (
            "2",
            "Build a city",
            "on its settlement with the highest corner score",
            m.get("BUILD_CITY", 0),
        ),
        (
            "3",
            "Play a knight",
            "only if the robber sits next to one of its buildings",
            m.get("PLAY_KNIGHT", 0),
        ),
        (
            "4",
            "Play Road Building",
            "only if it has no free settlement site",
            m.get("PLAY_ROAD_BUILDING", 0),
        ),
        (
            "",
            "Play Year of Plenty",
            "only if both cards fill a shortfall for the target",
            m.get("PLAY_YEAR_OF_PLENTY", 0),
        ),
        (
            "5",
            "Build a road",
            "only if no free site; toward the best site two corners away",
            m.get("BUILD_ROAD", 0),
        ),
        (
            "6",
            "Buy a development card",
            "only if the target is a dev card or a road",
            m.get("BUY_DEV_CARD", 0),
        ),
        (
            "7",
            "Propose a 1-for-1 trade",
            "once per turn: 1 spare card for 1 missing target card",
            m.get("PROPOSE_TRADE", 0),
        ),
        (
            "8",
            "Trade with the bank or a port",
            "give a surplus resource for a missing one",
            m.get("MARITIME_TRADE", 0),
        ),
        ("9", "End the turn", "nothing above applies", m.get("END_TURN", 0)),
    ]
    rows = "".join(
        f'<tr><td align="right"><font color="{MUTED}">{n or "&#160;"}</font></td>'
        f'<td align="left" port="r{i}"><b>{name}</b><br align="left"/><font point-size="10" color="{INK_2}">{cond}</font><br align="left"/></td>'
        f'<td align="right"><b>{pct(p)}</b></td></tr>'
        for i, (n, name, cond, p) in enumerate(rules)
    )
    return f"""digraph turn {{
  graph [rankdir=TB, bgcolor="{SURFACE}", nodesep=0.45, ranksep=0.5, pad=0.25, newrank=true,
         fontname="{FONT}", labelloc=t, fontsize=20, fontcolor="{INK}",
         label=<<b>How the heuristic bot plays a turn</b><br/><font point-size="12" color="{INK_2}">deterministic rules; percentages are how often each branch was taken in {c['games']} heuristic-only games ({turns / c['games']:.0f} turns and {main / turns:.1f} main-phase decisions per turn on average)</font><br/> >];
  node  [shape=box, style="rounded,filled", fillcolor="{PANEL}", color="{BORDER}",
         fontname="{FONT}", fontsize=12, fontcolor="{INK}", margin="0.16,0.09", penwidth=1.2];
  edge  [color="{INK_2}", fontname="{FONT}", fontsize=11, fontcolor="{INK}", penwidth=1.4,
         arrowsize=0.8];

  start [shape=circle, label="turn\\nstarts", fillcolor="{PANEL_2}", fontsize=11, width=0.8, fixedsize=true];
  preroll [label=<<b>Before rolling</b><br/>knight in hand and the robber<br/>next to one of its buildings?>];
  knight1 [label=<<b>Play knight</b>>];
  roll [label=<<b>Roll the dice</b>>];
  discard [label=<<b>Discard</b> (every player above 7 cards)<br/><font point-size="10">one card at a time: the resource it holds most of</font>>];
  robber [label=<<b>Move the robber</b><br align="left"/><font point-size="10">pick the tile maximising<br align="left"/>
pips × Σ buildings on it: opponent's (1 or 2) × (1 + their VP),<br align="left"/>
&#160;&#160;&#160;its own −10 each<br align="left"/>
then steal from the victim with the most VP (+ cards/10)<br align="left"/></font>>];
  main [shape=plain, label=<<table border="1" cellborder="0" cellspacing="0" cellpadding="5" bgcolor="#ffffff" color="{BORDER}" style="rounded">
<tr><td colspan="3" bgcolor="{PANEL_2}"><b>Main phase: the first rule that applies is played</b><br/><font point-size="10" color="{INK_2}">share of all {main:,} main-phase decisions</font></td></tr>
{rows}</table>>];
  target [shape=note, fillcolor="#fff6e0", color="#e2c98a", label=<<b>Target</b> (what it is saving for)<br align="left"/>
<font point-size="10">1. a city, if it has a settlement to upgrade<br align="left"/>
2. a settlement, if a free site is connected<br align="left"/>
3. a road, if it has roads and settlements left<br align="left"/>
4. otherwise a development card<br align="left"/>
Trades, Year of Plenty and dev-card buying all<br align="left"/>
aim at this target's cost.<br align="left"/></font>>];
  respond [label=<<b>Each other player responds</b><br align="left"/><font point-size="10">
accept if the offer shrinks its own shortfall<br align="left"/>
else counter with its own 1-for-1<br align="left"/>
else reject<br align="left"/></font>>];
  confirm [label=<<b>Proposer chooses</b><br align="left"/><font point-size="10">take the acceptance or counter that<br align="left"/>
shrinks its shortfall most, else cancel<br align="left"/></font>>];
  act [label=<<b>Do it</b><br/><font point-size="10">build, buy, play the card<br/>or trade with the bank</font>>];
  end [shape=doublecircle, label="turn\\nends", fillcolor="{PANEL_2}", fontsize=11, width=0.8, fixedsize=true];
  follow [shape=plain, label=<<table border="1" cellborder="0" cellspacing="0" cellpadding="4" bgcolor="#ffffff" color="{BORDER}" style="rounded">
<tr><td colspan="2" bgcolor="{PANEL_2}"><b>What it does next</b><br/><font point-size="10" color="{INK_2}">next main-phase action in the same turn</font></td></tr>
{''.join(follow_rows)}</table>>];
  never [shape=note, fillcolor="#fff6e0", color="#e2c98a", label=<<b>Never does</b><br align="left"/>
<font point-size="10">play Monopoly · ask others for offers (invite)<br align="left"/>
offer more than 1 card for 1 · save cards on purpose<br align="left"/>
(it passes only when nothing is affordable)<br align="left"/></font>>];

  start -> preroll;
  preroll -> knight1 [label=< yes  <b>{pct(knight_pre)}</b> >];
  preroll -> roll [label=< no  <b>{pct(1 - knight_pre)}</b> >];
  knight1 -> robber [label=" then robber, then roll "];
  roll -> main [label=< not 7  <b>{pct(after.get('MAIN', 0))}</b> >];
  roll -> discard [label=< 7, someone over 7 cards  <b>{pct(after.get('DISCARD', 0))}</b> >];
  roll -> robber [label=< 7, nobody over 7  <b>{pct(after.get('MOVE_ROBBER', 0))}</b> >];
  discard -> robber;
  robber -> main [label=< steals a card  <b>{pct(steal)}</b> >];
  main -> respond [label=< rule 7: propose  <b>{pct(m.get('PROPOSE_TRADE', 0))}</b> >];
  respond -> confirm [label=< per response: accept <b>{pct(c['TRADE_RESPONSE|ACCEPT_TRADE'] / responses)}</b>, counter <b>{pct(c['TRADE_RESPONSE|PROPOSE_TRADE'] / responses)}</b>, reject <b>{pct(c['TRADE_RESPONSE|REJECT_TRADE'] / responses)}</b> >];
  confirm -> main [label=< trade made <b>{pct(c['TRADE_CONFIRM|CONFIRM_TRADE'] / confirms)}</b>, cancelled <b>{pct(c['TRADE_CONFIRM|CANCEL_TRADE'] / confirms)}</b> >, constraint=false];
  main -> act [label=< rules 1-6, 8  <b>{pct(acted)}</b> >];
  act -> main [label=< decide again >];
  main -> end [label=< rule 9  <b>{pct(m.get('END_TURN', 0))}</b> >];
  target -> main [style=dashed, color="#c9a95a", arrowhead=none];
  main -> follow [style=invis];
  {{rank=same; knight1; roll}}
  {{rank=same; discard; robber}}
  {{rank=same; target; main; act}}
  {{rank=same; respond; end}}
  {{rank=same; follow; never}}
  respond -> follow [style=invis];
}}
"""


def figure_turn(stats: dict, out_dir: Path) -> Path:
    dot = turn_machine(stats)
    render(dot, out_dir / "heuristic_turn", "svg")
    return render(dot, out_dir / "heuristic_turn", "png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Figures that explain the heuristic bot.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="experiment config (engine rules)")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7, help="board of the setup example")
    parser.add_argument("--output", type=Path, default=Path("docs/img"))
    parser.add_argument("--stats", type=Path, help="reuse (or save) the game statistics here")
    args = parser.parse_args()
    engine = load_config(args.config, []).engine
    if args.stats is not None and args.stats.exists():
        stats = json.loads(args.stats.read_text(encoding="utf-8"))
    else:
        stats = collect_stats(engine, args.games, args.workers)
        if args.stats is not None:
            args.stats.write_text(json.dumps(stats), encoding="utf-8")
    args.output.mkdir(parents=True, exist_ok=True)
    print(figure_setup(stats, engine, args.output, args.seed))
    print(figure_turn(stats, args.output))


if __name__ == "__main__":
    main()
