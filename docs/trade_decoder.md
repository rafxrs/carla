# Neural domestic-trade offers

`configs/experiments/ppo_mlp_trade_decoder.yaml` enables domestic trading and sets
`model.trade_decoder: gru`. The same decoder also attaches to the GNN encoder; the MLP
configuration is the small, runnable starting point. `model.trade_decoder: none` keeps
the original one-for-one policy and its checkpoint architecture intact. The Transformer
setting remains unsupported.

To check the resolved settings and run one small training update from the repository root:

```bash
carla config configs/experiments/ppo_mlp_trade_decoder.yaml
carla train configs/experiments/ppo_mlp_trade_decoder.yaml ppo.total_env_steps=128 ppo.rollout_steps=32 ppo.epochs=1 ppo.minibatch_size=128 model.hidden_dim=32 engine.max_turns=60
```

This checks the training path, not whether the agent has learned to trade well. For a
full run, use the config without these overrides and measure throughput first.

## Where to read the code

| File | Responsibility |
| --- | --- |
| `envs/trade_tokens.py` | Convert between an offer and its unique token sequence. |
| `envs/observation.py` | Expose the acting player's exact hand and the engine card limit. |
| `models/trade_decoder.py` | Generate or rescore tokens with legal-prefix masks. |
| `models/distributions.py` | Join the action-type probability with the offer probability. |
| `envs/action_space.py` | Turn the saved action vector back into an engine action. |
| `envs/trade_action_wrapper.py` | Declare decoder actions to Gymnasium clients. |
| `rl/rollout.py`, `rl/ppo.py` | Store actions and use their scores in PPO. |

All paths in this table are under `src/carla/`. The engine's final legality check is
in `engine/rules.py`.

## Rules represented by the policy

An offer contains two five-resource count vectors from the **offer maker's** perspective.
This applies to an active player's proposal and a responder's counteroffer. Each side
must have at least one card, no resource type may appear on both sides, the maker must
hold everything it gives, and each side must respect the engine's configured cap.
The other player's hidden resource composition is never a policy input or an offer
legality mask. The engine still validates each submitted offer and checks the responder's
own hand on acceptance. The active player chooses one accepted offer or counteroffer
to confirm; only then do cards move.

CATAN does not prescribe CARLA's default three proposal rounds, three completed
domestic exchanges per turn, or four cards per side. These are configurable experiment
limits for bounded games. Bank and port trades keep their existing policy heads.

## Action format and probability

The type head chooses `PROPOSE_TRADE`; this action is also used for a counteroffer in
`TRADE_RESPONSE`. A GRU then emits resource tokens `0..4`, `SWITCH=5`, more resource
tokens, and `STOP=6`. Resource IDs must be nondecreasing within each side. Therefore
an offer has exactly one sequence. For example, give two lumber and one wool, receive
one brick and two ore is `0,0,2,5,1,4,4,6` in the project's resource order.

The decoder masks each next token using the actor's exact integer hand, the sequence
prefix, and the effective per-side cap. The effective cap is the smaller of the
engine limit and `model.max_trade_cards_per_side` (default four); if the engine limit
is unlimited, the model cap still applies. A four-card decoder has at most ten tokens.
The observation carries exact own-hand counts and the public engine limit separately
from the normalized hand feature. Older MLP/GNN feature dimensions are unchanged.

Policy actions store the original eight head indices plus padded trade tokens. For a
non-trade action the token part is all `-1` and contributes no probability. For an
offer, PPO stores the exact sampled tokens and their total log probability, then
teacher-forces those tokens at update time. The joint log probability is the type
head's log probability plus the sum of conditional token log probabilities. The old
one-for-one trade subhead is removed from the score for decoder offers. This prevents
duplicate action encodings and keeps PPO ratios valid. The decoder's entropy term is
a sampled conditional-token regularizer; the baseline factored distribution retains
its exact entropy for non-trade branches. With the decoder enabled,
`head_trade_entropy` reports conditional token entropy along sampled offer prefixes,
not entropy of the unused one-for-one head. Calling `entropy()` without first sampling
or scoring an action draws a prefix for this estimate.

## Gymnasium action space

Plain `CatanEnv` keeps the eight-head action space for the baseline policy. Wrap it
when using decoder actions with Gymnasium code that checks `action_space.contains`:

```python
from carla.config import load_config
from carla.envs import CatanEnv, TradeActionWrapper

config = load_config("configs/experiments/ppo_mlp_trade_decoder.yaml")
env = TradeActionWrapper(
    CatanEnv(config.engine, config.env),
    max_cards=config.model.max_trade_cards_per_side,
)
obs, info = env.reset(seed=0)
action = env.action_space.sample()  # legal, padded to the decoder's width
assert env.action_space.contains(action)
env.step(action)
```

The wrapper uses a standard fixed-width `MultiDiscrete` space with `-1` for unused
heads and padding. Set its cap to the policy's model cap; decoder outputs then have
the declared shape. `contains` checks shape and ranges, while the observation masks
and engine check whether a choice is legal in the current game. Generic per-head
Gymnasium sampling masks cannot express those conditional rules, so the wrapper's
`sample()` draws from the current legal actions instead. External trainers must still
use CARLA's conditional masks when choosing actions.

## Training, evaluation, and hardware

The existing batched rollout collector, CPU worker pool, PPO learner, checkpoints,
fixed-bot evaluation, and checkpoint tournament share the structured action format.
The rollout arrays include the tokens, so no special-case replay or PPO update code
is needed. Both MLP and GNN produce a context vector for the same GRU.

Use `runtime.device: cpu` for local tests or `runtime.device: auto`/`cuda` for learner
optimization on a GPU. Workers still sample and step environments on CPU, and the
existing worker boundary transfers weights from the learner. The decoder has no
separate GPU algorithm. Benchmark decisions per second before reserving substantial
GPU time; sequential token generation may make CPU inference the limiting step.

First compare trading disabled, one-for-one trading, and GRU trading on held-out
board/dice seeds with seat rotation. Report trade counts, execution rates, decisions
per game, and win rates. Checkpoint tournament CSVs include `trade_offers` and
`multi_card_offers` per seat; recorded-game trade metrics still supply execution
and feasibility rates. The current heuristic generates only one-for-one offers;
multi-card demonstrations and behavioural-cloning warm starts are follow-up work,
not prerequisites for using the decoder with PPO.

## Verification

`tests/models/test_trade_decoder.py` covers codec uniqueness, legal sampled offers,
counteroffer perspective, finite scores/gradients, and a PPO update using saved trade
tokens. Existing engine tests cover offer limits, hand checks, conservation, replay,
and negotiation phases. Training smoke runs exercise checkpoint writing and evaluation.
