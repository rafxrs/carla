# Synchronous MLP PPO baseline

This is the first trainable control condition. `carla train` collects learner decisions from one `CatanEnv` and updates a feed-forward MLP with clipped PPO. Each game has one learner and three fixed bots of the same configured kind (`random` or `heuristic`). The learner seat is random each episode by default. The action distribution uses the environment's legal-action masks; the current policy can propose one-for-one domestic trades. Opponents' hidden cards are not policy inputs.

## Run

```bash
carla train configs/smoke.yaml
carla train configs/experiments/ppo_mlp_baseline.yaml seed=0
carla evaluate runs/<run>/latest.pt --opponent random --boards 50 --seed 10000
carla evaluate runs/<run>/latest.pt --opponent heuristic --boards 50 --seed 10000
```

The baseline config sets `runtime.num_workers=1`, `runtime.envs_per_worker=1`, and one fixed opponent kind. The self-play, snapshot, and checkpoint opponent specifications in other experimental configs are reserved for the later opponent-pool implementation; this trainer rejects them explicitly. GNN, recurrent belief models, set attention, and variable-length trade decoding are likewise outside this baseline.

Each update collects up to `ppo.rollout_steps` decisions with one policy version. GAE is computed over episode boundaries, advantages are normalised across the update, and `ppo.epochs` passes use shuffled minibatches. PPO uses the stored action log probabilities, current legal masks, clipped policy loss, value MSE, entropy regularisation, gradient clipping, and optional KL early stop. The final update can be shorter than the rollout length so `total_env_steps` is exact. Logs report updates, decisions, episode ends in the latest rollout, policy/value losses, entropy, and approximate KL.

`latest.pt` is written atomically at `runtime.checkpoint_every` updates and at the end of a run. It contains model and Adam states, Python/NumPy/PyTorch RNG states, configuration, counters, and the live environment and observation. Resume at an update boundary with:

```bash
carla train configs/experiments/ppo_mlp_baseline.yaml --resume runs/<run>/latest.pt ppo.total_env_steps=200000
```

You may extend `ppo.total_env_steps` or change runtime logging/checkpoint settings on resume. Engine, environment, model, opponent, and other PPO settings must match the checkpoint. Use checkpoints only from trusted sources because loading a PyTorch checkpoint executes Python pickle data. `latest.pt` stores the current game, including hidden cards, for training continuation; do not expose it as public observation data.

`carla evaluate` takes a separate seed, plays each board with the learner in all four seats against three fixed bots, and chooses the highest-probability legal action. It prints wins, win rate, a game-level Wilson 95% interval, and mean victory points. Its interval treats games as independent; seat rotations share a board, so use multiple held-out boards and training seeds for final experimental conclusions. Evaluation never updates weights.
