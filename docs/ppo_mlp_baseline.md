# Synchronous MLP PPO baseline

For CSV metrics, periodic validation, retained checkpoints, multi-seed gamma/lambda/rollout sweeps,
resume recovery, and reproducible plots, see [experiments.md](experiments.md).

This MLP policy is the project's first trainable baseline. `carla train` collects decisions from `runtime.num_workers * runtime.envs_per_worker` independent `CatanEnv` instances and updates the policy with clipped PPO. Each CPU worker batches one forward pass across its environments at every rollout step. The learner waits for all workers before updating, so every batch uses the same policy version. With one worker, collection runs in the learner process.

Each game has one learner and three fixed bots of the same configured kind (`random` or `heuristic`). The learner seat is random each episode by default. The policy uses legal-action masks and can propose one-for-one domestic trades. Opponents' hidden cards are not policy inputs.

## Run

```bash
carla train configs/smoke.yaml
carla train configs/experiments/ppo_mlp_baseline.yaml seed=0
carla evaluate runs/<run>/latest.pt --opponent random --boards 50 --seed 10000
carla evaluate runs/<run>/latest.pt --opponent heuristic --boards 50 --seed 10000
```

The baseline config starts with `runtime.num_workers=1`, `runtime.envs_per_worker=4`, and one fixed opponent kind. Increase workers after measuring throughput; each worker needs a CPU core and sends its batches to the learner. The self-play, snapshot, and checkpoint opponent specifications in other experimental configs are reserved for the later opponent-pool implementation; this trainer rejects them explicitly. GNN, recurrent belief models, set attention, and variable-length trade decoding are likewise outside this baseline.

Each update collects up to `ppo.rollout_steps` decisions **per environment** with one policy version and returns one `RolloutBatch` per environment. GAE is computed independently across each environment's episode boundaries, advantages are normalised across all collected decisions, and `ppo.epochs` passes use shuffled minibatches. PPO uses the stored action log probabilities, current legal masks, clipped policy loss, value MSE, entropy regularisation, gradient clipping, and optional KL early stop. The final update can use fewer steps or environments so `total_env_steps` is exact. Logs report updates, total learner decisions, episode ends in the latest rollouts, policy/value losses, entropy, and approximate KL.

`latest.pt` is written atomically every 100 updates in the baseline config and at the end of a run. With four environments and 256 rollout steps, that is roughly every 102,400 decisions; change `runtime.checkpoint_every` after measuring checkpoint cost. It contains model and Adam states, Python/NumPy/PyTorch RNG states, configuration, counters, and every live environment and observation. Resume at an update boundary with:

```bash
carla train configs/experiments/ppo_mlp_baseline.yaml --resume runs/<run>/latest.pt ppo.total_env_steps=200000
```

You may extend `ppo.total_env_steps` or change runtime logging/checkpoint settings on resume. Keep `runtime.num_workers` and `runtime.envs_per_worker` unchanged; engine, environment, model, opponent, and other PPO settings must also match the checkpoint. Multiworker checkpoints include each worker's games, observations, episode counters, and random states. Use checkpoints only from trusted sources because loading a PyTorch checkpoint executes Python pickle data. `latest.pt` stores the current games, including hidden cards, for training continuation; do not expose it as public observation data.

`carla evaluate` takes a separate seed, plays each board with the learner in all four seats against three fixed bots, and chooses the highest-probability legal action. It prints wins, win rate, a game-level Wilson 95% interval, and mean victory points. Its interval treats games as independent; seat rotations share a board, so use multiple held-out boards and training seeds for final experimental conclusions. Evaluation never updates weights.
