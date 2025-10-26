# CFR Training Usage Guide

This guide covers how to launch CFR experiments, log experiment results to Weights and Biases, manage checkpoints, and serve trained models.

## Quick Start

Run a basic experiment with: `python -m models.cfr.run --num-games 10 --game-config-type-str tiny`

## Launching CFR Experiments

### Basic Experiment
```bash
python -m models.cfr.run \
  --num-games 100 \
  --sims-per-action 50 \
  --game-config-type-str small \
  --experiment-name "my-experiment"
```

If `--experiment-name` is not provided, a timestamped name will be generated.

### Key Parameters
- `--num-games`: Number of training games
- `--sims-per-action`: CFR simulations per action
- `--max-turns-per-game`: Game length limit
- `--test-games-interval-fast`: Evaluation frequency
- `--game-config-type-str`: Game size - `tiny`, `small`, `medium`
- `--random-seed`: Reproducibility seed

## Parallelism Modes

The system supports three parallelism strategies for CFR training, controlled by the `--parallelism-mode` parameter:

### Sequential
- **Description**: Runs one self-play game at a time
- **Flag**: `--parallelism-mode none`
- **Updates**: Applied synchronously after each game
- **Characteristics**: Fully deterministic execution, strict game-order consistency, no parallel speedup
- **Use case**: Debugging and development

### Parallel Unordered Update
- **Description**: Workers launch games concurrently, updates applied immediately upon completion
- **Flag**: `--parallelism-mode parallel-unordered-update`
- **Updates**: Applied asynchronously without regard to game index
- **Characteristics**: Fastest execution, not fully deterministic
- **Use case**: When speed is prioritized over reproducibility

### Parallel Batch Ordered Update (Default)
- **Description**: Games run in batches, updates applied synchronously in game order
- **Flag**: `--parallelism-mode parallel-batch-ordered-update`
- **Updates**: Applied after entire batch completes, maintaining game index order
- **Characteristics**: Fully deterministic execution, slower than unordered update
- **Use case**: Balance between speed and reproducibility

### Usage Examples
```bash
# Use fastest mode (non-deterministic)
python -m models.cfr.run --parallelism-mode parallel-unordered-update

# Use deterministic mode (default, deterministic)
python -m models.cfr.run --parallelism-mode parallel-batch-ordered-update

# Use sequential mode (slowest, deterministic)
python -m models.cfr.run --parallelism-mode none
```

## Logging to Weights and Biases

### New Run
```bash
python -m models.cfr.run \
  --log-to-wandb \
  --experiment-name "my-experiment"
```
The `--log-to-wandb` flag is required to log experiment results to Weights and Biases.

### Resume Existing Run
```bash
python -m models.cfr.run \
  --log-to-wandb \
  --resume-from-wandb-run-id "existing-run-id" \
  --experiment-name "my-experiment"
```
The `--resume-from-wandb-run-id` flag is required to resume an existing run from Weights and Biases. If the run ID doesn't exist, the system will raise an error.

## Resuming Experiments

### From Local Checkpoints
```bash
python -m models.cfr.run \
  --attempt-load-checkpoint \
  --experiment-name "my-experiment"
```
The system automatically finds the latest checkpoint by game index, searching in the local checkpoint directory. For instance, if checkpoint files `game_idx_50.json` and `game_idx_100.json` exist, the system will resume from `game_idx_100.json`.

### From Remote Checkpoints
```bash
python -m models.cfr.run \
  --attempt-load-checkpoint \
  --save-checkpoint-remote \
  --experiment-name "my-experiment"
```
The system automatically finds the latest checkpoint by game index, searching in the remote bucket.

## Model Serving

To serve a trained checkpoint via the web interface:

### Local Checkpoints
Update `app/models.yaml`:
```yaml
models:
  my-model:
    name: "My Trained Model"
    description: "CFR model trained on small game config"
    checkpoint_path: "checkpoints/experiment-20250101-120000/game_idx_100.json"
```

### Remote Checkpoints (GCS)
```yaml
models:
  my-model:
    name: "My Trained Model"
    description: "CFR model trained on small game config"
    checkpoint_path: "gs://my-bucket/checkpoints/experiment-20250101-120000/game_idx_100.json"
```

After updating `models.yaml`, restart the web server: `just dev` or `just docker-dev`

## State Abstraction and Abstract Actions

The system uses state abstraction to reduce the complexity of the game by mapping concrete actions to abstract actions. In Monopoly Deal, players can take various concrete actions like moving specific cards between piles (hand to property, hand to cash, etc.). The default **intent-based state abstraction** maps these concrete actions into abstract action categories, creating a many-to-one mapping that reduces the game's resolution while preserving strategic structure.

**Abstract Actions** represent higher-level strategic intents rather than specific card movements. For example, given an empty property pile, different-colored property cards in the player's hand might all map to the same abstract action "START_NEW_PROPERTY_SET".

## Information Set (InfoSet) Key Generation

InfoSet keys uniquely identify decision points in the game using the format `{player_idx}@{abstraction_cls_name}@{abstraction_key}`. The abstraction_key is an MD5 hash of the JSON-serialized abstracted state, which includes the state abstraction mechanism and player ID (0, 1). This ensures each unique information set has a consistent identifier across training runs.

## Inspecting Checkpoints

Checkpoints contain the complete CFR state at a specific game index.

**Game Configuration**: Contains parameters like `required_property_sets`, `initial_hand_size`, `max_consecutive_player_actions`, and `cash_card_values`. The `card_to_special_card_ratio` and `deck_size_multiplier` control deck composition (see code for details).

**Policy Manager**: Stores recent policy history with a configurable buffer size (e.g., 10). The `update_count` tracks how many times each InfoSet has been updated, while `player_buffers` contain the last N policy vectors for each InfoSet. Each policy vector is a probability distribution over abstract actions. The mapping from indices to AbstractAction enum values is available via the inspection tool:

```bash
python -m models.cfr.inspect actions
```

**Regret Manager**: Maps InfoSet keys to action regrets. For each action, it stores cumulative regret (`sum`) and visit count (`n`), indexed by abstract action index.

**CFR Reach Probability Counter**: Tracks how often each InfoSet is reached during training, storing probability estimates used for computing counterfactual values in CFR.

## Action Resolvers

Action resolvers determine which specific concrete action to take when multiple actions map to the same abstract action. The default **GreedyActionResolver** uses simple heuristics: for rent collection, it chooses the card with the highest rent value; for cash actions, it selects the card with the highest cash value; otherwise, it picks randomly. As needed, you can implement a custom resolver by extending `BaseActionResolver` and implementing the `resolve()` method that takes available actions and game state, returning a chosen `WrappedAction`.

## Action Selectors

The system includes several action selectors for evaluation and testing:

- **RandomSelector**: Randomly samples from available actions
- **RiskAwareSelector**: Uses configurable aggressiveness (0-1) to balance property vs. cash card preferences, with temperature controlling sensitivity
- **CFRActionSelector**: Uses the trained CFR policy to select actions (argmax from learned policy)

To implement a custom selector, extend `BaseActionSelector` and implement the `select()` method that takes available actions and game state, returning a chosen `WrappedAction`.

## CFR Core Components

**Policy Manager**: Maintains a rolling buffer of recent policies for each InfoSet, computing average policies for stable decision-making. The buffer size controls how much history is retained.

**Regret Manager**: Tracks cumulative regrets for each action in each InfoSet, storing both sum and visit count for computing regret-based policy updates.

**Reach Probability Counter**: Estimates how frequently each InfoSet is encountered during training, used for computing weighted regret updates in CFR.

## Model Evaluation

During training, the system automatically evaluates the CFR model against different opponents at configurable intervals. The `--test-games-interval-fast` parameter controls evaluation frequency, while `--num-test-games` sets the number of evaluation games per opponent. Models are tested against RandomSelector and RiskAwareSelector with different aggressiveness levels to measure learning progress.
