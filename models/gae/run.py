import datetime
import inspect
import os
from typing import Optional

import numpy as np
import typer

from app.settings import DEFAULT_WANDB_PROJECT_NAME
from game.config import GameConfigType
from game.constants import NUM_PLAYERS
from game.game import Game, PlayerSpec
from game.state import ABSTRACTION_NAME_TO_CLS, RESOLVER_NAME_TO_CLS
from game.util import set_random_seed
from models.cfr.cfr import CFR
from models.cfr.selector import CFRActionSelector
from models.checkpoint import CheckpointManager, CheckpointPath, load_checkpoint_data
from models.constants import MODEL_NAME_TO_CLS, MODEL_NAME_TO_SELECTOR
from models.gae.model import PolicyAndValueNetwork
from models.gae.selector import GAEActionSelector
from models.types import Trajectory
from models.util import play_games, produce_trajectory
import wandb


app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def run_experiment(
    cfr_checkpoint_path: str = "checkpoints/app/cfr-medium.json",
    model_update_interval: int = 1000,
    game_config_type_str: str = "medium",
    abstraction_cls: str = "IntentStateAbstraction",
    resolver_cls: str = "GreedyActionResolver",
    max_turns_per_game: int = 5,
    random_seed: int = 0,
    num_games: int = 10000,
    # Checkpointing parameters
    save_checkpoint_remote: bool = False,
    skip_checkpointing: bool = False,
    attempt_load_checkpoint: bool = False,
    experiment_name: Optional[str] = None,
    # Logging parameters
    log_to_wandb: bool = False,
    num_eval_games: int = 20,
    baseline_checkpoint: list[str] = [],
    stateless_baseline_model: list[str] = [],
    train_against_snapshot_threshold: float = 0.65,
    # Model parameters
    hidden_layer_sizes: list[int] = [],
    learning_rate: float = 0.0015,
    gamma: float = 0.99,
    lmbda: float = 0.95,
    value_loss_weight: float = 0.5,
    # Training hyperparameters
    entropy_coef: float = 0.02,  # Entropy regularization coefficient (starts here, decays over time)
    clip_epsilon: float = 0.2,  # PPO clipping epsilon
    weight_decay: float = 1e-5,  # L2 regularization weight decay
    gradient_clip: float = 1.0,  # Gradient clipping norm
    entropy_decay_games: int = 10000,  # Games over which entropy decays
    entropy_decay_min: float = 0.003,  # Minimum entropy coefficient after decay
    lr_decay_games: int = 20000,  # Games over which learning rate decays
    lr_decay_min: float = 0.2,  # Minimum learning rate factor after decay
) -> None:
    """Run a reinforcement learning experiment with checkpointing.

    Args:
        cfr_checkpoint_path: Path to CFR checkpoint for opponent model.
        model_update_interval: Number of trajectories to collect before updating model.
        game_config_type_str: Game configuration type ("tiny", "small", "medium").
        abstraction_cls: State abstraction class name.
        resolver_cls: Action resolver class name.
        max_turns_per_game: Maximum turns allowed per game.
        random_seed: Seed for random number generation.
        num_games: Total number of games to play.
        save_checkpoint_remote: Whether to save checkpoints to remote storage.
        skip_checkpointing: Whether to skip checkpointing.
        attempt_load_checkpoint: Whether to attempt loading from existing checkpoints.
        experiment_name: Name for the experiment (auto-generated if None).
        log_to_wandb: Whether to log to Weights & Biases.
        num_eval_games: Number of evaluation games to play against CFR at each interval.
        baseline_checkpoint: List of checkpoint paths for baseline models.
        stateless_baseline_model: List of stateless baseline model names.
        train_against_snapshot_threshold: Threshold for training against snapshot.
        hidden_layer_sizes: List of hidden layer sizes.
        learning_rate: Learning rate for the reinforcement learning model.
        gamma: Discount factor for the reinforcement learning model.
        lmbda: Lambda for the reinforcement learning model.
        value_loss_weight: Weight for the value loss in the reinforcement learning model.
        entropy_coef: Entropy regularization coefficient for the reinforcement learning model.
        clip_epsilon: PPO clipping epsilon for the reinforcement learning model.
        weight_decay: Weight decay for the reinforcement learning model.
        gradient_clip: Gradient clip for the reinforcement learning model.
        entropy_decay_games: Games over which entropy decays.
        entropy_decay_min: Minimum entropy coefficient after decay.
        lr_decay_games: Games over which learning rate decays.
        lr_decay_min: Minimum learning rate factor after decay.
    """
    print(f"Using random seed: {random_seed}")

    # Set PYTHONHASHSEED for deterministic dictionary iteration order
    if "PYTHONHASHSEED" not in os.environ:
        os.environ["PYTHONHASHSEED"] = str(random_seed)
        print(f"Set PYTHONHASHSEED={random_seed}")

    set_random_seed(random_seed)

    # Fill default experiment name if not provided
    if experiment_name is None:
        experiment_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"Experiment name: {experiment_name}")

    # Define game config
    game_config = GameConfigType[game_config_type_str.upper()].value

    # Set up checkpointing
    checkpoint_manager = CheckpointManager(experiment_name, remote=save_checkpoint_remote)
    checkpoint_path: Optional[CheckpointPath] = None
    if not checkpoint_manager.experiment_dir_empty and not attempt_load_checkpoint:
        raise ValueError(
            f"Experiment directory '{checkpoint_manager.experiment_dir}' is not empty. "
            f"Use --attempt-load-checkpoint to resume from existing checkpoints, "
            f"or choose a different experiment name."
        )

    if attempt_load_checkpoint:
        # Find latest checkpoint
        checkpoint_path = checkpoint_manager.find_latest_checkpoint_by_game_idx()
        if not checkpoint_path:
            print(f"No checkpoint found for experiment {experiment_name} in {checkpoint_manager.experiment_dir}")

    # Load CFR checkpoint for opponent (needed to determine target_player_index)
    cfr = CFR.from_checkpoint(cfr_checkpoint_path)

    # Validate game config matches
    if cfr.game_config.key != game_config.key:
        raise ValueError(f"Game config does not match: {cfr.game_config.key} != {game_config.key}")

    # Define opponent player index
    opponent_player_index = cfr.target_player_index

    # Define target player index
    target_player_index = 1 - opponent_player_index

    # Convert abstraction and resolver class names to classes
    abstraction_cls_type = ABSTRACTION_NAME_TO_CLS[abstraction_cls]
    resolver_cls_type = RESOLVER_NAME_TO_CLS[resolver_cls]

    # Define config
    config = {k: v for k, v in locals().items() if k in inspect.signature(run_experiment).parameters}

    # Load or create model
    if checkpoint_path:
        print(f"Loading model from checkpoint: {checkpoint_path}")
        model = PolicyAndValueNetwork.from_checkpoint(str(checkpoint_path))
        # Extract game idx from checkpoint path and set to next game
        checkpoint_game_idx = checkpoint_path.game_idx
        game_idx = checkpoint_game_idx + 1
        print(f"Resuming from game {game_idx} (checkpoint was at game {checkpoint_game_idx})...")
        # Validate loaded model matches expected config
        if model.target_player_index != target_player_index:
            raise ValueError(
                f"Checkpoint target_player_index {model.target_player_index} does not match expected {target_player_index}"
            )
        if model.game_config.key != game_config.key:
            raise ValueError(
                f"Checkpoint game_config {model.game_config.key} does not match expected {game_config.key}"
            )
    else:
        print("Instantiating model from scratch...")
        model = PolicyAndValueNetwork(
            learning_rate=learning_rate,
            gamma=gamma,
            lmbda=lmbda,
            value_loss_weight=value_loss_weight,
            target_player_index=target_player_index,
            game_config=game_config,
            abstraction_cls=abstraction_cls_type,
            resolver_cls=resolver_cls_type,
            hidden_layer_sizes=hidden_layer_sizes,
            random_seed=random_seed,
            entropy_coef=entropy_coef,
            clip_epsilon=clip_epsilon,
            weight_decay=weight_decay,
            gradient_clip=gradient_clip,
        )
        game_idx = 0

    # Create selectors (model is outer level, passed to selector)
    player_selector = GAEActionSelector(model=model)

    # Define opponent selector
    opponent_selector = CFRActionSelector(
        policy_manager=cfr.policy_manager.get_runtime_policy_manager(opponent_player_index)
    )

    # Map player indices to policies based on target_player_index
    player2selector = {
        target_player_index: player_selector,
        opponent_player_index: opponent_selector,
    }

    # Define player specs
    player_specs = {
        opponent_player_index: PlayerSpec(abstraction_cls=cfr.abstraction_cls, resolver_cls=cfr.resolver_cls),
        target_player_index: PlayerSpec(abstraction_cls=model.abstraction_cls, resolver_cls=model.resolver_cls),
    }

    # Define list of trajectories
    trajectories: list[Trajectory] = []

    # Define list of terminal rewards
    terminal_rewards: list[float] = []

    # Self-play snapshot - created/updated when performance > train_against_snapshot_threshold
    snapshot_model: PolicyAndValueNetwork | None = None

    # Load baseline models
    baseline_models = {}
    for baseline_checkpoint_path in baseline_checkpoint:
        # Load checkpoint data
        data = load_checkpoint_data(baseline_checkpoint_path)
        # Extract model_cls from data
        model_cls = data["model_cls"]
        # Load model and add to baseline models
        baseline_models[model_cls] = MODEL_NAME_TO_CLS[model_cls].from_json(data)

    for baseline_model in stateless_baseline_model:
        baseline_models[baseline_model] = MODEL_NAME_TO_CLS[baseline_model]()

    # Initialize W&B if requested
    if log_to_wandb:
        # Initialize Weights and Biases
        wandb.init(project=DEFAULT_WANDB_PROJECT_NAME, group=experiment_name, config=config)
        # Define custom x-axis metric (so we can log metrics out of order)
        wandb.define_metric("step_")
        # Define metrics, using custom x-axis metric
        for metric in [
            "win_rates/*",
            "mean_terminal_reward",
            "opponent_type",
            "total_loss",
            "policy_loss",
            "value_loss",
            "entropy_loss",
            "mean_advantage",
            "mean_value",
        ]:
            wandb.define_metric(metric, step_metric="step_")

    # Collect trajectories and update model
    last_update_diagnostics: dict[str, float] | None = None
    while game_idx < num_games:
        # Switch to snapshot if available (self-play)
        if snapshot_model is not None:
            player2selector[opponent_player_index] = GAEActionSelector(model=snapshot_model)
            spec = PlayerSpec(abstraction_cls=model.abstraction_cls, resolver_cls=model.resolver_cls)
            player_specs = {opponent_player_index: spec, target_player_index: spec}

        while len(trajectories) < model_update_interval and game_idx < num_games:
            # Instantiate game
            game = Game(
                config=game_config,
                player_specs=player_specs,
                init_player_index=game_idx % NUM_PLAYERS,
                random_seed=random_seed + game_idx,
            )

            # Produce trajectory
            trajectory = produce_trajectory(
                game=game,
                player2selector=player2selector,
                target_player_index=target_player_index,
                max_turns_per_game=max_turns_per_game,
                argmax=False,
            )

            # Add trajectory to list
            trajectories.append(trajectory)

            # Add terminal reward to list
            terminal_rewards.append(trajectory.reward)

            # Increment game index
            game_idx += 1

        # Compose training batch
        if trajectories and game_idx % model_update_interval == 0:
            print(f"Updating model with {len(trajectories)} trajectories")
            last_update_diagnostics = model.update(
                trajectories,
                game_idx=game_idx,
                entropy_decay_games=entropy_decay_games,
                entropy_decay_min=entropy_decay_min,
                lr_decay_games=lr_decay_games,
                lr_decay_min=lr_decay_min,
            )
            # Clear trajectories
            trajectories.clear()

        # Evaluate performance and update snapshot
        if terminal_rewards and game_idx % model_update_interval == 0:
            mean_terminal_reward = np.mean(terminal_rewards)
            opponent_type = "snapshot" if snapshot_model is not None else "CFR"
            print(f"Game {game_idx} | Mean terminal reward: {mean_terminal_reward:.3f} | Opponent: {opponent_type}")

            # Create or update snapshot for self-play when performance > train_against_snapshot_threshold
            if mean_terminal_reward > train_against_snapshot_threshold:
                if snapshot_model is None:
                    print(
                        f"Creating snapshot model for self-play at game {game_idx} (win rate: {mean_terminal_reward:.3f})"
                    )
                else:
                    print(f"Updating snapshot model at game {game_idx} (win rate: {mean_terminal_reward:.3f})")
                snapshot_model = model.copy()

            # Log to W&B
            win_rates = {}
            log_dict = {
                "mean_terminal_reward": mean_terminal_reward,
                "opponent_type": opponent_type,
                "step_": game_idx,
            }

            # Add training diagnostics if available
            if last_update_diagnostics is not None:
                log_dict.update(last_update_diagnostics)

            # Evaluate against baseline models
            if log_to_wandb and game_idx > 0 and game_idx % model_update_interval == 0:
                print(f"Game {game_idx} | Evaluating against baseline models...", flush=True)
                for bname, bmodel in baseline_models.items():
                    # Define evaluation player specs
                    eval_player_specs = {
                        opponent_player_index: PlayerSpec(
                            abstraction_cls=bmodel.abstraction_cls, resolver_cls=bmodel.resolver_cls
                        ),
                        target_player_index: PlayerSpec(
                            abstraction_cls=model.abstraction_cls, resolver_cls=model.resolver_cls
                        ),
                    }

                    # Define evaluation player selectors
                    eval_player2selector = {
                        target_player_index: player_selector,
                        opponent_player_index: MODEL_NAME_TO_SELECTOR[bname].from_model(
                            model=bmodel, opponent_player_index=opponent_player_index
                        ),
                    }

                    # Play evaluation games
                    eval_trajectories = play_games(
                        game_config=game_config,
                        player_specs=eval_player_specs,
                        player2selector=eval_player2selector,
                        target_player_index=target_player_index,
                        num_games=num_eval_games,
                        max_turns_per_game=max_turns_per_game,
                        random_seed_base=random_seed + game_idx,
                    )

                    # Collect eval rewards
                    eval_rewards = [trajectory.reward for trajectory in eval_trajectories]

                    win_rate_vs_baseline_model = np.mean(eval_rewards).item()
                    win_rates[f"win_rates/{bname}"] = float(win_rate_vs_baseline_model)
                    print(
                        f"Game {game_idx} | Win rate vs {bname}: {win_rate_vs_baseline_model:.3f}",
                        flush=True,
                    )

            # Evaluate against snapshot if training against snapshot
            if snapshot_model is not None:
                win_rates["win_rates/snapshot"] = float(
                    mean_terminal_reward
                )  # Current training performance is vs snapshot

            if log_to_wandb:
                if win_rates:
                    log_dict.update(win_rates)
                wandb.log(log_dict, step=game_idx)

            terminal_rewards.clear()

        # Save checkpoint at intervals
        if game_idx > 0 and game_idx % model_update_interval == 0 and not skip_checkpointing:
            print(f"Game {game_idx} | Saving model state...", flush=True)
            checkpoint_manager.save_model_state(game_idx=game_idx, model=model)

    # Save final checkpoint if we haven't already saved at this game_idx
    if game_idx > 0 and game_idx % model_update_interval != 0 and not skip_checkpointing:
        print(f"Game {game_idx} | Saving final model state...", flush=True)
        checkpoint_manager.save_model_state(game_idx=game_idx, model=model)

    if log_to_wandb:
        wandb.finish()


if __name__ == "__main__":
    try:
        app()
    except Exception:
        import sys
        import traceback

        print("🔥 Uncaught exception in top-level script:")
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(1)
