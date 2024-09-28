import datetime
import inspect
import json
import multiprocessing as mp
import os
import platform
import sys
from typing import Optional

from google.cloud import storage
import numpy as np
import psutil
import ray
from tqdm import tqdm
import typer

from app.settings import CHECKPOINTS_DIRNAME, DEFAULT_WANDB_PROJECT_NAME, GCP_POLICY_BUCKET_NAME
from game.action import AbstractAction
from game.config import GameConfigType
from game.state import ABSTRACTION_NAME_TO_CLS, RESOLVER_NAME_TO_CLS
from game.util import set_random_seed
from models.cfr.cfr import CFR, PolicyManager, get_median_abstract_action_probs, null_progress_bar
from models.cfr.selector import BaseActionSelector, CFRActionSelector, RandomSelector, RiskAwareSelector
from models.cfr.util import CheckpointFinder, CheckpointPath
import wandb


app = typer.Typer(pretty_exceptions_enable=False)


def save_cfr_state(cfr: CFR, experiment_name: str, remote: bool) -> None:
    """Save CFR state to a JSON file."""
    game_idx = cfr.game_idx
    checkpoint_path = CheckpointPath.create(experiment_name, game_idx, remote)

    if remote:
        client = storage.Client()
        bucket = client.bucket(GCP_POLICY_BUCKET_NAME)
        blob = bucket.blob(checkpoint_path.blob_name)
        blob.upload_from_string(json.dumps(cfr.to_json(), indent=4, default=str))
        print(f"✅ CFR state saved to GCS: {checkpoint_path.full_path}", flush=True)
    else:
        os.makedirs(f"{CHECKPOINTS_DIRNAME}/{experiment_name}", exist_ok=True)
        with open(checkpoint_path.full_path, "w") as f:
            json.dump(cfr.to_json(), f, indent=4)
        print(f"✅ CFR state saved to {checkpoint_path.full_path}", flush=True)


@ray.remote(num_cpus=1, resources={"eval_cpu": 1})
def evaluate_win_rate(
    i: int,
    name: str,
    cfr: CFR,
    player_selector: BaseActionSelector,
    opponent_selector: BaseActionSelector,
    num_games: int,
    max_turns_per_game: int,
    player_always_first: bool,
    target_player_index: Optional[int],
    random_seed: int,
) -> float:
    # Set random seed in the Ray worker process
    set_random_seed(random_seed)
    print(
        f"Game {i} | Eval {num_games} games vs {name} | First? {player_always_first}",
        flush=True,
    )
    return play_games(
        cfr=cfr,
        player_selector=player_selector,
        opponent_selector=opponent_selector,
        num_games=num_games,
        max_turns_per_game=max_turns_per_game,
        player_always_first=player_always_first,
        target_player_index=target_player_index or 0,
        random_seed=random_seed,
    )


def play_games(
    cfr: CFR,
    player_selector: BaseActionSelector,
    opponent_selector: BaseActionSelector,
    num_games: int,
    max_turns_per_game: int,
    target_player_index: int,
    player_always_first: bool = False,
    random_seed: int = 0,
) -> float:
    # Initialize results buffer
    results = []
    # Play games
    for i in range(num_games):
        # Instantiate game
        game = cfr.instantiate_new_game(
            init_player_index=target_player_index if player_always_first else i % 2,
            random_seed=random_seed + i,
        )
        # Map players to policies based on target_player_index
        player2selector = {
            game.players[target_player_index]: player_selector,
            game.players[1 - target_player_index]: opponent_selector,
        }

        while not game.over and game.turn_state.turn_idx < max_turns_per_game:
            # Get action selector
            selector = player2selector[game.player]
            # Get player actions
            actions = game.state.get_player_actions()
            # Select action
            wrapped_action = selector.select(
                actions=actions,
                game=game,
            )
            # Take game step
            game.step(selected_action=wrapped_action.action)

        # Compute result for the target player: 1 if target player wins, 0 if opponent wins, 0.5 if tie
        utility = 1 if game.winner and game.winner.index == target_player_index else 0 if game.winner else 0.5

        # Store results
        results.append(utility)

        # Reset policies
        for selector in player2selector.values():
            selector.reset()

    # Return the player's win rate
    win_rate = np.mean(results).item()
    return win_rate


@app.command()
def run_experiment(
    # Game parameters
    game_config_type_str: str = "tiny",
    # CFR parameters
    num_games: int = 2,
    target_player_index: int = 0,
    sims_per_action: int = 50,
    max_turns_per_game: int = 5,
    buffer_size: int = 10,
    abstraction_cls: str = "IntentStateAbstraction",
    resolver_cls: str = "GreedyActionResolver",
    uniform_external_sampling: bool = True,
    epsilon: float = 0.1,
    # Test-time parameters
    num_test_games: int = 2,
    test_games_interval_fast: int = -1,
    # Logging parameters
    remote: bool = False,
    experiment_name: Optional[str] = None,
    resume_from_run_id: Optional[str] = None,
    verbose: bool = False,
    experiment_description: str = "experiment run",
    suppress_progress_bar: bool = True,
    # CFR execution parameters
    attempt_load_checkpoint: bool = False,
    random_seed: int = 0,
    first_cpu_core_only: bool = False,
    no_parallelism: bool = False,
    # Code provenance parameters
    git_commit: Optional[str] = None,
) -> None:
    print(f"Using random seed: {random_seed}")
    set_random_seed(random_seed)

    # Fill default experiment name if not provided
    if experiment_name is None:
        experiment_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    if first_cpu_core_only:
        # Pin to 0th CPU core
        if platform.system() == "Linux":
            try:
                psutil.Process().cpu_affinity([0])  # type: ignore
            except Exception as e:
                print(f"⚠️ Failed to set CPU affinity: {e}", flush=True)

    if remote:
        # Initialize Weights and Biases
        config = {k: v for k, v in locals().items() if k in inspect.signature(run_experiment).parameters}

        # If run_id provided, resume from previous run
        if resume_from_run_id is not None:
            wandb.init(
                project=DEFAULT_WANDB_PROJECT_NAME,
                id=resume_from_run_id,
                config=config,
                resume="must",
            )
        else:
            wandb.init(project=DEFAULT_WANDB_PROJECT_NAME, group=experiment_name, config=config)
        # Define custom x-axis metric (so we can log metrics out of order)
        wandb.define_metric("step_")
        # Define metrics, using custom x-axis metric
        for metric in [
            "max_expected_regret",
            "game_length",
            "num_infosets",
            "median_num_updates",
            "regrets/*",
            "player_win_rates/*",
        ]:
            wandb.define_metric(metric, step_metric="step_")

    # Define game config
    game_config = GameConfigType[game_config_type_str.upper()].value

    # Instantiate CFR
    checkpoint_path = None
    if attempt_load_checkpoint:
        # Find latest checkpoint
        finder = CheckpointFinder(experiment_name, remote)
        checkpoint_path = finder.find_latest()
        if not checkpoint_path:
            print(f"No checkpoint found for experiment name: {experiment_name}")

    if checkpoint_path:
        print(f"Loading CFR from checkpoint: {checkpoint_path.full_path}")
        cfr = CFR.from_checkpoint(checkpoint_path.full_path)
        # Check constants hash
        if cfr.game_config.key != game_config.key:
            raise ValueError(f"Game config ID does not match: {cfr.game_config.key} != {game_config.key}")
        # Extract game idx from checkpoint path and set to next game
        checkpoint_game_idx = checkpoint_path.game_idx
        game_idx = checkpoint_game_idx + 1
        print(f"Resuming from game {game_idx} (checkpoint was at game {checkpoint_game_idx})")
    else:
        print("Instantiating CFR from scratch")
        cfr = CFR(
            game_config=game_config,
            epsilon=epsilon,
            abstraction_cls=ABSTRACTION_NAME_TO_CLS[abstraction_cls],
            resolver_cls=RESOLVER_NAME_TO_CLS[resolver_cls],
            target_player_index=target_player_index,
            sims_per_action=sims_per_action,
            max_turns_per_game=max_turns_per_game,
            uniform_external_sampling=uniform_external_sampling,
            policy_manager=PolicyManager(buffer_size=buffer_size),
            verbose=verbose,
        )
        game_idx = 0

    # Play game and update policies
    pbar_context = null_progress_bar() if suppress_progress_bar else tqdm(desc="Playing Game", dynamic_ncols=True)

    # Init ray
    cpu_count = mp.cpu_count()
    train_cpus = int(cpu_count * 0.9 if test_games_interval_fast > 0 else cpu_count)
    eval_cpus = cpu_count - train_cpus
    ray.init(resources={"train_cpu": train_cpus, "eval_cpu": eval_cpus})

    # Store CFR object in ray shared memory
    cfr_ref = ray.put(cfr)

    with pbar_context as pbar:

        @ray.remote(num_cpus=1, resources={"train_cpu": 1})
        def optimize_one_game(game_idx: int, cfr: CFR):
            # Define random seed
            seed = random_seed + game_idx
            # Set random seed in the Ray worker process
            set_random_seed(seed)
            i = game_idx
            print(f"Game {i} | Optimizing...", flush=True)
            # Instantiate game
            cfr.game = cfr.instantiate_new_game(random_seed=seed, init_player_index=i % 2)
            cfr.game_idx = i
            # Play game
            cfr_updates = []
            while not cfr.game.over and cfr.game.turn_state.turn_idx < max_turns_per_game:
                # Compute CFR update
                update = cfr.optimize()
                cfr_updates.append(update)

                # Take game step
                cfr.game.step(selected_action=update.wrapped_action.action)

                # Update progress bar
                pbar.update(1)
            # Compute training metrics
            max_expected_regret = cfr.compute_max_expected_regret(target_player_index)
            game_length = cfr.game.turn_state.turn_idx
            update_counts = [
                cfr.policy_manager.get_update_count(key)
                for key in cfr.policy_manager.get_player_buffer(target_player_index)
            ]
            num_infosets = len(update_counts)
            # Compute update count quantiles
            update_count_stats = {}
            for p in [0, 25, 50, 75, 100]:
                update_count_stats[f"update_counts/p{p}"] = np.percentile(update_counts, p)
            # Compute regret quantiles
            regret_stats = {}
            for aa in AbstractAction:
                regrets = cfr.regret_manager.get_means_for_action(aa, target_player_index)
                if regrets:
                    regret_stats[f"regrets/{aa.value}/mean"] = np.mean(regrets)
                    for p in [50]:
                        regret_stats[f"regrets/{aa.value}/p{p}"] = np.percentile(regrets, p)
            # Compute median policy
            median_policy = get_median_abstract_action_probs(cfr.policy_manager, target_player_index)
            median_policy_stats = {f"median_policy/{aa.value}/prob": p for aa, p in median_policy.items()}
            # Play game against various competitors
            win_rates_futures = {}
            matches = [
                (
                    "RandomSelector",
                    RandomSelector(),
                    lambda x: x % test_games_interval_fast == 0 and test_games_interval_fast > 0 and x > 0,
                    num_test_games,
                ),
                (
                    "RiskAwareSelector",
                    RiskAwareSelector(),
                    lambda x: x % test_games_interval_fast == 0 and test_games_interval_fast > 0 and x > 0,
                    num_test_games,
                ),
            ]
            # Play games against each opponent and record win rates
            for (
                name,
                opponent_selector,
                should_run,
                num_test_games_,
            ) in matches:
                if should_run(i):
                    for player_always_first in [True, False]:
                        slug = f"player_win_rates/{name}__player_always_first={player_always_first}"
                        # Call evaluate_win_rate remote function and store future
                        win_rates_futures[slug] = evaluate_win_rate.remote(
                            i,
                            name,
                            cfr,
                            CFRActionSelector(
                                policy_manager=cfr.policy_manager.get_runtime_policy_manager(target_player_index)
                            ),
                            opponent_selector,
                            num_test_games_,
                            max_turns_per_game,
                            player_always_first,
                            target_player_index,
                            seed,
                        )
            win_rates = {slug: ray.get(future) for slug, future in win_rates_futures.items()}

            if verbose:
                print(
                    f"Game {i} | Max Expected Regret: {max_expected_regret}",
                    flush=True,
                )
            if win_rates:
                print(f"Game {i} | Win Rates: {win_rates}", flush=True)
            # Save policies and CFR state
            if i > 0 and (i % test_games_interval_fast == 0 or i == num_games - 1):
                print(f"Game {i} | Saving CFR state...", flush=True)
                save_cfr_state(cfr, experiment_name, remote)
            # Return metrics
            return cfr_updates, {
                "max_expected_regret": max_expected_regret,
                **regret_stats,
                "game_length": game_length,
                "num_infosets": num_infosets,
                **update_count_stats,
                **median_policy_stats,
                **win_rates,
                "step_": i,  # Our custom x-axis metric
            }

        # Run optimization in parallel
        futures = []

        # Kick off initial batch
        initial_batch_size = min(1 if no_parallelism else num_games, train_cpus)
        while game_idx < num_games and len(futures) < initial_batch_size:
            futures.append((game_idx, optimize_one_game.remote(game_idx, cfr_ref)))
            game_idx += 1

        # Process results and enqueue new ones as space frees up
        while futures:
            # Wait for one future to finish
            done_ids, _ = ray.wait([f for _, f in futures], num_returns=1)
            (done_id,) = done_ids

            # Find and remove the completed future
            future = next(f for _, f in futures if f == done_id)
            futures = [(i, f) for i, f in futures if f != done_id]

            # Retrieve result
            cfr_updates, metrics = ray.get(future)

            # Log to wandb if enabled
            if remote:
                wandb.log(metrics)

            # Update CFR object
            cfr.apply_updates(cfr_updates)

            # Snapshot the CFR object
            cfr_ref = ray.put(cfr)

            # Launch next job if any remain
            if game_idx < num_games:
                futures.append((game_idx, optimize_one_game.remote(game_idx, cfr_ref)))
                game_idx += 1

    if remote:
        # Close Weights and Biases
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
