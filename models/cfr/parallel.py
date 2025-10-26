"""Parallelism strategies for CFR training."""

from enum import Enum
from typing import Callable

import ray

import wandb


class ParallelismMode(Enum):
    """Enumeration of available parallelism modes."""

    PARALLEL_UNORDERED_UPDATE = "parallel-unordered-update"
    PARALLEL_BATCH_ORDERED_UPDATE = "parallel-batch-ordered-update"
    NONE = "none"


def get_initial_batch_size(mode: ParallelismMode, train_cpus: int, num_games: int) -> int:
    """Get the initial batch size for launching workers."""
    if mode == ParallelismMode.NONE:
        return 1
    else:
        return min(num_games, train_cpus)


def should_launch_next_worker(
    mode: ParallelismMode, game_idx: int, num_games: int, active_futures: int, train_cpus: int
) -> bool:
    """Determine if we should launch the next worker."""
    if mode == ParallelismMode.NONE:
        return False  # Sequential processing, no new workers
    else:
        return game_idx < num_games and active_futures < train_cpus


def process_results(
    mode: ParallelismMode,
    cfr,
    cfr_ref,
    log_to_wandb: bool,
    num_games: int,
    game_idx: int,
    optimize_one_game_func: Callable,
    train_cpus: int,
):
    """Process results according to the selected strategy."""
    if mode == ParallelismMode.PARALLEL_UNORDERED_UPDATE:
        return _process_unordered(cfr, cfr_ref, log_to_wandb, num_games, game_idx, optimize_one_game_func, train_cpus)
    elif mode == ParallelismMode.PARALLEL_BATCH_ORDERED_UPDATE:
        return _process_batch_ordered(
            cfr, cfr_ref, log_to_wandb, num_games, game_idx, optimize_one_game_func, train_cpus
        )
    elif mode == ParallelismMode.NONE:
        return _process_sequential(cfr, cfr_ref, log_to_wandb, num_games, game_idx, optimize_one_game_func)


def _process_unordered(
    cfr,
    cfr_ref,
    log_to_wandb: bool,
    num_games: int,
    game_idx: int,
    optimize_one_game_func: Callable,
    train_cpus: int,
):
    """Process results in unordered manner."""
    # Launch initial batch
    futures = []
    init_batch_size = get_initial_batch_size(ParallelismMode.PARALLEL_UNORDERED_UPDATE, train_cpus, num_games)
    while game_idx < num_games and len(futures) < init_batch_size:
        futures.append((game_idx, optimize_one_game_func(game_idx, cfr_ref)))
        game_idx += 1

    while futures:
        # Wait for one future to finish
        done_ids, _ = ray.wait([f for _, f in futures], num_returns=1)
        (done_id,) = done_ids

        # Find and remove the completed future
        futures.remove(next((i, f) for i, f in futures if f == done_id))

        # Retrieve result
        cfr_updates, metrics = ray.get(done_id)

        # Log to wandb if enabled
        if log_to_wandb:
            wandb.log(metrics)

        # Update CFR object
        cfr.apply_updates(cfr_updates)

        # Snapshot the CFR object
        cfr_ref = ray.put(cfr)

        # Launch next job if any remain
        if should_launch_next_worker(
            ParallelismMode.PARALLEL_UNORDERED_UPDATE, game_idx, num_games, len(futures), train_cpus
        ):
            futures.append((game_idx, optimize_one_game_func(game_idx, cfr_ref)))
            game_idx += 1


def _process_batch_ordered(
    cfr,
    cfr_ref,
    log_to_wandb: bool,
    num_games: int,
    game_idx: int,
    optimize_one_game_func: Callable,
    train_cpus: int,
):
    """Process batches of games with ordered updates - batched pipeline processing."""
    # Process all games in batches
    while game_idx < num_games:
        # Launch a full batch with the current CFR state
        batch_futures = []
        init_batch_size = get_initial_batch_size(
            ParallelismMode.PARALLEL_BATCH_ORDERED_UPDATE, train_cpus, num_games - game_idx
        )

        # Launch init_batch_size games (or remaining games)
        while game_idx < num_games and len(batch_futures) < init_batch_size:
            future = optimize_one_game_func(game_idx, cfr_ref)
            batch_futures.append((game_idx, future))
            game_idx += 1

        # Wait for entire batch to complete
        batch_results = []
        for game_idx_in_batch, future in batch_futures:
            cfr_updates, metrics = ray.get(future)
            batch_results.append((game_idx_in_batch, cfr_updates, metrics))

        # Apply updates from this batch (in game order for determinism)
        batch_results.sort(key=lambda x: x[0])  # Sort by game index

        for game_idx_in_batch, cfr_updates, metrics in batch_results:
            # Log to wandb if enabled
            if log_to_wandb:
                wandb.log(metrics)

            # Update CFR object
            cfr.apply_updates(cfr_updates)

        # Update CFR state for next batch
        cfr_ref = ray.put(cfr)


def _process_sequential(
    cfr,
    cfr_ref,
    log_to_wandb: bool,
    num_games: int,
    game_idx: int,
    optimize_one_game_func: Callable,
):
    """Sequential processing - no parallelism."""
    # Process games one by one
    while game_idx < num_games:
        # Launch one game
        future = optimize_one_game_func(game_idx, cfr_ref)

        # Wait for it to complete
        cfr_updates, metrics = ray.get(future)

        # Log to wandb if enabled
        if log_to_wandb:
            wandb.log(metrics)

        # Update CFR object
        cfr.apply_updates(cfr_updates)

        # Snapshot the CFR object
        cfr_ref = ray.put(cfr)

        # Move to next game
        game_idx += 1


class ParallelismStrategy:
    """Strategy pattern for different parallelism modes in CFR training."""

    def __init__(self, mode: str, train_cpus: int):
        try:
            self.mode = ParallelismMode(mode)
        except ValueError:
            valid_modes = [mode.value for mode in ParallelismMode]
            raise ValueError(f"Invalid parallelism_mode: {mode}. Must be one of: {valid_modes}")
        self.train_cpus = train_cpus

    def get_initial_batch_size(self, num_games: int) -> int:
        """Get the initial batch size for launching workers."""
        return get_initial_batch_size(self.mode, self.train_cpus, num_games)

    def should_launch_next_worker(self, game_idx: int, num_games: int, active_futures: int) -> bool:
        """Determine if we should launch the next worker."""
        return should_launch_next_worker(self.mode, game_idx, num_games, active_futures, self.train_cpus)

    def process_results(
        self,
        cfr,
        cfr_ref,
        log_to_wandb: bool,
        num_games: int,
        game_idx: int,
        optimize_one_game_func,
    ):
        """Process results according to the selected strategy."""
        return process_results(
            self.mode, cfr, cfr_ref, log_to_wandb, num_games, game_idx, optimize_one_game_func, self.train_cpus
        )
