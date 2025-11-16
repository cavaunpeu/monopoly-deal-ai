from __future__ import annotations

from collections import Counter, deque
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, cast


if TYPE_CHECKING:
    from models.cfr.selector import CFRActionSelector

import numpy as np
from tqdm import tqdm

from game.action import AbstractAction, BaseActionResolver, WrappedAction
from game.cards import Card
from game.config import GameConfig
from game.constants import NUM_PLAYERS
from game.game import Game, PlayerSpec
from game.state import ABSTRACTION_NAME_TO_CLS, RESOLVER_NAME_TO_CLS, BaseStateAbstraction, GameState
from game.util import Serializable
from models.cfr.constants import EPSILON
from models.types import Policy


def sort_dict_by_keys(d: dict) -> dict:
    """Sort a dictionary by its keys.

    Args:
        d: The dictionary to sort.

    Returns:
        The sorted dictionary.
    """
    return {k: v for k, v in sorted(d.items())}


class CounterfactualReachProbUpdate(NamedTuple):
    key: str
    cf_reach_prob: float


class RegretUpdate(NamedTuple):
    key: str
    action: AbstractAction
    regret: float


class PolicyManagerUpdate(NamedTuple):
    key: str
    policy: Policy


@dataclass
class IncrementalEstimator(Serializable):
    sum: float = 0
    n: float = 0

    def update(self, x):
        self.sum = self.sum * 0.99 + x
        self.n = self.n * 0.99 + 1

    @property
    def mean(self):
        if self.n == 0:
            return 0
        return self.sum / self.n

    def to_json(self):
        return {
            "sum": float(self.sum),
            "n": float(self.n),
        }

    @classmethod
    def from_json(cls, data: dict):
        return cls(sum=float(data["sum"]), n=int(data["n"]))


@dataclass
class PlayerBuffer:
    buffer_size: int
    buffer: Dict[str, deque[List[float]]] = field(default_factory=dict)

    def update(self, key: str, encoded_probs: List[float]) -> None:
        dq = self.buffer.setdefault(key, deque(maxlen=self.buffer_size))
        dq.append(encoded_probs)

    def get(self, key: str) -> List[List[float]]:
        dq = self.buffer.get(key)
        return list(dq) if dq else []

    def __contains__(self, key: str) -> bool:
        return key in self.buffer

    def __iter__(self):
        return iter(self.buffer)

    def to_json(self) -> Dict:
        return {
            "buffer": {k: list(dq) for k, dq in self.buffer.items()},
            "buffer_size": self.buffer_size,
        }

    @classmethod
    def from_json(cls, data: Dict) -> "PlayerBuffer":
        buffer = {}
        for k, enc_list in data["buffer"].items():
            buffer[k] = deque((list(enc) for enc in enc_list), maxlen=data["buffer_size"])
        return cls(buffer_size=data["buffer_size"], buffer=buffer)


def get_median_abstract_action_probs(policy_manager: "PolicyManager", player_idx: int) -> dict[AbstractAction, float]:
    """Get median probabilities for each abstract action across all policies for a player.

    Args:
        policy_manager: The policy manager containing player policies.
        player_idx: Index of the player to get statistics for.

    Returns:
        Dictionary mapping abstract actions to their median probabilities.
    """
    buffer = policy_manager.get_player_buffer(player_idx)
    action_to_probs: dict[AbstractAction, list[float]] = {}

    # For each abstract action, get all keys where it was available
    for aa, keys in policy_manager.action_to_keys.items():
        if not keys:
            continue

        probs = []
        for key in keys:
            # Compute the average policy for this key
            dq = buffer.get(key)
            if dq:
                encoded = np.asarray(dq, dtype=float)
                avg_policy = encoded.mean(axis=0)
                aa_idx = aa.encode()
                if aa_idx < len(avg_policy):
                    probs.append(float(avg_policy[aa_idx]))

        if probs:
            action_to_probs[aa] = probs

    # Return median probabilities for each abstract action
    return {aa: float(np.median(probs)) for aa, probs in action_to_probs.items()}


@dataclass
class RuntimePolicyManager:
    player_buffer: PlayerBuffer
    update_count: Counter[str] = field(default_factory=Counter)
    _cache: dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)

    def get(self, state: "GameState") -> "Policy":
        # Get the infoset key
        key = state.key
        # Get the player buffer
        dq = self.player_buffer.get(key)
        if not dq:
            return Policy.build_uniform_policy(state)
        # Compute the average policy
        if key not in self._cache:
            encoded = np.asarray(dq, dtype=float)
            self._cache[key] = encoded.mean(axis=0)
        # Return the average policy
        return Policy.from_encoded_probs(self._cache[key].tolist(), state)


@dataclass
class PolicyManager(Serializable):
    buffer_size: int
    update_count: Counter[str] = field(default_factory=Counter)
    player_buffers: List[PlayerBuffer] = field(default_factory=list, repr=False)
    action_to_keys: dict[AbstractAction, set[str]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.player_buffers:
            self.player_buffers = [PlayerBuffer(self.buffer_size) for _ in range(NUM_PLAYERS)]
        self._cache = {}

    def update(self, update: "PolicyManagerUpdate") -> None:
        key = update.key
        # Update update count
        self.update_count[key] += 1
        # Parse the key
        parsed = GameState.parse_key(update.key)
        # Update player buffer
        self.player_buffers[parsed.player_idx].update(key, update.policy.encode_probs())
        # Update action_to_keys mapping
        for action in update.policy.actions:
            aa = action.abstract_action
            if aa not in self.action_to_keys:
                self.action_to_keys[aa] = set()
            self.action_to_keys[aa].add(key)

    def get_average_policy(self, state: "GameState") -> "Policy":
        return self._get_policy(state, latest=False)

    def get_latest_policy(self, state: "GameState") -> "Policy":
        return self._get_policy(state, latest=True)

    def _get_policy(self, state: "GameState", latest: bool) -> "Policy":
        key = state.key
        # Parse the key
        parsed = GameState.parse_key(key)
        # Get player buffer
        dq = self.player_buffers[parsed.player_idx].get(key)
        # If no policy, use uniform policy
        if not dq:
            return Policy.build_uniform_policy(state)
        # If latest, use the latest policy
        elif latest:
            return Policy.from_encoded_probs(dq[-1], state)
        # Get average policy from cache
        if key not in self._cache:
            encoded = np.asarray(dq, dtype=float)
            self._cache[key] = encoded.mean(axis=0)
        return Policy.from_encoded_probs(self._cache[key].tolist(), state)

    def get_update_count(self, key: str) -> int:
        return int(self.update_count.get(key, 0))

    def __contains__(self, key: str) -> bool:
        return any(key in player_buf for player_buf in self.player_buffers)

    def get_player_buffer(self, player_idx: int) -> PlayerBuffer:
        return self.player_buffers[player_idx]

    def to_json(self) -> dict:
        return {
            "buffer_size": self.buffer_size,
            "update_count": dict(self.update_count),
            "player_buffers": [buf.to_json() for buf in self.player_buffers],
            "action_to_keys": {
                aa.encode(): sorted(list(keys))
                for aa, keys in sorted(self.action_to_keys.items(), key=lambda item: item[0].encode())
            },
        }

    @classmethod
    def from_json(cls, data: dict) -> "PolicyManager":
        return cls(
            buffer_size=data["buffer_size"],
            update_count=Counter(data["update_count"]),
            player_buffers=[PlayerBuffer.from_json(buf_data) for buf_data in data["player_buffers"]],
            action_to_keys={
                AbstractAction.decode(idx): set(keys) for idx, keys in data.get("action_to_keys", {}).items()
            },
        )

    def get_runtime_policy_manager(self, player_idx: int) -> "RuntimePolicyManager":
        return RuntimePolicyManager(self.player_buffers[player_idx], update_count=self.update_count)


@dataclass
class RegretManager(Serializable, Mapping[str, dict[AbstractAction, IncrementalEstimator]]):
    _estimators: dict[str, dict[AbstractAction, IncrementalEstimator]] = field(default_factory=lambda: {})

    def update(self, key: str, action: AbstractAction, regret: float):
        if key not in self._estimators:
            self._estimators[key] = {}
        if action not in self._estimators[key]:
            self._estimators[key][action] = IncrementalEstimator()
        self._estimators[key][action].update(regret)

    def get_means_for_action(self, action: AbstractAction, player_idx: int) -> list[float]:
        vals = []
        for key in self._estimators:
            if player_idx == GameState.player_idx_from_key(key):
                for aa, estimator in self._estimators[key].items():
                    if aa == action:
                        vals.append(estimator.mean)
        return vals

    def to_json(self) -> dict:
        return {
            key: {a.encode(): est.to_json() for a, est in action2est.items()}
            for key, action2est in self._estimators.items()
        }

    @classmethod
    def from_json(cls, data: dict) -> "RegretManager":
        return cls(
            _estimators={
                key: {AbstractAction.decode(abstract_action): IncrementalEstimator.from_json(estimator)}
                for key, val in data.items()
                for abstract_action, estimator in val.items()
            }
        )

    def __getitem__(self, key: str) -> dict[AbstractAction, IncrementalEstimator]:
        return self._estimators[key]

    def __iter__(self):
        return iter(self._estimators)

    def __len__(self):
        return len(self._estimators)


class ReachProbabilityCounter(Serializable):
    def __init__(self, data: Mapping[str, float] | None = None):
        self._data: dict[str, float] = dict(data or {})

    def __getitem__(self, key: str) -> float:
        return self._data.get(key, 0.0)

    def __setitem__(self, key: str, value: float) -> None:
        self._data[key] = float(value)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def items(self):
        return self._data.items()

    def to_json(self) -> dict[str, float]:
        return dict(self._data)

    @classmethod
    def from_json(cls, data: dict[str, float]) -> "ReachProbabilityCounter":
        return cls(data)


def prepare_rollouts(
    *,
    actions: list[WrappedAction],
    game: Game,
    game_idx: int,
    policy_manager: PolicyManager | RuntimePolicyManager,
    max_turns_per_game: int,
    max_streaks: int | float,
    sims_per_action: int,
    opponent_hand_posterior: dict[tuple[Card, ...], float] | None = None,
    training_mode: bool = False,
    verbose: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Prepare rollout inputs for CFR simulation.

    This function creates all the necessary input configurations for running
    Monte Carlo rollouts in CFR.

    Args:
        actions: List of actions to simulate rollouts for.
        game: Current game state to base rollouts on.
        game_idx: Index of the current game for tracking.
        policy_manager: Policy manager for getting opponent policies during rollouts.
        max_turns_per_game: Maximum turns allowed per game simulation.
        max_streaks: Maximum consecutive actions per player before turn ends.
        sims_per_action: Number of Monte Carlo simulations per action.
        opponent_hand_posterior: Probability distribution over possible opponent hands.
            If None and training_mode=True, uses actual opponent hand.
        training_mode: Whether to run in training mode (uses actual opponent hand).
        verbose: Whether to enable verbose logging.

    Returns:
        Tuple of (rollout_index, rollout_inputs) where:
        - rollout_index: List of metadata for each action being rolled out
        - rollout_inputs: List of complete input configurations for each simulation
    """
    rollout_index, rollout_inputs = [], []

    # Compose opponent hand posterior
    if not opponent_hand_posterior:
        if training_mode:
            opponent_hand_posterior = {tuple(game.opponent.hand): 1}
        else:
            raise ValueError("Opponent hand posterior must be provided.")

    # Iterate through player actions
    for wrapped in actions:
        # Copy game
        g = game.clone(flush_draw_probs=True)
        # Store opponent index
        opponent_idx = g.opponent.index
        # Store the acting player before taking the action
        acting_player_idx = g.player.index
        # Take game step
        g.step(selected_action=wrapped.action)
        # Store rollout keys
        rollout_index.append(
            {
                "wrapped_action": wrapped,
                "num_opponent_hands": len(opponent_hand_posterior),
            }
        )
        # Iterate through opponent hands
        for opponent_hand, opponent_hand_prob in opponent_hand_posterior.items():
            if len(opponent_hand) != game.opponent_hand_size:
                raise ValueError(
                    f"Fictional opponent hand must have the same length as the actual opponent hand. Expected {game.opponent_hand_size}, got {len(opponent_hand)}."
                )

            for i in range(sims_per_action):
                # Copy game
                _g = g.clone(flush_draw_probs=True)

                # Set opponent hand
                _g.set_player_hand(opponent_idx, opponent_hand)

                # Store rollout input
                rollout_inputs.append(
                    {
                        "game_idx": game_idx,
                        "rollout_idx": i,
                        "wrapped_action": wrapped,
                        "game": _g,
                        "player_idx": acting_player_idx,  # Use the player whose turn it was to act
                        "policy_manager": policy_manager,
                        "max_turns": max_turns_per_game,
                        "max_streaks": max_streaks,
                        "opponent_hand_prob": opponent_hand_prob,
                        "verbose": verbose,
                    }
                )
    return rollout_index, rollout_inputs


def perform_rollouts(rollout_inputs: list[dict], verbose: bool = False) -> list[dict]:
    """Perform rollouts for all input configurations.

    This function executes the actual game simulations for each rollout input,
    computing utilities and probabilities for CFR training.

    Args:
        rollout_inputs: List of rollout input dictionaries containing game state,
            policy manager, and simulation parameters.
        verbose: Whether to enable verbose logging with progress bars.

    Returns:
        List of rollout result dictionaries, each containing utility, probability,
        and number of turns for the simulation.
    """
    iterator = tqdm(rollout_inputs, desc="Performing rollouts...") if verbose else rollout_inputs
    return [simulate_rollout(inp) for inp in iterator]


def compute_expected_utility(utilities: Sequence[float], probs: Sequence[float]) -> float:
    """Compute expected utility given utilities and probabilities.

    Args:
        utilities: List of utility values.
        probs: List of corresponding probabilities.

    Returns:
        Expected utility value.
    """
    utilities_arr = np.array(utilities)
    probs_arr = np.array(probs)

    # Calculate expected utility
    probs_arr_safe = probs_arr + EPSILON
    expected_utility = np.round(
        utilities_arr @ np.exp(np.log(probs_arr_safe) - np.log(np.sum(probs_arr_safe))),
        decimals=15,
    )

    return float(expected_utility)


def aggregate_rollouts(
    rollout_index: list[dict],
    rollouts: list[dict],
    sims_per_action: int,
    uniform_external_sampling: bool,
) -> dict[WrappedAction, float]:
    """Aggregate rollout results to compute expected utilities for each action.

    This function processes the results from Monte Carlo rollouts and computes
    the expected utility for each action, handling probability normalization
    and uniform external sampling if enabled.

    Args:
        rollout_index: List of metadata for each action being rolled out.
        rollouts: List of rollout results from perform_rollouts().
        sims_per_action: Number of simulations per action (used for grouping results).
        uniform_external_sampling: Whether to use uniform probabilities instead of
            actual trajectory probabilities.

    Returns:
        Dictionary mapping each WrappedAction to its expected utility value.
    """
    action_to_utility = {}
    # Process rollouts for each abstract action
    for i, idx in enumerate(rollout_index):
        # Unpack input
        wrapped_action, num_opponent_hands = (
            idx["wrapped_action"],
            idx["num_opponent_hands"],
        )
        step_size = sims_per_action * num_opponent_hands
        results = rollouts[i * step_size : (i + 1) * step_size]
        utilities, probs = zip(*[(r["utility"], r["probability"]) for r in results])
        # Normalize probabilities
        total_prob = sum(probs)
        probs = [prob / total_prob for prob in probs]
        # If using uniform external sampling, set all probabilities to 1 / len(probs)
        if uniform_external_sampling:
            probs = [1 / len(probs) for _ in probs]
        # Compute expected utility
        expected_utility = compute_expected_utility(utilities, probs)
        action_to_utility[wrapped_action] = expected_utility
    return action_to_utility


def execute_rollouts(
    *,
    actions: list[WrappedAction],
    game: Game,
    game_idx: int,
    policy_manager: PolicyManager,
    sims_per_action: int,
    max_turns_per_game: int,
    max_streaks: int | float,
    opponent_hand_posterior: dict[tuple[Card, ...], float] | None = None,
    uniform_external_sampling: bool = True,
    training_mode: bool = False,
    verbose: bool = False,
) -> dict[WrappedAction, float]:
    """Execute complete rollout pipeline for CFR action evaluation.

    This is the main entry point for running Monte Carlo rollouts in CFR.
    It orchestrates the entire process: preparing inputs, performing simulations,
    and aggregating results to compute expected utilities.

    Args:
        actions: List of actions to evaluate through rollouts.
        game: Current game state to base rollouts on.
        game_idx: Index of the current game for tracking.
        policy_manager: Policy manager for getting opponent policies.
        sims_per_action: Number of Monte Carlo simulations per action.
        max_turns_per_game: Maximum turns allowed per game simulation.
        max_streaks: Maximum consecutive actions per player before turn ends.
        opponent_hand_posterior: Probability distribution over possible opponent hands.
        uniform_external_sampling: Whether to use uniform probabilities instead of actual trajectory probabilities.
        training_mode: Whether to run in training mode (uses actual opponent hand).
        verbose: Whether to enable verbose logging.

    Returns:
        Dictionary mapping each WrappedAction to its expected utility value.
    """
    # Prepare rollouts
    rollout_index, rollout_inputs = prepare_rollouts(
        actions=actions,
        game=game,
        game_idx=game_idx,
        policy_manager=policy_manager,
        max_turns_per_game=max_turns_per_game,
        max_streaks=max_streaks,
        sims_per_action=sims_per_action,
        opponent_hand_posterior=opponent_hand_posterior,
        training_mode=training_mode,
        verbose=verbose,
    )
    # Perform rollouts
    rollouts = perform_rollouts(rollout_inputs)

    # Aggregate rollout results
    action_to_utility = aggregate_rollouts(
        rollout_index,
        rollouts,
        sims_per_action,
        uniform_external_sampling,
    )
    return action_to_utility


def simulate_rollout(args: dict) -> dict:
    """Simulate a single rollout for CFR.

    Args:
        args: Dictionary containing rollout parameters.

    Returns:
        Dictionary containing rollout results.
    """
    return _simulate_rollout(**args)


def _simulate_rollout(
    *,
    game_idx: int,
    rollout_idx: int,
    wrapped_action: WrappedAction,
    game: Game,
    player_idx: int,
    policy_manager: PolicyManager,
    max_turns: int,
    max_streaks: int | float,
    opponent_hand_prob: float,
    verbose: bool,
) -> dict:
    """Simulate a single Monte Carlo rollout for CFR training.

    This function executes one complete game simulation from the current state
    until termination, using the specified policies for action selection.
    It computes the utility and probability for the rollout trajectory.

    Args:
        game_idx: Index of the current game for tracking.
        rollout_idx: Index of this specific rollout for tracking.
        wrapped_action: The action being evaluated in this rollout.
        game: Game state to simulate from (will be modified during simulation).
        player_idx: Index of the player whose utility is being computed.
        policy_manager: Policy manager for getting opponent policies.
        max_turns: Maximum number of turns allowed in the simulation.
        max_streaks: Maximum consecutive actions per player before turn ends.
        opponent_hand_prob: Probability of the opponent hand configuration.
        verbose: Whether to enable verbose logging.

    Returns:
        Dictionary containing:
        - utility: Terminal utility for the player (1.0 for win, -1.0 for loss, 0.0 for draw)
        - probability: Joint probability of the rollout trajectory
        - num_turns: Number of turns taken in the simulation
    """
    # Rollout execution for action

    if verbose:
        print(
            f"Game {game_idx} | Turn Idx {game.turn_state.turn_idx} | Rollout Idx {rollout_idx} | {wrapped_action.abstract_action} | Simulating...",
            flush=True,
        )

    # Begin simulation
    action_probs, turns, player_actions, streaks = [], [], [], 0
    while True:
        # If game is over or exceeds limits, compute utility and return
        if game.over or len(turns) >= max_turns or streaks >= max_streaks:
            # Compute joint probability of trajectory
            # Add small epsilon to prevent log(0) numerical issues
            prob = np.exp(np.sum(np.log(np.array(action_probs + game.chance_probs) + EPSILON)))

            # Calculate simple terminal reward: +1 for win, -1 for loss, 0 for draw
            if game.winner is None:
                utility = 0.0  # Draw
            elif game.winner.index == player_idx:
                utility = 1.0  # Win
            else:
                utility = -1.0  # Loss

            # Break out of loop when game is over
            break
        # Get the player's policy
        if game.player.index == player_idx:
            # In CFR, the rollout player uses their current policies
            policy = policy_manager.get_latest_policy(game.state)
        else:
            # In CFR, the rollout opponent uses their average policy
            policy = policy_manager.get_average_policy(game.state)
        # Select action from policy
        wrapped_action = policy.sample()
        # Record player action
        if game.player.index == player_idx:
            player_actions.append(wrapped_action)
        # Record action prob
        action_probs.append(policy[wrapped_action])
        # Record turn
        turns.append(
            {
                "turn_idx": game.turn_state.turn_idx,
                "streak_idx": game.turn_state.streak_idx,
                "is_player": game.player.index == player_idx,
                "policy": policy,
                "wrapped_action": wrapped_action,
            }
        )
        # If game is over, take game step
        if not game.over:
            game.step(selected_action=wrapped_action.action)
        # Record streak
        if game.turn_state.streak_idx == 0:
            streaks += 1

    # Return everything needed from rollout (outside the loop)
    return {
        "utility": utility,
        "probability": opponent_hand_prob * prob,
        "num_turns": len(turns),
    }


class CFRStep(NamedTuple):
    wrapped_action: WrappedAction
    cf_reach_prob_update: CounterfactualReachProbUpdate
    regret_updates: list[RegretUpdate]
    key: str
    policy_manager_update: Optional[PolicyManagerUpdate] = None
    key_abstraction_mapping: Optional[dict] = None  # key -> abstraction_json


@dataclass
class CFR(Serializable):
    """Counterfactual Regret Minimization (CFR) algorithm implementation.

    This class implements the CFR algorithm for training policies in imperfect
    information games, with support for parallel training and checkpointing.
    """

    # Internal state
    _game: Game | None = field(init=False, default=None)
    _game_idx: int | None = field(init=False, default=None)

    # Core config
    game_config: GameConfig
    abstraction_cls: type[BaseStateAbstraction]
    resolver_cls: type[BaseActionResolver]

    # Policy manager
    policy_manager: PolicyManager

    # Initial player
    target_player_index: int

    # Exploration
    epsilon: float

    # Game parameters
    sims_per_action: int
    max_turns_per_game: int
    max_streaks: int | float = int(1e10)

    # Flags
    uniform_external_sampling: bool = True
    verbose: bool = False

    # Other dependencies / managers
    regret_manager: RegretManager = field(default_factory=RegretManager)
    cf_reach_prob_counter: ReachProbabilityCounter = field(default_factory=ReachProbabilityCounter)

    @property
    def game(self) -> Game:
        """Get the current game instance.

        Returns:
            The current game instance.

        Raises:
            ValueError: If no game has been set.
        """
        if self._game is None:
            raise ValueError("Game must be set via `cfr.game = ...` before playing.")
        return self._game

    @property
    def game_idx(self) -> int:
        """Get the current game index.

        Returns:
            The current game index.

        Raises:
            ValueError: If no game index has been set.
        """
        if self._game_idx is None:
            raise ValueError("Game index must be set via `cfr.game_idx = ...` before playing.")
        return self._game_idx

    def instantiate_new_game(
        self,
        random_seed: int,
        init_player_index: int,
    ) -> Game:
        """Create a new game instance with the given parameters.

        Args:
            random_seed: Seed for random number generation.
            init_player_index: Index of the player to start the game.

        Returns:
            New Game instance.
        """
        # For CFR training, both players use the same abstraction/resolver from the model
        player_spec = PlayerSpec(
            abstraction_cls=self.abstraction_cls,
            resolver_cls=self.resolver_cls,
        )
        player_specs = {0: player_spec, 1: player_spec}

        return Game(
            config=self.game_config,
            init_player_index=init_player_index,
            player_specs=player_specs,
            random_seed=random_seed,
        )

    @game.setter
    def game(self, game: Game):
        self._game = game
        self.action_probs = game.init_player_buffers()

    @game_idx.setter
    def game_idx(self, game_idx: int):
        self._game_idx = game_idx

    def compute_max_expected_regret(self, player_idx: int) -> float:
        """Compute the maximum expected regret for a specific player.

        Args:
            player_idx: Index of the player to compute regret for.

        Returns:
            Maximum expected regret value.
        """
        max_regrets = []
        weights = []
        for key, action2regret in self.regret_manager.items():
            if GameState.parse_key(key).player_idx == player_idx:
                values = [val.mean for val in action2regret.values()]
                max_regret = max(values)
                weight = self.cf_reach_prob_counter[key]
                max_regrets.append(max_regret)
                weights.append(weight)
        # Compute expected max regret
        if not (max_regrets and weights):
            return 0
        return float(np.array(max_regrets) @ np.array(weights) / sum(weights))

    def apply_updates(self, cfr_steps: list[CFRStep]) -> None:
        """Apply a list of CFR updates to the model.

        Args:
            cfr_steps: List of CFR step updates to apply.
        """
        for step in cfr_steps:
            # Update counterfactual reach probability
            self.cf_reach_prob_counter[step.cf_reach_prob_update.key] += step.cf_reach_prob_update.cf_reach_prob
            # Update regret manager
            for u in step.regret_updates:
                self.regret_manager.update(u.key, u.action, u.regret)
            # Update policy manager
            if step.policy_manager_update:
                self.policy_manager.update(step.policy_manager_update)

    def optimize(self) -> CFRStep:
        # Define game
        game = self.game

        # Define CFR step
        cfr_step: dict[str, Any] = {"regret_updates": []}

        # Define key and capture abstraction mapping (only for target player)
        key = game.state.key

        # Get player policy (if this is the first time we've seen this infoset, will get uniform policy)
        policy = self.policy_manager.get_latest_policy(game.state)

        # Define list of wrapped actions
        actions = policy.actions

        # Execute rollouts
        action_to_utility = execute_rollouts(
            actions=actions,
            game=game,
            game_idx=self.game_idx,
            policy_manager=self.policy_manager,
            sims_per_action=self.sims_per_action,
            max_turns_per_game=self.max_turns_per_game,
            max_streaks=self.max_streaks,
            uniform_external_sampling=self.uniform_external_sampling,
            training_mode=True,
            verbose=self.verbose,
        )

        # Cast action_to_utility to dict[WrappedAction, float]
        action_to_utility = cast(dict[WrappedAction, float], action_to_utility)

        # Compute counterfactual infoset reach probability
        opponent_action_probs = self.action_probs[game.opponent.index]

        # Add small epsilon to prevent log(0) numerical issues
        cf_infoset_reach_prob = np.exp(np.sum(np.log(np.array(opponent_action_probs + game.chance_probs) + EPSILON)))

        # Store counterfactual reach probability and update
        self.cf_reach_prob_counter[key] += cf_infoset_reach_prob
        cfr_step["cf_reach_prob_update"] = CounterfactualReachProbUpdate(key, cf_infoset_reach_prob)

        # Compute strategy utility
        strategy_utility = sum([policy[a] * action_to_utility[a] for a in actions])

        # Update regrets
        action_to_regret = {}
        for a in actions:
            # How much better is the action utility than the strategy utility? Ignore reach probability to reduce variance
            regret = action_to_utility[a] - strategy_utility
            self.regret_manager.update(key, a.abstract_action, regret)

            # Store regret update
            cfr_step["regret_updates"].append(RegretUpdate(key, a.abstract_action, regret))
            action_to_regret[a] = regret

        # Compute clipped average regrets
        clipped_regrets = [max(self.regret_manager[key][a.abstract_action].sum, 0) for a in actions]

        # Never learn 'other'
        other_idx = next((i for i, a in enumerate(actions) if a.abstract_action == AbstractAction.OTHER), None)
        if other_idx is not None:
            clipped_regrets[other_idx] = 0

        # If any card actions have positive regret, clamp non-card actions to 0
        has_card_regret = any(regret > 0 for a, regret in zip(actions, clipped_regrets) if a.action.plays_card)
        if has_card_regret:
            clipped_regrets = [0 if not a.action.plays_card else regret for a, regret in zip(actions, clipped_regrets)]

        # Update policy
        total_regret = sum(clipped_regrets)

        if total_regret > 0:
            # Compute policy probabilities
            probs = [regret / total_regret for regret in clipped_regrets]

            # Update policy
            update = PolicyManagerUpdate(key, Policy(actions, probs))
            self.policy_manager.update(update)
            cfr_step["policy_manager_update"] = update

        # Get policy (if total regret was 0, will get previous policy)
        policy = self.policy_manager.get_latest_policy(game.state)

        # Apply epsilon-greedy exploration during training
        if np.random.random() < self.epsilon:
            wrapped_action = np.random.choice(np.array(actions, dtype=object))
        else:
            wrapped_action = policy.sample()

        # Record selected action
        cfr_step["wrapped_action"] = wrapped_action

        # Record action prob
        prob = policy[wrapped_action]
        self.action_probs[game.player.index].append(prob)

        # Record key
        cfr_step["key"] = key

        # Return the CFR step
        return CFRStep(**cfr_step)

    def to_json(self) -> dict:
        # Core config
        data = {
            "game_config": sort_dict_by_keys(self.game_config.to_json()),
            "abstraction_cls": self.abstraction_cls.__name__,
            "resolver_cls": self.resolver_cls.__name__,
        }

        # Initial player
        data.update(
            {
                "target_player_index": self.target_player_index,
            }
        )

        # Game parameters
        data.update(
            {
                "sims_per_action": self.sims_per_action,
                "max_turns_per_game": self.max_turns_per_game,
                "max_streaks": self.max_streaks,
            }
        )

        # Flags
        data.update(
            {
                "uniform_external_sampling": self.uniform_external_sampling,
                "verbose": self.verbose,
                "epsilon": self.epsilon,
            }
        )

        # Dependencies / Managers
        data.update(
            {
                "policy_manager": sort_dict_by_keys(self.policy_manager.to_json()),
                "regret_manager": sort_dict_by_keys(self.regret_manager.to_json()),
                "cf_reach_prob_counter": sort_dict_by_keys(self.cf_reach_prob_counter.to_json()),
            }
        )

        # Add model class
        data["model_cls"] = self.__class__.__name__

        return data

    def create_selector(self) -> CFRActionSelector:
        """Create an action selector for this CFR model.

        Returns:
            CFRActionSelector instance configured for this model's target player.
        """
        from models.cfr.selector import CFRActionSelector

        return CFRActionSelector(
            policy_manager=self.policy_manager.get_runtime_policy_manager(self.target_player_index)
        )

    def get_player_spec(self) -> PlayerSpec:
        """Get the PlayerSpec for this model's target player.

        Returns:
            PlayerSpec with the model's abstraction and resolver classes.
        """
        return PlayerSpec(
            abstraction_cls=self.abstraction_cls,
            resolver_cls=self.resolver_cls,
        )

    @classmethod
    def from_json(cls, data: dict) -> "CFR":
        return cls(
            # Core config
            game_config=GameConfig.from_json(data["game_config"]),
            abstraction_cls=ABSTRACTION_NAME_TO_CLS[data["abstraction_cls"]],
            resolver_cls=RESOLVER_NAME_TO_CLS[data["resolver_cls"]],
            # Initial player
            target_player_index=data["target_player_index"],
            # Game parameters
            sims_per_action=data["sims_per_action"],
            max_turns_per_game=data["max_turns_per_game"],
            max_streaks=data["max_streaks"],
            # Flags
            uniform_external_sampling=data["uniform_external_sampling"],
            verbose=data["verbose"],
            epsilon=data["epsilon"],
            # Dependencies / Managers
            policy_manager=PolicyManager.from_json(data["policy_manager"]),
            regret_manager=RegretManager.from_json(data["regret_manager"]),
            cf_reach_prob_counter=ReachProbabilityCounter.from_json(data["cf_reach_prob_counter"]),
        )

    @classmethod
    def from_checkpoint(
        cls,
        load_path: str,
    ) -> "CFR":
        """Load CFR model from checkpoint file (local or GCS).

        Args:
            load_path: Path to checkpoint file (local file path or GCS gs:// path).

        Returns:
            CFR instance loaded from checkpoint.
        """
        from models.checkpoint import load_checkpoint_data

        data = load_checkpoint_data(load_path)
        return cls.from_json(data)


class NullProgressBar:
    def __init__(self):
        self.n = 0

    def update(self, n=1):
        self.n += n


@contextmanager
def null_progress_bar():
    """Create a null progress bar context manager for suppressing progress bars."""
    yield NullProgressBar()
