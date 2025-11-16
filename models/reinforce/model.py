from __future__ import annotations

from abc import abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
from jax import Array, random
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpy as np
import optax

from game.action import AbstractAction, BaseActionResolver, WrappedAction
from game.config import GameConfig
from game.game import PlayerSpec
from game.state import ABSTRACTION_NAME_TO_CLS, RESOLVER_NAME_TO_CLS, BaseStateAbstraction, GameState
from game.util import Serializable
from models.checkpoint import load_checkpoint_data
from models.reinforce.selector import ReinforceActionSelector
from models.types import Policy
from models.util import softmax


if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class ModelAction:
    """One model action."""

    state_key: str
    state_vector_encoding: np.ndarray
    turn_idx: int
    streak_idx: int
    streaking_player_idx: int
    valid_actions: list[AbstractAction]
    action: AbstractAction
    reward: float


@dataclass(frozen=True)
class TrainingSample:
    """One model action."""

    state_key: str
    state_vector_encoding: np.ndarray
    valid_actions: list[AbstractAction]
    action: AbstractAction
    reward_to_go: float


class BaseReinforceModel(Serializable):
    """Base class for reinforcement learning models."""

    def __init__(
        self,
        learning_rate: float,
        target_player_index: int,
        game_config: GameConfig,
        abstraction_cls: type[BaseStateAbstraction],
        resolver_cls: type[BaseActionResolver],
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.target_player_index = target_player_index
        self.game_config = game_config
        self.abstraction_cls = abstraction_cls
        self.resolver_cls = resolver_cls
        self.update_count: Counter[str] = Counter()

    @abstractmethod
    def select(self, actions: list[WrappedAction], state: GameState) -> WrappedAction:
        """Select an action based on the reinforcement learning model."""
        raise NotImplementedError

    @abstractmethod
    def update(self, batch: list[TrainingSample]) -> None:
        """Update the reinforcement learning model."""
        raise NotImplementedError

    @abstractmethod
    def actions_to_probs(
        self, actions: list[AbstractAction], state_key: str, state_vector_encoding: np.ndarray
    ) -> np.ndarray:
        """Convert actions to probabilities."""
        raise NotImplementedError

    def create_selector(self) -> "ReinforceActionSelector":
        """Create an action selector for this Reinforce model.

        Returns:
            ReinforceSelector instance configured for this model.
        """
        from models.reinforce.selector import ReinforceActionSelector

        return ReinforceActionSelector(model=self)

    def get_player_spec(self) -> PlayerSpec:
        """Get the PlayerSpec for this model's target player.

        Returns:
            PlayerSpec with the model's abstraction and resolver classes.
        """
        from game.game import PlayerSpec

        return PlayerSpec(
            abstraction_cls=self.abstraction_cls,
            resolver_cls=self.resolver_cls,
        )

    def copy(self) -> BaseReinforceModel:
        """Create a deep copy of this model for use as a snapshot in self-play.

        Returns:
            A new BaseReinforceModel instance with the same state as this model.
        """
        return self.from_json(self.to_json())

    @classmethod
    def from_checkpoint(cls, load_path: str) -> BaseReinforceModel:
        """Load model from checkpoint file (local or GCS).

        Args:
            load_path: Path to checkpoint file (local file path or GCS gs:// path).

        Returns:
            BaseReinforceModel instance loaded from checkpoint.
        """
        data = load_checkpoint_data(load_path)
        return cls.from_json(data)

    def to_json(self) -> dict:
        """Convert model to JSON-serializable dictionary."""
        raise NotImplementedError

    @classmethod
    def from_json(cls, data: dict) -> BaseReinforceModel:
        """Create model from JSON dictionary."""
        raise NotImplementedError


def random_layer_params(m: int, n: int, key: Array, scale: float = 1e-2) -> tuple[Array, Array]:
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))


def init_network_params(sizes: list[int], key: Array) -> list[tuple[Array, Array]]:
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


def relu(x: Array) -> Array:
    return jnp.maximum(x, 0)


def compute_logits(params: list[tuple[Array, Array]], x: Array) -> Array:
    activations = x
    for w, b in params[:-1]:
        activations = relu(jnp.dot(w, activations) + b)
    w_final, b_final = params[-1]
    logits = jnp.dot(w_final, activations) + b_final
    return logits


batch_compute_logits = jax.jit(jax.vmap(compute_logits, in_axes=(None, 0)))


class TabularReinforceModel(BaseReinforceModel, Serializable):
    """Tabular reinforcement learning model."""

    def __init__(
        self,
        learning_rate: float,
        target_player_index: int,
        game_config: GameConfig,
        abstraction_cls: type[BaseStateAbstraction],
        resolver_cls: type[BaseActionResolver],
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            target_player_index=target_player_index,
            game_config=game_config,
            abstraction_cls=abstraction_cls,
            resolver_cls=resolver_cls,
        )
        self.lookup: defaultdict[str, np.ndarray] = defaultdict(lambda: np.zeros(len(AbstractAction), dtype=float))

    def select(self, actions: list[WrappedAction], state: GameState) -> WrappedAction:
        # Get abstract actions
        abstract_actions = [a.abstract_action for a in actions]
        # Get symmetric state key (no player index)
        state_key = state.symmetric_key
        # Get action indices and probabilities
        probs = self.actions_to_probs(abstract_actions, state_key, state.vector_encoding())
        # Sample action
        return Policy(actions=actions, probs=probs.tolist()).sample()

    def actions_to_probs(
        self, actions: list[AbstractAction], state_key: str, state_vector_encoding: np.ndarray
    ) -> np.ndarray:
        # Get action indices
        idxs = [a.encode() for a in actions]
        # Get logits for action
        logits = self.lookup[state_key][idxs]
        # Get action probabilities
        return softmax(logits)

    def update(self, batch: list[TrainingSample]) -> None:
        """Update the reinforcement learning model."""
        for sample in batch:
            # Increment update count for this state key
            self.update_count[sample.state_key] += 1
            # Get action indices
            idxs = [a.encode() for a in sample.valid_actions]
            # Get action probabilities
            probs = self.actions_to_probs(sample.valid_actions, sample.state_key, sample.state_vector_encoding)
            # Compute gradients
            grads = np.array([(sample.action == a) - p for a, p in zip(sample.valid_actions, probs)])
            # Update logits
            self.lookup[sample.state_key][idxs] += self.learning_rate * grads * sample.reward_to_go

    def to_json(self) -> dict:
        """Convert model to JSON-serializable dictionary.

        Returns:
            Dictionary containing learning_rate, lookup, update_count, and required model metadata.
        """
        return {
            "learning_rate": self.learning_rate,
            "lookup": {state_key: logits.tolist() for state_key, logits in self.lookup.items()},
            "update_count": dict(self.update_count),
            "target_player_index": self.target_player_index,
            "game_config": self.game_config.to_json(),
            "abstraction_cls": self.abstraction_cls.__name__,
            "resolver_cls": self.resolver_cls.__name__,
            "model_cls": self.__class__.__name__,
        }

    @classmethod
    def from_json(cls, data: dict) -> "TabularReinforceModel":
        """Create model from JSON dictionary.

        Args:
            data: Dictionary containing learning_rate, lookup, and required model metadata.

        Returns:
            New TabularReinforceModel instance.

        Raises:
            ValueError: If any required field is missing.
        """
        target_player_index = data["target_player_index"]
        game_config = GameConfig.from_json(data["game_config"])
        abstraction_cls = ABSTRACTION_NAME_TO_CLS[data["abstraction_cls"]]
        resolver_cls = RESOLVER_NAME_TO_CLS[data["resolver_cls"]]
        # Create model
        model = cls(
            learning_rate=data["learning_rate"],
            target_player_index=target_player_index,
            game_config=game_config,
            abstraction_cls=abstraction_cls,
            resolver_cls=resolver_cls,
        )

        # Populate lookup from serialized data, converting lists back to numpy arrays
        for state_key, logits_list in data["lookup"].items():
            model.lookup[state_key] = np.array(logits_list, dtype=float)

        # Populate update_count from serialized data
        model.update_count = Counter(data["update_count"])

        return model


class NeuralNetworkReinforceModel(BaseReinforceModel, Serializable):
    """Neural network reinforcement learning model."""

    def __init__(
        self,
        learning_rate: float,
        target_player_index: int,
        game_config: GameConfig,
        abstraction_cls: type[BaseStateAbstraction],
        resolver_cls: type[BaseActionResolver],
        hidden_layer_sizes: list[int],
        random_seed: int,
        weight_decay: float,
        epochs_per_update: int,
    ):
        super().__init__(learning_rate, target_player_index, game_config, abstraction_cls, resolver_cls)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.random_seed = random_seed
        self.weight_decay = weight_decay
        self.epochs_per_update = epochs_per_update
        # Compute updated layer sizes
        output_layer_size = len(AbstractAction)
        input_layer_size = abstraction_cls.vector_encoding_length()
        layer_sizes = [input_layer_size] + hidden_layer_sizes + [output_layer_size]
        # Store network parameters
        self.params = init_network_params(layer_sizes, random.key(random_seed))
        # Store optimizer state
        self.optimizer = (
            optax.adamw(learning_rate, weight_decay=weight_decay) if weight_decay > 0 else optax.adam(learning_rate)
        )
        self.optimizer_state = self.optimizer.init(self.params)

    def _parse_batch(self, batch: list[TrainingSample]) -> tuple[Array, Array, Array, Array, list[str]]:
        """Parse a batch of training samples into JAX arrays, and a list of state keys."""
        encodings, masks, action_idxs, rewards, state_keys = [], [], [], [], []
        for sample in batch:
            # Append state vector encoding
            encodings.append(sample.state_vector_encoding)
            # Compute binary mask for valid actions
            idxs = [a.encode() for a in sample.valid_actions]
            # Compute index of action take, w.r.t. valid action indices
            action_idxs.append(sample.action.encode())
            # Build mask for valid actions
            mask = jnp.zeros(len(AbstractAction)).at[jnp.array(idxs)].set(1)
            masks.append(mask)
            # Append reward to go
            rewards.append(sample.reward_to_go)
            # Append state key
            state_keys.append(sample.state_key)
        # Compose JAX arrays
        inp = jnp.stack(encodings)
        mask = jnp.stack(masks).astype(bool)
        action_idxs = jnp.array(action_idxs)
        rewards = jnp.array(rewards)
        return inp, mask, action_idxs, rewards, state_keys

    @staticmethod
    @jax.jit
    def loss_fn(
        params: list[tuple[Array, Array]],
        input: Array,
        mask: Array,
        action_idxs: Array,
        rewards: Array,
    ) -> Array:
        """
        Compute the loss for a batch of training samples.
        """
        # Compute logits
        logits = batch_compute_logits(params, input)
        valid_logits = logits * mask + (1 - mask) * -1e15
        # Compute log probabilities
        valid_log_probs = valid_logits - logsumexp(valid_logits, axis=-1, keepdims=True)
        # Get log probability of chosen actions
        action_log_probs = jnp.take_along_axis(valid_log_probs, action_idxs[:, None], axis=-1)
        # Compute loss
        loss = -(action_log_probs * rewards[:, None]).mean()
        return loss

    def actions_to_probs(
        self, actions: list[AbstractAction], state_key: str, state_vector_encoding: np.ndarray
    ) -> np.ndarray:
        # Get action indices
        idxs = [a.encode() for a in actions]
        # Compute logits
        inp = jnp.array(state_vector_encoding).reshape(1, -1)
        logits = batch_compute_logits(self.params, inp)
        # Compute probabilities using masked logits
        probs = jax.nn.softmax(logits[:, idxs], axis=-1)
        # Cast to numpy array
        probs = np.asarray(probs, dtype=np.float64).ravel()
        # Re-normalize probabilities to avoid precision issues
        probs = probs / probs.sum()
        # Return probabilities
        return probs

    def select(self, actions: list[WrappedAction], state: GameState) -> WrappedAction:
        # Get abstract actions
        abstract_actions = [a.abstract_action for a in actions]
        # Compute action probabilities
        probs = self.actions_to_probs(abstract_actions, state.symmetric_key, state.vector_encoding())
        # Sample action
        return Policy(actions=actions, probs=probs.tolist()).sample()

    def update(self, batch: list[TrainingSample]) -> float:
        """Update the reinforcement learning model."""
        # Parse batch
        inp, mask, action_idxs, rewards, state_keys = self._parse_batch(batch)
        for _ in range(self.epochs_per_update):
            # Compute loss and gradients
            loss, grads = jax.value_and_grad(self.loss_fn)(self.params, inp, mask, action_idxs, rewards)
            updates, self.optimizer_state = self.optimizer.update(grads, self.optimizer_state, self.params)
            self.params = optax.apply_updates(self.params, updates)

        # Update update count for each state key
        self.update_count.update(state_keys)

        return float(loss)

    def to_json(self) -> dict:
        """Convert model to JSON-serializable dictionary.

        Returns:
            Dictionary containing learning_rate, lookup, update_count, and required model metadata.
        """
        return {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "epochs_per_update": self.epochs_per_update,
            "update_count": dict(self.update_count),
            "target_player_index": self.target_player_index,
            "game_config": self.game_config.to_json(),
            "abstraction_cls": self.abstraction_cls.__name__,
            "resolver_cls": self.resolver_cls.__name__,
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "random_seed": self.random_seed,
            "params": [{"weights": w.tolist(), "biases": b.tolist()} for w, b in self.params],
            "model_cls": self.__class__.__name__,
        }

    @classmethod
    def from_json(cls, data: dict) -> "NeuralNetworkReinforceModel":
        """Create model from JSON dictionary.

        Args:
            data: Dictionary containing learning_rate, lookup, and required model metadata.

        Returns:
            New NeuralNetworkReinforceModel instance.

        Raises:
            ValueError: If any required field is missing.
        """
        target_player_index = data["target_player_index"]
        game_config = GameConfig.from_json(data["game_config"])
        abstraction_cls = ABSTRACTION_NAME_TO_CLS[data["abstraction_cls"]]
        resolver_cls = RESOLVER_NAME_TO_CLS[data["resolver_cls"]]
        # Create model
        model = cls(
            learning_rate=data["learning_rate"],
            weight_decay=data["weight_decay"],
            epochs_per_update=data["epochs_per_update"],
            target_player_index=target_player_index,
            game_config=game_config,
            abstraction_cls=abstraction_cls,
            resolver_cls=resolver_cls,
            hidden_layer_sizes=data["hidden_layer_sizes"],
            random_seed=data["random_seed"],
        )

        # Rebuild params from serialized data
        model.params = [
            (jnp.array(param_dict["weights"], dtype=jnp.float32), jnp.array(param_dict["biases"], dtype=jnp.float32))
            for param_dict in data["params"]
        ]

        # Populate update_count from serialized data
        model.update_count = Counter(data["update_count"])

        return model
