from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
from jax import Array, random
import jax.numpy as jnp
import numpy as np
import optax

from game.action import AbstractAction, BaseActionResolver, WrappedAction
from game.config import GameConfig
from game.game import PlayerSpec
from game.state import ABSTRACTION_NAME_TO_CLS, RESOLVER_NAME_TO_CLS, BaseStateAbstraction, GameState
from game.util import Serializable
from models.checkpoint import load_checkpoint_data
from models.selector import BaseActionSelector
from models.types import Policy, Trajectory


if TYPE_CHECKING:
    from game.game import PlayerSpec


@dataclass(frozen=True)
class TrainingSample:
    """One training sample."""

    state_key: str
    state_vector_encoding: np.ndarray
    future_state_vector_encoding: np.ndarray
    valid_actions: list[AbstractAction]
    action: AbstractAction
    rewards: list[float]


def he_init(m: int, n: int, key: Array) -> tuple[Array, Array]:
    """He initialization for ReLU activations."""
    w_key, _ = random.split(key)
    # He initialization: std = sqrt(2 / fan_in)
    std = jnp.sqrt(2.0 / m)
    w = std * random.normal(w_key, (n, m))
    b = jnp.zeros((n,))
    return w, b


def init_network_params(sizes: list[int], key: Array) -> list[tuple[Array, Array]]:
    """Initialize network parameters with He initialization for hidden layers."""
    keys = random.split(key, len(sizes))
    params = []
    for i, (m, n, k) in enumerate(zip(sizes[:-1], sizes[1:], keys)):
        if i < len(sizes) - 2:  # Hidden layers
            params.append(he_init(m, n, k))
        else:  # Output layer
            w_key, _ = random.split(k)
            std = jnp.sqrt(1.0 / m)
            w = std * random.normal(w_key, (n, m))
            b = jnp.zeros((n,))
            params.append((w, b))
    return params


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


class PolicyAndValueNetwork(Serializable):
    """Policy and value network."""

    def __init__(
        self,
        learning_rate: float,
        gamma: float,
        lmbda: float,
        value_loss_weight: float,
        target_player_index: int,
        game_config: GameConfig,
        abstraction_cls: type[BaseStateAbstraction],
        resolver_cls: type[BaseActionResolver],
        hidden_layer_sizes: list[int],
        random_seed: int,
        entropy_coef: float = 0.02,
        clip_epsilon: float = 0.2,
        weight_decay: float = 1e-5,
        gradient_clip: float = 1.0,
        epochs_per_update: int = 10,
    ):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lmbda = lmbda
        self.value_loss_weight = value_loss_weight
        self.entropy_coef = entropy_coef
        self.clip_epsilon = clip_epsilon
        self.weight_decay = weight_decay
        self.gradient_clip = gradient_clip
        self.target_player_index = target_player_index
        self.game_config = game_config
        self.abstraction_cls = abstraction_cls
        self.resolver_cls = resolver_cls
        self.update_count: Counter[str] = Counter()
        self.hidden_layer_sizes = hidden_layer_sizes
        self.random_seed = random_seed
        self.epochs_per_update = epochs_per_update
        # Compute updated layer sizes
        output_layer_size = len(AbstractAction) + 1  # +1 for value prediction
        input_layer_size = abstraction_cls.vector_encoding_length()
        layer_sizes = [input_layer_size] + hidden_layer_sizes + [output_layer_size]
        # Store network parameters
        self.params = init_network_params(layer_sizes, random.key(random_seed))
        # Store optimizer state with weight decay and gradient clipping
        optimizer = optax.chain(
            optax.clip_by_global_norm(gradient_clip) if gradient_clip > 0 else optax.identity(),
            optax.adamw(learning_rate, weight_decay=weight_decay) if weight_decay > 0 else optax.adam(learning_rate),
        )
        self.tx = optimizer
        self.opt_state = self.tx.init(self.params)
        self.base_learning_rate = learning_rate

    def copy(self) -> "PolicyAndValueNetwork":
        """Create a deep copy of this model for use as a snapshot in self-play.

        Returns:
            A new PolicyAndValueNetwork instance with the same state as this model.
        """
        return self.from_json(self.to_json())

    def create_selector(self) -> BaseActionSelector:
        """Create an action selector for this GAE model.

        Returns:
            BaseActionSelector instance configured for this model.
        """
        from models.gae.selector import GAEActionSelector

        return GAEActionSelector(model=self)

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

    @classmethod
    def from_checkpoint(cls, load_path: str) -> PolicyAndValueNetwork:
        """Load model from checkpoint file (local or GCS).

        Args:
            load_path: Path to checkpoint file (local file path or GCS gs:// path).

        Returns:
            PolicyAndValueNetwork instance loaded from checkpoint.
        """
        data = load_checkpoint_data(load_path)
        return cls.from_json(data)

    def _parse_trajectories(
        self, trajectories: list[Trajectory]
    ) -> tuple[Array, Array, Array, Array, Array, list[str]]:
        """Parse a batch of training samples into JAX arrays, and a list of state keys."""
        encodings, action_masks, action_idxs, is_done, rewards, state_keys = [], [], [], [], [], []
        for trajectory in trajectories:
            for i, action in enumerate(trajectory.model_actions):
                # Append state vector encoding
                encodings.append(action.state_vector_encoding)
                # Compute binary mask for valid actions
                idxs = [a.encode() for a in action.valid_actions]
                # Compute index of action take, w.r.t. valid action indices
                action_idxs.append(action.action.encode())
                # Build mask for valid actions
                mask = jnp.zeros(len(AbstractAction)).at[jnp.array(idxs)].set(1)
                action_masks.append(mask)
                # Append state key
                state_keys.append(action.state_key)
                # Append whether this is the end state
                done = i == len(trajectory.model_actions) - 1
                is_done.append(done)
                # Append reward
                rewards.append(action.reward + done * trajectory.reward)
        # Compose JAX arrays
        inp = jnp.stack(encodings)
        action_mask = jnp.stack(action_masks).astype(bool)
        action_idxs = jnp.array(action_idxs)
        is_done = jnp.array(is_done)
        rewards = jnp.array(rewards)
        return inp, action_mask, action_idxs, is_done, rewards, state_keys

    @staticmethod
    @jax.jit
    def loss_fn(
        params: list[tuple[Array, Array]],
        inp: Array,
        action_mask: Array,
        action_idxs: Array,
        rewards: Array,
        is_done: Array,
        gamma: float,
        lmbda: float,
        value_loss_weight: float,
        entropy_coef: float,
        clip_epsilon: float,
        old_log_probs: Array | None,
    ) -> tuple[Array, dict[str, Array]]:
        """
        Compute the loss for a batch of training samples with PPO-style clipping and entropy regularization.

        Returns:
            Tuple of (total_loss, aux_dict) where aux_dict contains diagnostic metrics.
        """
        # Compute logits
        logits = batch_compute_logits(params, inp)

        # Separate policy logits from value logits
        policy_logits = logits[:, :-1]
        value_logits = logits[:, -1]

        # Estimate values for each state in the trajectory
        values = jax.nn.sigmoid(value_logits)

        # Compute value estimates for "next" state
        next_values = jnp.concatenate([values[1:], jnp.zeros(1)])
        next_values = jnp.where(is_done, 0.0, next_values)

        # Compute deltas
        deltas = rewards + gamma * next_values - values

        # Define step function for computing advantages
        def step(adv_next: jnp.ndarray, input: tuple[Array, Array]) -> tuple[jnp.ndarray, jnp.ndarray]:
            delta_t, done_t = input
            adv_t = delta_t + gamma * lmbda * (1 - done_t) * adv_next
            return adv_t, adv_t

        # Scan over step function to compute advantages
        _, adv_rev = jax.lax.scan(step, jnp.array([0.0]), (deltas[::-1], is_done[::-1]))

        # Reverse the reversed advantages
        advantages = adv_rev[::-1].ravel()

        # Compute returns (for value learning): returns = advantages + values
        returns = jax.lax.stop_gradient(advantages + values)

        # Normalize advantages for policy learning (z-score) - but use raw advantages for value target
        norm_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-15)

        # Compute log probabilities, setting the logits of invalid actions to -1e15
        masked_logits = policy_logits * action_mask + (1 - action_mask) * -1e15
        log_probs = jax.nn.log_softmax(masked_logits, axis=-1)
        probs = jax.nn.softmax(masked_logits, axis=-1)

        # Get log probability of chosen actions
        action_logp = jnp.take_along_axis(log_probs, action_idxs[:, None], axis=-1).ravel()

        # Compute entropy (for exploration)
        entropy = -(probs * log_probs).sum(axis=-1).mean()

        # Compute policy loss with PPO-style clipping
        if old_log_probs is not None:
            # PPO clipped loss
            ratio = jnp.exp(action_logp - old_log_probs)
            clipped_ratio = jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
            policy_loss = -jnp.minimum(ratio * norm_advantages, clipped_ratio * norm_advantages).mean()
        else:
            # Vanilla policy gradient (first update)
            policy_loss = -(action_logp * jax.lax.stop_gradient(norm_advantages)).mean()

        # Compute clipped value loss
        value_pred_clipped = values + jnp.clip(values - returns, -clip_epsilon, clip_epsilon)
        value_loss = 0.5 * jnp.maximum((values - returns) ** 2, (value_pred_clipped - returns) ** 2).mean()

        # Compute total loss: policy loss - entropy bonus + value loss
        total_loss = policy_loss - entropy_coef * entropy + value_loss_weight * value_loss

        # Return loss and auxiliary diagnostics
        aux = {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_loss": -entropy_coef * entropy,
            "mean_advantage": advantages.mean(),
            "mean_value": values.mean(),
        }
        return total_loss, aux

    @staticmethod
    @jax.jit
    def value_loss_fn(
        logits: Array,
        future_logits: Array,
        rewards: Array,
    ) -> Array:
        """
        Compute the loss for a batch of training samples.
        """
        # Compute value predictions
        preds = jax.nn.sigmoid(logits)
        # Compute log probabilities
        future_preds = jax.nn.sigmoid(future_logits)
        # Compute target
        targets = rewards + future_preds
        # Compute loss
        loss = jnp.mean((preds - targets) ** 2)
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

    def update(
        self,
        trajectories: list[Trajectory],
        game_idx: int = 0,
        entropy_decay_games: int = 20000,
        entropy_decay_min: float = 0.003,
        lr_decay_games: int = 30000,
        lr_decay_min: float = 0.2,
    ) -> dict[str, float]:
        """Update the reinforcement learning model.

        Args:
            trajectories: List of trajectories to train on.
            game_idx: Current game index for entropy/learning rate decay.
            entropy_decay_games: Number of games over which entropy decays.
            entropy_decay_min: Minimum entropy coefficient after decay.
            lr_decay_games: Number of games over which learning rate decays.
            lr_decay_min: Minimum learning rate factor after decay.
        """
        # Parse trajectories
        inp, action_mask, action_idxs, is_done, rewards, state_keys = self._parse_trajectories(trajectories)

        # Decay entropy coefficient over time
        # entropy_coef = initial * (min/initial)^(game_idx/decay_games)
        entropy_decay_factor = (entropy_decay_min / self.entropy_coef) ** (game_idx / float(entropy_decay_games))
        current_entropy_coef = max(entropy_decay_min, self.entropy_coef * entropy_decay_factor)

        # Compute old log probabilities for PPO clipping (before update)
        logits = batch_compute_logits(self.params, inp)
        policy_logits = logits[:, :-1]
        masked_logits = policy_logits * action_mask + (1 - action_mask) * -1e15
        old_log_probs = jax.nn.log_softmax(masked_logits, axis=-1)
        old_log_probs = jnp.take_along_axis(old_log_probs, action_idxs[:, None], axis=-1).ravel()

        # Apply learning rate decay: decay from base_learning_rate to lr_decay_min * base_learning_rate
        # Less aggressive decay to maintain learning capacity
        lr_decay_factor = max(lr_decay_min, 1.0 - (1.0 - lr_decay_min) * (game_idx / float(lr_decay_games)))
        current_lr = self.base_learning_rate * lr_decay_factor

        # Recreate optimizer with decayed learning rate if it changed significantly
        if game_idx % 2000 == 0 and abs(lr_decay_factor - 1.0) > 0.05:
            optimizer = optax.chain(
                optax.clip_by_global_norm(self.gradient_clip) if self.gradient_clip > 0 else optax.identity(),
                optax.adamw(current_lr, weight_decay=self.weight_decay)
                if self.weight_decay > 0
                else optax.adam(current_lr),
            )
            # Reinitialize optimizer state with new learning rate
            self.tx = optimizer
            self.opt_state = self.tx.init(self.params)

        # Compute loss and gradients (with auxiliary values)
        for _ in range(self.epochs_per_update):
            (loss, aux), grads = jax.value_and_grad(self.loss_fn, has_aux=True)(
                self.params,
                inp,
                action_mask,
                action_idxs,
                rewards,
                is_done,
                self.gamma,
                self.lmbda,
                self.value_loss_weight,
                current_entropy_coef,
                self.clip_epsilon,
                old_log_probs,
            )
            updates, self.opt_state = self.tx.update(grads, self.opt_state, self.params)
            self.params = optax.apply_updates(self.params, updates)

        # Update update count for each state key
        self.update_count.update(state_keys)

        # Return loss and diagnostics
        diagnostics = {k: float(v) for k, v in aux.items()}
        diagnostics["total_loss"] = float(loss)
        return diagnostics

    def to_json(self) -> dict:
        """Convert model to JSON-serializable dictionary.

        Returns:
            Dictionary containing learning_rate, lookup, update_count, and required model metadata.
        """
        return {
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "lmbda": self.lmbda,
            "value_loss_weight": self.value_loss_weight,
            "entropy_coef": self.entropy_coef,
            "clip_epsilon": self.clip_epsilon,
            "weight_decay": self.weight_decay,
            "gradient_clip": self.gradient_clip,
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
    def from_json(cls, data: dict) -> "PolicyAndValueNetwork":
        """Create model from JSON dictionary.

        Args:
            data: Dictionary containing learning_rate, lookup, and required model metadata.

        Returns:
            New PolicyAndValueNetwork instance.

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
            gamma=data["gamma"],
            lmbda=data["lmbda"],
            value_loss_weight=data["value_loss_weight"],
            target_player_index=target_player_index,
            game_config=game_config,
            abstraction_cls=abstraction_cls,
            resolver_cls=resolver_cls,
            hidden_layer_sizes=data["hidden_layer_sizes"],
            random_seed=data["random_seed"],
            entropy_coef=data["entropy_coef"],
            clip_epsilon=data["clip_epsilon"],
            weight_decay=data["weight_decay"],
            gradient_clip=data["gradient_clip"],
            epochs_per_update=data["epochs_per_update"],
        )

        # Rebuild params from serialized data
        model.params = [
            (jnp.array(param_dict["weights"], dtype=jnp.float32), jnp.array(param_dict["biases"], dtype=jnp.float32))
            for param_dict in data["params"]
        ]

        # Reinitialize optimizer state with loaded params (they were initialized with random params in __init__)
        model.opt_state = model.tx.init(model.params)

        # Populate update_count from serialized data
        model.update_count = Counter(data["update_count"])

        return model
