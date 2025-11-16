import numpy as np

from game.config import GameConfig
from game.constants import NUM_PLAYERS
from game.game import Game, PlayerSpec
from models.cfr.selector import CFRActionSelector
from models.gae.selector import GAEActionSelector
from models.reinforce.selector import ReinforceActionSelector
from models.types import ModelAction, Trajectory


def softmax(logits: np.ndarray) -> np.ndarray:
    exponentiated = np.exp(logits)
    denom = np.sum(exponentiated)
    return exponentiated / denom


def play_games(
    game_config: GameConfig,
    player_specs: dict[int, PlayerSpec],
    player2selector: dict[int, CFRActionSelector | ReinforceActionSelector | GAEActionSelector],
    target_player_index: int,
    num_games: int,
    max_turns_per_game: int,
    random_seed_base: int,
) -> list[Trajectory]:
    """Play multiple games and collect rewards and trajectories.

    Args:
        game_config: Game configuration.
        player_specs: Player specifications dict.
        player2selector: Mapping of player index to selector.
        target_player_index: Index of target player for reward computation.
        num_games: Number of games to play.
        max_turns_per_game: Maximum turns per game.
        random_seed_base: Base random seed (will be incremented per game).

    Returns:
        Tuple of (rewards list, trajectories list).
    """
    trajectories: list[Trajectory] = []

    for i in range(num_games):
        game = Game(
            config=game_config,
            player_specs=player_specs,
            init_player_index=i % NUM_PLAYERS,
            random_seed=random_seed_base + i,
        )

        trajectory = produce_trajectory(
            game=game,
            player2selector=player2selector,
            target_player_index=target_player_index,
            max_turns_per_game=max_turns_per_game,
            argmax=True,
        )

        trajectories.append(trajectory)

        # Reset selectors after each game
        for selector in player2selector.values():
            selector.reset()

    return trajectories


def produce_trajectory(
    game: Game,
    player2selector: dict[int, CFRActionSelector | ReinforceActionSelector | GAEActionSelector],
    target_player_index: int,
    max_turns_per_game: int,
    argmax: bool = False,
) -> Trajectory:
    """Produce a trajectory for a single game."""
    # Define list of model actions
    model_actions: list[ModelAction] = []

    # Collect model actions
    while not game.over and game.turn_state.turn_idx < max_turns_per_game:
        # Get player actions
        actions = game.state.get_player_actions()
        # Get action selector
        selector = player2selector[game.player.index]
        # Select action
        wrapped_action = selector.select(
            actions=actions,
            state=game.state,
            argmax=argmax,
        )
        # If target player, record model action
        if game.player.index == target_player_index:
            # Add model action to actions. Use symmetric_key (no player index) so the model can be shared between players
            model_actions.append(
                ModelAction(
                    state_key=game.state.symmetric_key,
                    state_vector_encoding=game.state.vector_encoding(),
                    turn_idx=game.turn_state.turn_idx,
                    streak_idx=game.turn_state.streak_idx,
                    streaking_player_idx=game.turn_state.streaking_player_idx,
                    valid_actions=[a.abstract_action for a in actions],
                    action=wrapped_action.abstract_action,
                    reward=0,
                )
            )
        # Take selected action
        game.step(selected_action=wrapped_action.action)

    # Define reward
    reward = 1 if game.winner and game.winner.index == target_player_index else 0 if game.winner else 0.5

    # Return trajectory
    return Trajectory(model_actions=model_actions, reward=reward)
