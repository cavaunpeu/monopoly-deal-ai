from collections import Counter
from dataclasses import dataclass
from math import comb
import random
from typing import cast

import numpy as np

from game.action import (
    BaseAction,
    BaseActionResolver,
    CardAction,
    DrawAction,
    GameAction,
    ResponseGameAction,
    SelectedAction,
)
from game.cards import Card
from game.config import GameConfig
from game.constants import NUM_PLAYERS
from game.deck import Deck
from game.pile import BasePile, Discard, Hand, Pile
from game.player import Player
from game.state import BaseStateAbstraction, GameState, OpponentState, PlayerState, ResponseContext, TurnState


@dataclass(frozen=True)
class PlayerSpec:
    abstraction_cls: type[BaseStateAbstraction]
    resolver_cls: type[BaseActionResolver]


class Game:
    """Main game class that orchestrates Monopoly Deal gameplay.

    This class manages the game state, player actions, and game progression
    for the modified Monopoly Deal card game.
    """

    def __init__(
        self,
        config: GameConfig,
        init_player_index: int,
        player_specs: dict[int, PlayerSpec],
        verbose: bool = False,
        random_seed: int = 0,
    ):
        """Initialize a new game instance.

        Args:
            config: Game configuration defining rules and parameters.
            init_player_index: Index of the player who starts the game (0 or 1).
            player_specs: Dictionary mapping player indices (0, 1) to PlayerSpec objects.
            verbose: Whether to enable verbose logging.
            random_seed: Seed for random number generation.
        """
        # Validate player specs
        assert set(player_specs.keys()) == set(range(NUM_PLAYERS)), (
            f"player_specs must have keys {list(range(NUM_PLAYERS))}"
        )

        # Define players
        self.players = [Player(index=i) for i in range(NUM_PLAYERS)]

        # Map player to player spec
        self.player2spec = {player: player_specs[player.index] for player in self.players}

        # Set global random seed to match game's random seed for consistency
        random.seed(random_seed)

        # Set game config
        self.config = config

        # Initialize deck and discard piles
        self.deck = Deck.build(
            game_config=config,
            random_seed=random_seed,
        )
        self.discard = Discard([])

        # Initialize turn state
        self.turn_state = TurnState(
            turn_idx=0,
            streak_idx=0,
            streaking_player_idx=init_player_index,
        )

        # Initialize played cards history
        self.selected_actions: list[SelectedAction] = []

        # Initialize buffers for draw probabilities
        self.draw_probs = self.init_player_buffers()

        # Set flags
        self.verbose = verbose

        # Set random seed
        self.random_seed = random_seed

        # Store the original starting player index
        self.init_player_index = init_player_index

        # Draw an initial hand for both players
        self.draw_cards(self.players)

        # Draw cards for the current player's turn
        self.draw_cards([self.player])

    def init_player_buffers(self):
        """Initializes empty buffers for all players."""
        return [[] for _ in range(len(self.players))]

    def get_pile_map(self, player: Player) -> dict[Pile, BasePile]:
        opponent = self.players[1 - player.index]
        return {
            Pile.HAND: player.hand,
            Pile.PROPERTY: player.properties,
            Pile.CASH: player.cash,
            Pile.PLAYED_CARDS: player.played_cards,
            Pile.DISCARD: self.discard,
            Pile.OPPONENT_CASH: opponent.cash,
            Pile.DECK: self.deck,
        }

    def _apply_action(self, action: BaseAction, player: Player):
        pile_map = self.get_pile_map(player)
        if action.plays_card:
            # Action must be a card action
            action = cast(CardAction, action)
            # Remove card from source pile
            pile_map[action.src].remove(action.card)
            # Add card to destination pile
            pile_map[action.dst].add(action.card)
            # Add card to played cards pile
            pile_map[Pile.PLAYED_CARDS].add(action.card)
        elif action.is_draw:
            # Action must be a draw action
            action = cast(DrawAction, action)
            # Remove card from source pile
            pile_map[Pile.DECK].remove(action.card)
            # Add card to destination pile
            pile_map[Pile.HAND].add(action.card)

        # Add non-draw actions to selected actions history
        if not action.is_draw:
            self.selected_actions.append(
                SelectedAction(
                    turn_idx=self.turn_state.turn_idx,
                    streak_idx=self.turn_state.streak_idx,
                    player_idx=player.index,
                    action=action,
                )
            )

    @property
    def chance_probs(self):
        """Returns a list of joint probabilities for all chance events."""
        return [prob for player in self.players for prob in self.draw_probs[player.index]]

    def draw_cards(self, players: list[Player]):
        """Initializes a turn for the current player."""
        for player in players:
            draws = []
            num_cards = self.config.initial_hand_size if not player.hand else self.config.new_cards_per_turn
            for _ in range(num_cards):
                if len(self.deck) > 0:
                    # Choose card from deck
                    card = self.deck.pick()
                    # Apply draw action
                    self._apply_action(DrawAction(card), player)
                    # Append to draws
                    draws.append(card)
            # Compute joint probability of draw
            if draws:
                deck = self.deck
                draw_counter = Counter(draws)
                # Compute numerator terms
                numer_terms = []
                for card, count in draw_counter.items():
                    init_count = deck[card] + count
                    numer_terms.append(comb(init_count, count))
                # Compute denominator term
                denom = comb(len(self.deck) + len(draws), len(draws))
                # Compute joint probability
                joint_prob = np.prod(numer_terms) / denom
                # Store draw event
                self.draw_probs[player.index].append(joint_prob)

    @property
    def player_abstraction_cls(self) -> type[BaseStateAbstraction]:
        return self.player2spec[self.player].abstraction_cls

    @property
    def player_resolver_cls(self) -> type[BaseActionResolver]:
        return self.player2spec[self.player].resolver_cls

    @property
    def state(self):
        """Returns the current game state."""
        return GameState(
            turn=self.turn_state,
            player=self.player_state,
            opponent=self.opponent_state,
            discard=self.discard,
            config=self.config,
            abstraction_cls=self.player_abstraction_cls,
            resolver_cls=self.player_resolver_cls,
            random_seed=self.random_seed,
        )

    @property
    def player(self) -> Player:
        """Returns the current player."""
        return self.players[self.turn_state.acting_player_index]

    @property
    def opponent(self) -> Player:
        """Returns the opponent of the current player."""
        return self.players[self.turn_state.other_player_index]

    @property
    def streaking_player_state(self) -> PlayerState:
        streaking_player_idx = self.turn_state.streaking_player_idx
        player = self.players[streaking_player_idx]
        return PlayerState(
            hand=player.hand,
            properties=player.properties,
            cash=player.cash,
        )

    @property
    def player_state(self) -> PlayerState:
        return PlayerState(
            hand=self.player.hand,
            properties=self.player.properties,
            cash=self.player.cash,
        )

    @property
    def opponent_state(self) -> OpponentState:
        return OpponentState(
            properties=self.opponent.properties,
            cash=self.opponent.cash,
        )

    @property
    def acting_player(self) -> Player:
        """Returns the current player."""
        return self.players[self.turn_state.acting_player_index]

    @property
    def other_player(self) -> Player:
        """Returns the opponent of the current player."""
        return self.players[self.turn_state.other_player_index]

    @property
    def opponent_hand_size(self):
        return len(self.opponent.hand)

    @property
    def winner(self):
        """Returns the winner of the game."""
        for player in self.players:
            if player.num_complete_property_sets >= self.config.required_property_sets:
                return player

    @property
    def over(self):
        """Check if the game is over (e.g., a player has the required number of property sets)."""
        if self.winner is not None:
            return True
        return self.exhausted

    @property
    def exhausted(self):
        """Check if the deck is empty and all players have no cards."""
        return len(self.deck) == 0 and len(self.player.hand) == 0 and len(self.opponent.hand) == 0

    def set_player_hand(self, player_idx: int, fictional_hand: list[Card] | tuple[Card, ...]):
        """Sets players's hand to the given fictional hand."""
        player = self.players[player_idx]
        for card in player.hand:
            self.deck.add(card)
        for card in fictional_hand:
            self.deck.remove(card)
        player.hand = Hand(fictional_hand)

    def step(self, selected_action: BaseAction):
        """Execute a single game step with the given action.

        Args:
            selected_action: The action to execute for the current player.
        """
        # Apply selected action
        self._apply_action(selected_action, self.player)

        # Alias turn state
        ts = self.turn_state

        # Define should_complete_streak function
        def should_complete_streak(streak_idx: int):
            return streak_idx + 1 >= self.config.max_consecutive_player_actions

        # Define complete_streak_and_draw function
        def complete_streak_and_draw():
            ts.complete_streak()
            self.draw_cards([self.player])

        # Is the opponent responding?
        if ts.responding:
            ctx = cast(ResponseContext, ts.response_ctx)
            # Add response to response context
            ctx.add_response(cast(ResponseGameAction, selected_action))
            # Is the response complete?
            response_complete = ctx.init_action_taken.response_def.response_complete(
                streaking_player_init_state=ctx.streaking_player_init_state,
                streaking_player_curr_state=self.streaking_player_state,
                response_actions_taken=ctx.response_actions_taken,
            )
            # Complete the response
            if response_complete:
                ts.complete_response()
                # Complete the streak
                if should_complete_streak(ts.streak_idx):
                    complete_streak_and_draw()
                else:
                    ts.increment_streak_idx()
                # Increment turn
                ts.increment_turn_idx()
        # Does the action require a response?
        elif selected_action.plays_card and cast(GameAction, selected_action).response_def.response_required(
            self.player_state
        ):
            # Set response context
            ts.set_response_ctx(
                ResponseContext(
                    streaking_player_init_state=self.state.player.clone(),
                    init_action_taken=cast(GameAction, selected_action),
                )
            )
        # If not responding and action doesn't play card (i.e. it's a pass), complete the streak
        elif not selected_action.plays_card:
            complete_streak_and_draw()
            ts.increment_turn_idx()
        # Otherwise, increment the streak and turn idxs
        elif should_complete_streak(ts.streak_idx):
            complete_streak_and_draw()
            ts.increment_turn_idx()
        else:
            ts.increment_streak_idx()
            ts.increment_turn_idx()

    def clone(self, flush_draw_probs: bool = False) -> "Game":
        """Create a deep copy of this game instance.

        Args:
            flush_draw_probs: Whether to clear draw probability history.

        Returns:
            A new Game instance with the same state.
        """
        players = [p.clone() for p in self.players]
        new_game = self.__class__.__new__(self.__class__)

        # Shallow-copy config fields
        new_game.players = players
        new_game.verbose = self.verbose
        new_game.config = self.config
        # Rebuild player2spec mapping with new player objects, but keep the same PlayerSpec objects
        new_game.player2spec = {
            new_player: self.player2spec[old_player] for new_player, old_player in zip(players, self.players)
        }
        new_game.random_seed = self.random_seed
        new_game.init_player_index = self.init_player_index

        # Copy dynamic state
        new_game.turn_state = self.turn_state.clone()

        # Clone the deck, discard piles, and played cards history
        new_game.deck = self.deck.clone()
        new_game.discard = self.discard.clone()
        new_game.selected_actions = list(self.selected_actions)

        # Handle draw probabilities
        if flush_draw_probs:
            new_game.draw_probs = new_game.init_player_buffers()
        else:
            new_game.draw_probs = [list(events) for events in self.draw_probs]

        return new_game
