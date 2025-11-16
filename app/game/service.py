from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import uuid

from sqlalchemy.orm import Session
import yaml

from app.api_models import serialize_card
from app.db_models import Game as GameModel
from app.db_models import SelectedAction as SelectedActionModel
from app.game.cache import GameCache
from game.action import BaseAction, SelectedAction, WrappedAction, decode_action
from game.cards import PropertyTypeCard
from game.config import GameConfig, GameConfigType
from game.game import Game, PlayerSpec
from game.state import ABSTRACTION_NAME_TO_CLS, RESOLVER_NAME_TO_CLS, GameState, OpponentState, PlayerState
from models.selector import BaseActionSelector
from models.types import SelectorModel


@dataclass(frozen=True)
class PublicPileSizes:
    deck: int
    discard: int


def player_specs_to_json(player_specs: dict[int, PlayerSpec]) -> dict[str, dict[str, str]]:
    """Convert a dict of PlayerSpec objects to JSON-serializable format for database."""
    return {
        str(i): {
            "abstraction_cls": spec.abstraction_cls.__name__,
            "resolver_cls": spec.resolver_cls.__name__,
        }
        for i, spec in player_specs.items()
    }


def json_to_player_specs(json_specs: dict[str, dict[str, str]]) -> dict[int, PlayerSpec]:
    """Convert JSON-serialized player specs (dict from database) to dict[int, PlayerSpec]."""
    return {
        int(player_idx): PlayerSpec(
            abstraction_cls=ABSTRACTION_NAME_TO_CLS[spec["abstraction_cls"]],
            resolver_cls=RESOLVER_NAME_TO_CLS[spec["resolver_cls"]],
        )
        for player_idx, spec in json_specs.items()
    }


class GameService:
    """Service for managing game sessions and AI interactions.

    This service handles game creation, state management, and AI action selection
    for the web interface.
    """

    _models_cache: dict = {}
    _loaded_models: dict[str, SelectorModel] = {}  # model_name -> SelectorModel (protocol)
    _load_errors: dict[str, str] = {}  # model_name -> error_message for failed loads

    def __init__(
        self,
        cache: GameCache,
        selector: BaseActionSelector,
        config: GameConfig,
        target_player_index: int,
        model_name: str,
        db: Optional[Session] = None,
    ) -> None:
        """Initialize the game service.

        Args:
            cache: Game cache for storing game states.
            selector: Action selector for AI decisions.
            config: Game configuration.
            target_player_index: Index of the AI player (0 or 1).
            model_name: Name of the model used to create this service.
            db: Optional database session.
        """
        self.cache = cache
        self.selector = selector
        self.config = config
        self.target_player_index = target_player_index
        self.model_name = model_name
        self._db = db

        # Set human player index to be opposite of AI's target player index
        # AI uses the model's target_player_index, human gets the other index
        self.HUMAN_PLAYER_INDEX = 1 - target_player_index

        # Load the model for this service (lazy loading)
        self._model = GameService._load_model(self.model_name)

    @property
    def db(self) -> Optional[Session]:
        """Get the database session."""
        return self._db

    @db.setter
    def db(self, value: Session) -> None:
        """Set the database session."""
        self._db = value

    def _get_config_name(self) -> str:
        """Get the config name for the current GameConfig by matching against GameConfigType enum values."""
        for config_type in GameConfigType:
            if config_type.value == self.config:
                return config_type.name
        raise ValueError("Custom GameConfig not supported - must use predefined GameConfigType values")

    def create_game(self, model_name: str | None = None) -> str:
        """Create a new game and return its ID.

        Args:
            model_name: Optional model name used to create this game.

        Returns:
            Unique game ID for the created game.
        """
        game_id = str(uuid.uuid4())
        # Convert game_id to deterministic integer seed
        random_seed = hash(game_id) % (2**31)  # Keep within int32 range

        # Both players use the same abstraction/resolver from the model
        spec = self._model.get_player_spec()
        player_specs = {0: spec, 1: spec}

        # Human always goes first
        game = Game(
            config=self.config,
            init_player_index=self.HUMAN_PLAYER_INDEX,
            player_specs=player_specs,
            random_seed=random_seed,
        )
        self.cache.set(game_id, game)

        # Save the game to the database if we have a database connection
        if self.db:
            game_record = GameModel(
                id=game_id,
                config_name=self._get_config_name(),
                player_specs=player_specs_to_json(player_specs),
                init_player_index=self.HUMAN_PLAYER_INDEX,
                random_seed=game.random_seed,
                model_name=model_name,
            )
            self.db.add(game_record)
            self.db.commit()

        return game_id

    def _recreate_game_from_db(self, game_id: str, db: Session) -> Game:
        # Query the game record
        record = db.query(GameModel).filter(GameModel.id == game_id).first()
        if record is None:
            raise ValueError(f"Game {game_id} not found in database.")

        # Convert player_specs from JSON to PlayerSpec objects
        player_specs = json_to_player_specs(record.player_specs)

        # Create a new game with the same config, initial player, random seed, and player specs
        game = Game(
            config=self.config,
            init_player_index=self.HUMAN_PLAYER_INDEX,
            player_specs=player_specs,
            random_seed=record.random_seed,
        )

        # Query all selected actions for this game, ordered by turn_idx, streak_idx, created_at
        selected_actions = (
            db.query(SelectedActionModel)
            .filter(SelectedActionModel.game_id == game_id)
            .order_by(
                SelectedActionModel.turn_idx,
                SelectedActionModel.streak_idx,
                SelectedActionModel.created_at,
            )
            .all()
        )

        # Replay all actions to reconstruct the game state
        for selected_action_record in selected_actions:
            action = decode_action(selected_action_record.action_id)
            game.step(action)

        # Return the recreated game
        return game

    def _get_game(self, game_id: str) -> Game:
        game = self.cache.get(game_id)
        if game is None:
            # Try to recreate the game from the database
            if self.db:
                try:
                    # Recreate the game from the database
                    game = self._recreate_game_from_db(game_id, self.db)

                    # Cache the recreated game
                    self.cache.set(game_id, game)

                    # Return the recreated game
                    return game

                except Exception as e:
                    # If database recreation fails, raise an error
                    raise ValueError(f"Game {game_id} could not be recreated from database: {e}")
            else:
                raise ValueError(f"No database connection found. Could not recreate game {game_id}.")

        # Return the cached game
        return game

    def get_game_state(self, game_id: str) -> GameState:
        """Get the current game state for a given game ID.

        Args:
            game_id: Unique identifier for the game.

        Returns:
            Current game state.
        """
        return self._get_game(game_id).state

    def _get_player_wrapped_actions(self, game_id: str) -> list[WrappedAction]:
        """Get wrapped actions for the current player.

        Args:
            game_id: Unique identifier for the game.

        Returns:
            List of wrapped actions available to the current player.
        """
        return self.get_game_state(game_id).get_player_actions(dedupe=False)

    def take_game_step(self, game_id: str, selected_action: BaseAction):
        game = self._get_game(game_id)

        # Get current game state to extract turn/streak/player info
        turn_state = game.turn_state
        turn_idx = turn_state.turn_idx
        streak_idx = turn_state.streak_idx
        player_idx = turn_state.acting_player_index

        # Insert selected action into database if we have a database connection
        if self.db:
            selected_action_record = SelectedActionModel(
                turn_idx=turn_idx,
                streak_idx=streak_idx,
                player_idx=player_idx,
                game_id=game_id,
                action_id=selected_action.encode(),
            )
            self.db.add(selected_action_record)
            self.db.commit()

        # Execute the game step
        game.step(selected_action)

    def get_player_actions(self, game_id: str) -> list[BaseAction]:
        wrapped_actions = self._get_player_wrapped_actions(game_id)
        return [action.action for action in wrapped_actions]

    def select_bot_action(self, game_id: str) -> BaseAction:
        game = self._get_game(game_id)
        wrapped_actions = self._get_player_wrapped_actions(game_id)
        return self.selector.select(wrapped_actions, game.state).action

    def bot_is_acting_player(self, game_id: str) -> bool:
        game = self._get_game(game_id)
        return game.turn_state.acting_player_index != self.HUMAN_PLAYER_INDEX

    def game_is_over(self, game_id: str) -> bool:
        game = self._get_game(game_id)
        return game.over

    def get_human_state(self, game_id: str) -> PlayerState:
        game = self._get_game(game_id)
        player = game.players[self.HUMAN_PLAYER_INDEX]
        return PlayerState(
            hand=player.hand,
            properties=player.properties,
            cash=player.cash,
        )

    def get_bot_state(self, game_id: str, show_hand: bool = False) -> tuple[OpponentState, int, list | None]:
        game = self._get_game(game_id)
        bot_player = game.players[1 - self.HUMAN_PLAYER_INDEX]  # Bot is the other player
        opponent_state = OpponentState(
            properties=bot_player.properties,
            cash=bot_player.cash,
        )
        hand_count = len(bot_player.hand)
        ai_hand = [serialize_card(c) for c in bot_player.hand.cards] if show_hand else None
        return opponent_state, hand_count, ai_hand

    def get_public_pile_sizes(self, game_id: str) -> PublicPileSizes:
        game = self._get_game(game_id)
        return PublicPileSizes(
            deck=len(game.deck),
            discard=len(game.discard),
        )

    def get_selected_actions_history(self, game_id: str) -> list[SelectedAction]:
        """Get the chronological history of all actions selected in the game"""
        game = self._get_game(game_id)
        return game.selected_actions

    def get_game_config(self) -> GameConfig:
        return self.config

    def get_property_types(self) -> list[PropertyTypeCard]:
        return list(PropertyTypeCard)

    @classmethod
    def _load_model_manifest(cls) -> None:
        """Load the model manifest from YAML file (without loading actual models)."""
        if cls._models_cache:
            return  # Already loaded

        current_dir = Path(__file__).parent
        models_file = current_dir.parent / "models.yaml"

        try:
            with open(models_file, "r") as f:
                manifest = yaml.safe_load(f)
                cls._models_cache = manifest.get("models", {})
        except FileNotFoundError:
            print("Warning: models.yaml not found, no models available")
            cls._models_cache = {}
        except yaml.YAMLError as e:
            print(f"Error parsing models.yaml: {e}")
            cls._models_cache = {}

    @classmethod
    def _load_model(cls, model_name: str) -> SelectorModel:
        """Load a single model on demand and cache it.

        Args:
            model_name: Name of the model to load.

        Returns:
            Loaded model instance.

        Raises:
            ValueError: If model_name is not in manifest or fails to load.
        """
        # Check if already loaded
        if model_name in cls._loaded_models:
            return cls._loaded_models[model_name]

        # Check if it previously failed to load
        if model_name in cls._load_errors:
            raise ValueError(f"Model '{model_name}' previously failed to load: {cls._load_errors[model_name]}")

        # Ensure manifest is loaded
        cls._load_model_manifest()

        # Check if model exists in manifest
        if model_name not in cls._models_cache:
            raise ValueError(
                f"Model '{model_name}' not found in models.yaml. Available models: {list(cls._models_cache.keys())}"
            )

        model_info = cls._models_cache[model_name]
        checkpoint_path = model_info["checkpoint_path"]
        model_type = model_info.get("model_type")

        if not model_type:
            error_msg = (
                f"Missing 'model_type' field for model '{model_name}' in models.yaml. "
                f"Required: 'cfr', 'reinforce-tabular', 'reinforce-neural', or 'actor-critic'"
            )
            cls._load_errors[model_name] = error_msg
            raise ValueError(error_msg)

        try:
            # Use match/case for loading models (only place we discriminate by type)
            match model_type:
                case "cfr":
                    from models.cfr.cfr import CFR

                    model = CFR.from_checkpoint(checkpoint_path)
                case "reinforce-tabular":
                    from models.reinforce.model import TabularReinforceModel

                    model = TabularReinforceModel.from_checkpoint(checkpoint_path)
                case "reinforce-neural":
                    from models.reinforce.model import NeuralNetworkReinforceModel

                    model = NeuralNetworkReinforceModel.from_checkpoint(checkpoint_path)
                case "actor-critic":
                    from models.gae.model import PolicyAndValueNetwork

                    model = PolicyAndValueNetwork.from_checkpoint(checkpoint_path)
                case _:
                    error_msg = (
                        f"Unknown model type '{model_type}' for model '{model_name}'. "
                        f"Must be 'cfr', 'reinforce-tabular', 'reinforce-neural', or 'actor-critic'"
                    )
                    cls._load_errors[model_name] = error_msg
                    raise ValueError(error_msg)

            # Cache the loaded model
            cls._loaded_models[model_name] = model
            print(f"Loaded {model_type} model {model_name} from {checkpoint_path}")
            return model
        except Exception as e:
            error_msg = str(e)
            cls._load_errors[model_name] = error_msg
            print(f"Failed to load model {model_name}: {error_msg}")
            raise

    @classmethod
    def get_model_manifest(cls) -> dict:
        """Return the cached model manifest (does not load actual models)."""
        cls._load_model_manifest()
        return {"models": cls._models_cache}

    @classmethod
    def get_loaded_models(cls) -> dict[str, SelectorModel]:
        """Get all currently loaded models (only models that have been accessed).

        Note: This does not load all models, only returns those that have been
        loaded on demand. Use get_model_manifest() to see all available models.
        """
        return cls._loaded_models.copy()

    @classmethod
    def get_load_errors(cls) -> dict[str, str]:
        """Get load errors for models that failed to load.

        Returns:
            Dictionary mapping model_name to error message for models that failed to load.
        """
        return cls._load_errors.copy()

    @classmethod
    def get_default_model_name(cls) -> str:
        """Get the default model name (first model in models.yaml).

        Returns:
            The name of the first model in models.yaml.

        Raises:
            RuntimeError: If no models are available.
        """
        cls._load_model_manifest()
        if not cls._models_cache:
            raise RuntimeError("No models available in models.yaml")
        return list(cls._models_cache.keys())[0]

    @classmethod
    def register_model_for_testing(cls, model_name: str, model: SelectorModel) -> None:
        """Register a model for testing purposes.

        This allows tests to register models without requiring models.yaml.
        Should only be used in test code.

        Args:
            model_name: Name to register the model under.
            model: The model instance to register.
        """
        cls._loaded_models[model_name] = model

    def _determine_winner(self, game) -> str | None:
        """Determine the winner if the game is over"""
        if not game.over:
            return None

        winner = game.winner
        if winner is None:
            return "tie"

        # Map the winner player index to human/bot
        if winner.index == self.HUMAN_PLAYER_INDEX:
            return "human"
        else:
            return "bot"

    def _build_response_info(self, turn_state) -> dict | None:
        """Build response information if the player is responding"""
        if not turn_state.responding:
            return None

        initiating_card = turn_state.response_ctx.init_action_taken.card

        # Determine players
        initiating_player = "human" if turn_state.streaking_player_idx == self.HUMAN_PLAYER_INDEX else "bot"
        responding_player = "human" if turn_state.acting_player_index == self.HUMAN_PLAYER_INDEX else "bot"

        # Get response cards played so far
        response_cards_played = []
        response_actions = turn_state.response_ctx.response_actions_taken
        for action in response_actions:
            if hasattr(action, "card"):
                response_cards_played.append(action.card)

        return {
            "initiating_card": initiating_card,
            "initiating_player": initiating_player,
            "responding_player": responding_player,
            "response_cards_played": response_cards_played,
        }

    def get_enhanced_turn_state(self, game_id: str) -> dict:
        game = self._get_game(game_id)
        turn_state = game.turn_state

        # Calculate card-related values
        max_cards = self.config.max_consecutive_player_actions
        remaining_cards = max(0, max_cards - turn_state.streak_idx)

        # Get selected actions for display
        selected_actions = game.selected_actions

        # Determine winner if game is over
        winner = self._determine_winner(game)

        # Build response info if responding
        response_info = self._build_response_info(turn_state)

        return {
            "turn_idx": turn_state.turn_idx,
            "streak_idx": turn_state.streak_idx,
            "streaking_player_idx": turn_state.streaking_player_idx,
            "acting_player_idx": turn_state.acting_player_index,
            "is_human_turn": turn_state.acting_player_index == self.HUMAN_PLAYER_INDEX,
            "human_player_index": self.HUMAN_PLAYER_INDEX,
            "cards_played_this_turn": turn_state.streak_idx,
            "max_cards_per_turn": max_cards,
            "remaining_cards": remaining_cards,
            "selected_actions": selected_actions,
            "game_over": game.over,
            "winner": winner,
            "is_responding": turn_state.responding,
            "response_info": response_info,
        }
