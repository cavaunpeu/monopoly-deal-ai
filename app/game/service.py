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
from game.game import Game
from game.state import GameState, OpponentState, PlayerState
from models.cfr.cfr import CFR
from models.cfr.selector import BaseActionSelector


@dataclass(frozen=True)
class PublicPileSizes:
    deck: int
    discard: int


class GameService:
    _models_cache: dict = {}
    _loaded_models: dict = {}  # model_name -> CFR object

    def __init__(
        self,
        cache: GameCache,
        selector: BaseActionSelector,
        config: GameConfig,
        target_player_index: int,
        db: Optional[Session] = None,
    ):
        self.cache = cache
        self.selector = selector
        self.config = config
        self._db = db

        # Set human player index to be opposite of AI's target player index
        # AI uses the model's target_player_index, human gets the other index
        self.HUMAN_PLAYER_INDEX = 1 - target_player_index

        # Load models on first instantiation
        if not GameService._loaded_models:
            GameService._load_models()

    @property
    def db(self) -> Optional[Session]:
        return self._db

    @db.setter
    def db(self, value: Session) -> None:
        self._db = value

    def _get_config_name(self) -> str:
        """Get the config name for the current GameConfig by matching against GameConfigType enum values."""
        for config_type in GameConfigType:
            if config_type.value == self.config:
                return config_type.name
        raise ValueError("Custom GameConfig not supported - must use predefined GameConfigType values")

    def create_game(self) -> str:
        game_id = str(uuid.uuid4())
        # Convert game_id to deterministic integer seed
        random_seed = hash(game_id) % (2**31)  # Keep within int32 range
        game = Game(config=self.config, init_player_index=self.HUMAN_PLAYER_INDEX, random_seed=random_seed)
        self.cache.set(game_id, game)

        # Save the game to the database if we have a database connection
        if self.db:
            game_record = GameModel(
                id=game_id,
                config_name=self._get_config_name(),
                abstraction_cls=game.abstraction_cls.__name__,
                resolver_cls=game.resolver_cls.__name__,
                init_player_index=self.HUMAN_PLAYER_INDEX,
                random_seed=game.random_seed,
            )
            self.db.add(game_record)
            self.db.commit()

        return game_id

    def _recreate_game_from_db(self, game_id: str, db: Session) -> Game:
        # Query the game record
        record = db.query(GameModel).filter(GameModel.id == game_id).first()
        if record is None:
            raise ValueError(f"Game {game_id} not found in database.")

        # Create a new game with the same config, initial player, and random seed
        game = Game(config=self.config, init_player_index=record.init_player_index, random_seed=record.random_seed)

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
        return self._get_game(game_id).state

    def _get_player_wrapped_actions(self, game_id: str) -> list[WrappedAction]:
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
        return self.selector.select(wrapped_actions, game).action

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
    def _load_models(cls) -> None:
        """Load all models from the YAML file and cache them."""
        current_dir = Path(__file__).parent
        models_file = current_dir.parent / "models.yaml"

        try:
            with open(models_file, "r") as f:
                manifest = yaml.safe_load(f)
                cls._models_cache = manifest.get("models", {})

                # Load actual CFR models
                cls._loaded_models = {}
                for model_name, model_info in cls._models_cache.items():
                    try:
                        cfr = CFR.from_checkpoint(model_info["checkpoint_path"])
                        cls._loaded_models[model_name] = cfr
                        print(f"Loaded model {model_name} from {model_info['checkpoint_path']}")
                    except Exception as e:
                        print(f"Failed to load model {model_name}: {e}")

                print(f"Successfully loaded {len(cls._loaded_models)}/{len(cls._models_cache)} models")

        except FileNotFoundError:
            print("Warning: models.yaml not found, no models loaded")
            cls._models_cache = {}
            cls._loaded_models = {}
        except yaml.YAMLError as e:
            print(f"Error parsing models.yaml: {e}")
            cls._models_cache = {}
            cls._loaded_models = {}

    @classmethod
    def get_model_manifest(cls) -> dict:
        """Return the cached model manifest. Loads models if not already loaded."""
        if not cls._models_cache:
            cls._load_models()
        return {"models": cls._models_cache}

    @classmethod
    def get_loaded_models(cls) -> dict[str, CFR]:
        """Get all loaded CFR models. Loads models if not already loaded."""
        if not cls._loaded_models:
            cls._load_models()
        return cls._loaded_models.copy()

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
