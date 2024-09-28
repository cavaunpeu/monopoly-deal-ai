from sqlalchemy.orm import Session

from app.db_models import Action
from db.utils import insert_actions
from game.action import ACTION_TO_IDX, PassAction, YieldAction


class TestActionsDatabase:
    """Test database operations for actions table"""

    def test_insert_actions_initial_insert(self, test_db_session: Session):
        """Test inserting actions for the first time"""
        # Initially, no actions should exist
        actions = test_db_session.query(Action).all()
        assert len(actions) == 0

        # Insert actions
        insert_actions(test_db_session)

        # Verify actions were inserted
        actions = test_db_session.query(Action).all()
        assert len(actions) == len(ACTION_TO_IDX)

        # Verify specific actions exist
        pass_action = test_db_session.query(Action).filter(Action.id == ACTION_TO_IDX[PassAction()]).first()
        assert pass_action is not None
        assert pass_action.plays_card is False
        assert pass_action.is_legal is True
        assert pass_action.is_response is False
        assert pass_action.is_draw is False
        assert pass_action.src is None
        assert pass_action.dst is None
        assert pass_action.card is None
        assert pass_action.response_def_cls is None

        yield_action = test_db_session.query(Action).filter(Action.id == ACTION_TO_IDX[YieldAction()]).first()
        assert yield_action is not None
        assert yield_action.plays_card is False
        assert yield_action.is_legal is True
        assert yield_action.is_response is True
        assert yield_action.is_draw is False

    def test_actions_have_correct_properties(self, test_db_session: Session):
        """Test that actions have the correct properties based on their type"""
        insert_actions(test_db_session)

        actions = test_db_session.query(Action).all()

        for action in actions:
            # All actions should have required fields
            assert action.id is not None
            assert action.plays_card is not None
            assert action.is_legal is not None
            assert action.is_response is not None
            assert action.is_draw is not None
            assert action.created_at is not None

    def test_append_only_constraint(self, test_db_session: Session):
        """Test that insert_actions is idempotent and doesn't create duplicates"""
        # First insert - should work fine
        insert_actions(test_db_session)
        initial_count = test_db_session.query(Action).count()
        assert initial_count == len(ACTION_TO_IDX)

        # Second insert - should not create duplicates (idempotent)
        insert_actions(test_db_session)
        final_count = test_db_session.query(Action).count()
        assert final_count == initial_count  # No duplicates created

        # Verify we can read the actions
        actions = test_db_session.query(Action).all()
        assert len(actions) == len(ACTION_TO_IDX)

        # Verify specific action properties are preserved
        pass_action = test_db_session.query(Action).filter(Action.id == ACTION_TO_IDX[PassAction()]).first()
        assert pass_action is not None
        assert pass_action.plays_card is False
        assert pass_action.is_legal is True
