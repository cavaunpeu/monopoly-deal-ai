-- migrate:up
CREATE TABLE selected_actions (
  turn_idx INT NOT NULL,
  streak_idx INT NOT NULL,
  player_idx INT NOT NULL,
  action_id INT NOT NULL,
  game_id VARCHAR(36) NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT NOW(),
  FOREIGN KEY (action_id) REFERENCES actions(id) ON DELETE CASCADE,
  FOREIGN KEY (game_id) REFERENCES games(id) ON DELETE CASCADE,
  PRIMARY KEY (turn_idx, streak_idx, player_idx, game_id, created_at)
);

-- Add append-only constraint using privilege revokes
-- Revoke UPDATE and DELETE privileges on selected_actions table
REVOKE UPDATE, DELETE ON selected_actions FROM PUBLIC;
REVOKE UPDATE, DELETE ON selected_actions FROM CURRENT_USER;

-- migrate:down
-- Restore UPDATE and DELETE privileges on selected_actions table
GRANT UPDATE, DELETE ON selected_actions TO PUBLIC;
GRANT UPDATE, DELETE ON selected_actions TO CURRENT_USER;
DROP TABLE selected_actions;