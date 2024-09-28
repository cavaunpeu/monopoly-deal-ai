-- migrate:up
CREATE TABLE actions (
  id INT PRIMARY KEY,
  plays_card BOOLEAN NOT NULL,
  is_legal BOOLEAN NOT NULL,
  is_response BOOLEAN NOT NULL,
  is_draw BOOLEAN NOT NULL,
  -- Optional fields for actions that have them
  src VARCHAR(20),
  dst VARCHAR(20),
  card VARCHAR(50),
  response_def_cls VARCHAR(100),
  created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Add append-only constraint using privilege revokes
-- Revoke UPDATE and DELETE privileges on actions table
REVOKE UPDATE, DELETE ON actions FROM PUBLIC;
REVOKE UPDATE, DELETE ON actions FROM CURRENT_USER;


-- migrate:down
-- Restore UPDATE and DELETE privileges on actions table
GRANT UPDATE, DELETE ON actions TO PUBLIC;
GRANT UPDATE, DELETE ON actions TO CURRENT_USER;
DROP TABLE actions;