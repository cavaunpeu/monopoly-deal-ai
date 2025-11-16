-- migrate:up
-- Add model_name column to store which AI model was used for this game
ALTER TABLE games ADD COLUMN model_name VARCHAR(80);

-- migrate:down
-- Remove model_name column
ALTER TABLE games DROP COLUMN model_name;
