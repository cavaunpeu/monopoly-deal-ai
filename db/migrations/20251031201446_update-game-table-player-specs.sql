-- migrate:up
-- Add player_specs column as JSON object (dict)
-- Format: {"0": {"abstraction_cls": "IntentStateAbstraction", "resolver_cls": "GreedyActionResolver"}, "1": {...}}
ALTER TABLE games ADD COLUMN player_specs JSON NOT NULL DEFAULT '{}';

-- Backfill existing data: create player_specs from abstraction_cls and resolver_cls
-- For each existing game, create player_specs dict with the same abstraction_cls/resolver_cls for both players
UPDATE games
SET player_specs = json_build_object(
  '0', json_build_object('abstraction_cls', abstraction_cls, 'resolver_cls', resolver_cls),
  '1', json_build_object('abstraction_cls', abstraction_cls, 'resolver_cls', resolver_cls)
);

-- Remove old columns
ALTER TABLE games DROP COLUMN abstraction_cls;
ALTER TABLE games DROP COLUMN resolver_cls;

-- migrate:down
-- Re-add old columns
ALTER TABLE games ADD COLUMN abstraction_cls VARCHAR(80) NOT NULL DEFAULT 'IntentStateAbstraction';
ALTER TABLE games ADD COLUMN resolver_cls VARCHAR(80) NOT NULL DEFAULT 'GreedyActionResolver';

-- Restore data from player_specs (use player 0's spec)
UPDATE games
SET abstraction_cls = player_specs->'0'->>'abstraction_cls',
    resolver_cls = player_specs->'0'->>'resolver_cls'
WHERE json_typeof(player_specs) = 'object' AND player_specs ? '0';

-- Remove player_specs column
ALTER TABLE games DROP COLUMN player_specs;
