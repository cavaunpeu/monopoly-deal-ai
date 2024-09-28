-- migrate:up
CREATE TABLE configs (
  name VARCHAR(36) PRIMARY KEY, -- short name for the config
  cash_card_values JSON NOT NULL,
  rent_cards_per_property_type INT NOT NULL,
  required_property_sets INT NOT NULL,
  deck_size_multiplier INT NOT NULL,
  initial_hand_size INT NOT NULL,
  new_cards_per_turn INT NOT NULL,
  max_consecutive_player_actions INT NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMP
);

CREATE TABLE games (
  id VARCHAR(36) PRIMARY KEY, -- uuid
  config_name VARCHAR(36) NOT NULL,
  init_player_index int NOT NULL,
  abstraction_cls varchar(80) NOT NULL,
  resolver_cls varchar(80) NOT NULL,
  random_seed INT NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT NOW(),
  FOREIGN KEY (config_name) REFERENCES configs(name) ON DELETE CASCADE
);

-- migrate:down
DROP TABLE games;
DROP TABLE configs;