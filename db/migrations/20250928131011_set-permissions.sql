-- migrate:up

-- Grant permissions to postgres user on all tables and sequences
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;
GRANT USAGE ON SCHEMA public TO postgres;

-- migrate:down

-- Revoke permissions from postgres user
REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA public FROM postgres;
REVOKE ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public FROM postgres;
REVOKE USAGE ON SCHEMA public FROM postgres;

