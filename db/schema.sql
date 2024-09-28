\restrict wGDT9RORC3bXt1oZhTLBwEjEym1icEHXBb9x1p1QdTknrKVQzmtdjMIllwSuVJw

-- Dumped from database version 16.10 (Homebrew)
-- Dumped by pg_dump version 16.10 (Homebrew)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: actions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.actions (
    id integer NOT NULL,
    plays_card boolean NOT NULL,
    is_legal boolean NOT NULL,
    is_response boolean NOT NULL,
    is_draw boolean NOT NULL,
    src character varying(20),
    dst character varying(20),
    card character varying(50),
    response_def_cls character varying(100),
    created_at timestamp without time zone DEFAULT now() NOT NULL
);


--
-- Name: configs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.configs (
    name character varying(36) NOT NULL,
    cash_card_values json NOT NULL,
    rent_cards_per_property_type integer NOT NULL,
    required_property_sets integer NOT NULL,
    deck_size_multiplier integer NOT NULL,
    initial_hand_size integer NOT NULL,
    new_cards_per_turn integer NOT NULL,
    max_consecutive_player_actions integer NOT NULL,
    created_at timestamp without time zone DEFAULT now() NOT NULL,
    updated_at timestamp without time zone
);


--
-- Name: games; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.games (
    id character varying(36) NOT NULL,
    config_name character varying(36) NOT NULL,
    init_player_index integer NOT NULL,
    abstraction_cls character varying(80) NOT NULL,
    resolver_cls character varying(80) NOT NULL,
    random_seed integer NOT NULL,
    created_at timestamp without time zone DEFAULT now() NOT NULL
);


--
-- Name: schema_migrations; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.schema_migrations (
    version character varying NOT NULL
);


--
-- Name: selected_actions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.selected_actions (
    turn_idx integer NOT NULL,
    streak_idx integer NOT NULL,
    player_idx integer NOT NULL,
    action_id integer NOT NULL,
    game_id character varying(36) NOT NULL,
    created_at timestamp without time zone DEFAULT now() NOT NULL
);


--
-- Name: actions actions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.actions
    ADD CONSTRAINT actions_pkey PRIMARY KEY (id);


--
-- Name: configs configs_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.configs
    ADD CONSTRAINT configs_pkey PRIMARY KEY (name);


--
-- Name: games games_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.games
    ADD CONSTRAINT games_pkey PRIMARY KEY (id);


--
-- Name: schema_migrations schema_migrations_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.schema_migrations
    ADD CONSTRAINT schema_migrations_pkey PRIMARY KEY (version);


--
-- Name: selected_actions selected_actions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.selected_actions
    ADD CONSTRAINT selected_actions_pkey PRIMARY KEY (turn_idx, streak_idx, player_idx, game_id, created_at);


--
-- Name: games games_config_name_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.games
    ADD CONSTRAINT games_config_name_fkey FOREIGN KEY (config_name) REFERENCES public.configs(name) ON DELETE CASCADE;


--
-- Name: selected_actions selected_actions_action_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.selected_actions
    ADD CONSTRAINT selected_actions_action_id_fkey FOREIGN KEY (action_id) REFERENCES public.actions(id) ON DELETE CASCADE;


--
-- Name: selected_actions selected_actions_game_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.selected_actions
    ADD CONSTRAINT selected_actions_game_id_fkey FOREIGN KEY (game_id) REFERENCES public.games(id) ON DELETE CASCADE;


--
-- PostgreSQL database dump complete
--

\unrestrict wGDT9RORC3bXt1oZhTLBwEjEym1icEHXBb9x1p1QdTknrKVQzmtdjMIllwSuVJw


--
-- Dbmate schema migrations
--

INSERT INTO public.schema_migrations (version) VALUES
    ('20250910105900'),
    ('20250911014003'),
    ('20250912112002'),
    ('20250928131011');
