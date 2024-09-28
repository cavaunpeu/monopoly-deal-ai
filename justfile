# Format and lint all Python files
format:
    uv run ruff check --select I --fix . && uv run ruff check --fix . && uv run ruff format .


# Format and lint a specific file
format-file file:
    uv run ruff check --select I --fix {{file}} && uv run ruff check --fix {{file}} && uv run ruff format {{file}}

# Type check all Python files
typecheck:
    uv run pyright app game models db tests

# Type check a specific file
typecheck-file file:
    uv run pyright {{file}}

# Format and type check
format-and-typecheck:
    just format && just typecheck

# Test backend, test frontend, format, typecheck
test-and-format-and-typecheck:
    just test-all && just format-and-typecheck

# Run tests
test:
    uv run pytest

# Run frontend tests
test-frontend:
    cd frontend && npm run test:run

# Run all tests (backend + frontend)
test-all:
    just test && just test-frontend

# Install requirements
install-requirements:
    uv sync

# Install requirements (including dev)
install-dev-requirements:
    uv sync --group dev

# Database management commands
db-create:
    createdb mdeal || true

db-status:
    dbmate status

db-migrate:
    dbmate up

db-rollback:
    dbmate down

db-drop-and-create:
    dbmate drop && dbmate create

db-seed:
    uv run python db/seed_data.py

db-recreate-migrate-seed:
    dbmate drop && dbmate create && dbmate up && uv run python db/seed_data.py

db-new-migration name:
    dbmate new {{name}}

# Start FastAPI server
server:
    uv run python -m uvicorn app.main:app --reload --port 8000

# Start frontend development server
frontend:
    cd frontend && npm run dev

# Build frontend for production
build-frontend:
    cd frontend && npm run build

# Start both server and frontend in parallel
dev:
    just server & just frontend & wait

# Build docker images for the full stack
docker-build:
    docker compose build

# Set up database in docker environment
docker-db-setup:
    docker compose exec backend bash -c "dbmate drop || true"
    docker compose exec backend dbmate create
    docker compose exec backend dbmate up
    docker compose exec backend python db/seed_data.py

# Wait for database to be ready
docker-wait-for-db:
    docker compose exec backend bash -c "until python -c 'import psycopg2; psycopg2.connect(host=\"db\", port=5432, user=\"postgres\", password=\"postgres\", database=\"postgres\")' 2>/dev/null; do sleep 1; done"

# Start full stack with docker-compose (backend, frontend, database)
docker-dev:
    docker compose up --detach
    just docker-wait-for-db
    just docker-db-setup
    docker compose logs --follow

# Deploy with pre-checks (runs tests, format, typecheck, then deploys)
deploy version="patch":
    just test-and-format-and-typecheck
    ./scripts/deploy.sh {{version}}