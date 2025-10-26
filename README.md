# Monopoly Deal Research Platform

This work presents a modified version of the card game [Monopoly Deal](https://en.wikipedia.org/wiki/Monopoly_Deal). It serves as a platform for ongoing independent research on systems and algorithms for sequential decision-making under imperfect information, focusing on classical and modern approaches from game theory and reinforcement learning. It includes models, training pipelines, experiment tracking, and a full web application for human–AI interaction—providing a practical testbed for studying complex games.

**Play against the AI**: [monopolydeal.ai](https://monopolydeal.ai)

## Game Overview

This work implements a modified two-player version of [Monopoly Deal](https://en.wikipedia.org/wiki/Monopoly_Deal), the card game based on the classic board game Monopoly. In this game, players compete to be the first to complete two full property sets. On each turn, they draw two new cards, then play up to two cards total from their hand, including property cards (for building colored sets), cash cards (for collecting cash), and rent cards (for charging rent). When rent is charged, the opponent must respond by selecting cash or property to pay the debt, or by using a "Just Say No" card to counter the action.

The game provides a testbed for studying various algorithmic and systems challenges related to sequential decision-making in imperfect-information games.

## System Overview

The platform is designed to be modular, extensible, and interactive. It consists of the following components:

- **Model Training**: Training loops and experiment tracking via Weights & Biases
- **Backend API**: FastAPI-based service managing game state
- **Frontend Web App**: React/Next.js interface for human-AI gameplay with real-time state visualization
- **Database**: PostgreSQL for game metadata and individual game tracking
- **Deployment**: Docker-based deployment to Google Cloud Run

## Prerequisites

Before getting started, ensure you have the following installed:

- **uv** for Python package management: [docs.astral.sh/uv/](https://docs.astral.sh/uv/)
- **npm** for Node.js package management: [nodejs.org](https://nodejs.org/)
- **PostgreSQL 15+**: [postgresql.org](https://www.postgresql.org/download/)
- **dbmate** for database management: [github.com/amacneil/dbmate](https://github.com/amacneil/dbmate)
- **just** command runner: [github.com/casey/just](https://github.com/casey/just)
- **Docker & Docker Compose**: [docs.docker.com](https://docs.docker.com/get-docker/)
- **direnv** for environment variable management: [direnv.net](https://direnv.net/docs/installation.html)

## Getting Started

### Local Development (Non-Containerized)

For development with hot reloads (and non-trivial setup):

```bash
# Clone the repository
git clone https://github.com/cavaunpeu/monopoly-deal-ai.git
cd monopoly-deal-ai

# Set up Python environment
uv venv
source .venv/bin/activate
just install-dev-requirements

# Set up frontend dependencies
cd frontend && npm install && cd ..

# Activate the environment variables
direnv allow

# Copy the example .envrc file
cp .envrc.example .envrc

# Set up PostgreSQL database
just db-recreate-migrate-seed

# Start the development servers
just dev
```

**Access the application:**
- Frontend: http://localhost:3000
- Backend API docs: http://localhost:8000/docs

### Containerized Local Development

For development without hot reloads but with easier setup:

```bash
# Clone the repository
git clone https://github.com/cavaunpeu/monopoly-deal-ai.git
cd monopoly-deal-ai

# Start the full stack (backend, frontend, database)
just docker-dev
```

**Access the application:**
- Frontend: http://localhost:3000
- Backend API docs: http://localhost:8000/docs

## Repository Structure

- `models/cfr/` - Counterfactual regret minimization implementation and training tools
- `app/` - Backend API service and database models
- `frontend/` - React web application
- `game/` - Core game logic and state management
- `db/` - Database schema and migrations

## Application Architecture

The web application uses a standard microservices architecture with FastAPI backend, React frontend, and PostgreSQL database:

- **FastAPI Backend**: RESTful API handling game state and model inference
- **React/Next.js Frontend**: Type-safe web interface for human-AI interaction
- **PostgreSQL Database**: Stores game metadata and individual game results
- **Docker Containerization**: Multi-stage builds with single container per service
- **Google Cloud Run**: Deployment and liveness probes
- **Fault Tolerance**: Reconstruct game state from database logs when cache fails

## Training and Evaluation Architecture

The CFR learning pipeline parallelized CFR trainining, experiment tracking, and checkpoint management:

- **Ray**: Multi-core parallelization with shared memory for CFR self-play rollouts
- **Weights & Biases**: Experiment tracking with metrics logging
- **Kubernetes Jobs**: Training infrastructure on GKE with configurable CPU/memory resources
- **Checkpoint Management**: Model checkpointing to GCS
- **Evaluation**: Model evaluation against different opponents (random, risk-aware)
- **State Abstraction**: Configurable abstractions to reduce game tree complexity
- **Reproducibility**: Git commit tracking, random seeds, and deterministic training

## Publications

- **Monopoly Deal: A Benchmark Environment for Bounded One-Sided Response Games** (in progress)

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## Legal Notice

This project implements a modified version of Monopoly Deal for research purposes. "Monopoly Deal" is a trademark of Hasbro. This is not an official product.

## Citation

If you use this work in your research, please cite:

```bibtex
@software{wolf2025monopolydealai,
  title={Monopoly Deal Research Platform},
  author={Wolf, Will},
  year={2025},
  url={https://github.com/cavaunpeu/monopoly-deal-ai},
  license={Apache-2.0}
}
```
