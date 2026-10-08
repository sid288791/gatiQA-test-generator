# gatiQA-test-generator

AI-powered test case generation service built with FastAPI and Pydantic AI.

## Overview

gatiQA-test-generator is a backend service that generates QA test cases using LLMs. It exposes a FastAPI API for test generation, test case storage, and analysis summaries, with observability via Langfuse and persistence in PostgreSQL (with pgvector for embeddings).

## Tech Stack

- **FastAPI** — API framework
- **Pydantic AI** — LLM orchestration
- **PostgreSQL** (pgvector) — test case storage & embeddings
- **Langfuse** — observability / tracing
- **Docker Compose** — local infrastructure (Postgres + Grafana k6 MCP server)
- **pytest** — testing

## Getting Started

### Prerequisites

- Python 3.11+
- Docker & Docker Compose

### Installation

```bash
# Clone the repo
git clone https://github.com/sid288791/gatiQA-test-generator.git
cd gatiQA-test-generator

# Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configuration

Copy `.env` and configure settings (LLM API keys, database URL, Langfuse keys, etc.).

### Run Infrastructure

```bash
docker compose up -d
```

This starts:
- **PostgreSQL** (pgvector) on port 5432
- **Grafana k6 MCP server** on port 8080

### Run the Server

```bash
python main.py
```

The API will be available at `http://localhost:8000`.

### Run Tests

```bash
pytest
```

## Project Structure

```
.
├── app/
│   ├── config.py          # Settings via pydantic-settings
│   ├── db.py              # PostgreSQL test case store
│   ├── embeddings.py      # Embedding generation
│   ├── models.py          # Pydantic request/response models
│   ├── observability.py   # Langfuse tracing setup
│   ├── server.py          # FastAPI app & routes
│   ├── service.py         # Test generation business logic
│   └── validation.py      # Input validation
├── db/                    # DB init scripts
├── tests/                 # pytest test suite
├── docker-compose.yml     # Local infrastructure
├── main.py                # Entry point
└── requirements.txt       # Python dependencies
```

## License

This project is proprietary.
