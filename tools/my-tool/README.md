# PLLM Plus — Agentic Python Dependency Resolver

An improved version of the PLLM baseline tool for the FSE-AIWare 2026 competition.
Built on top of PLLM, with three main improvements added.

## What It Does

Given a Python snippet file, the tool automatically figures out which packages
and versions are needed to make it run. It does this by:

1. Parsing the snippet to find all imports
2. Checking a local knowledge graph (SQLite DB) built from historical PLLM results
3. If we already know working versions, try them directly in Docker
4. If not, ask an LLM (via Ollama) to suggest versions using a Proposer -> Critic -> Decider debate
5. Build a Docker container with the suggested packages and actually run the snippet
6. If it fails, classify the error and loop back to the LLM with that info
7. Repeat up to N times

Success is defined the same way as PLLM, the snippet must both build AND run without dependency errors.

## The Three Improvements Over PLLM

**1. Knowledge Graph** — before calling the LLM, we check a SQLite database of
known-working package versions from previous PLLM experiment runs. If we have
coverage of 70%+ of the required packages, we skip the LLM entirely and go
straight to Docker.

**2. Structured Error Classifier** — instead of dumping raw Docker logs into the
LLM, we first classify the error (version conflict, missing distribution, Python
version mismatch, etc.) and extract the key facts. The LLM gets a focused prompt
instead of 200 lines of noise.

**3. Multi-Agent Debate** — instead of one LLM call per iteration, we run three
roles: a Proposer that suggests requirements, a Critic that reviews them, and a
Decider that either accepts or refines the proposal. The Critic often catches bad
versions before a Docker build is even attempted.

## Folder Structure

```
tools/pllm-plus/
├── README.md
├── build_kg.py          # run once to build the knowledge graph DB
├── run.py               # main entry point — run this for each snippet
├── agent.py             # LangGraph pipeline
├── Dockerfile
├── docker-compose.yml
└── helpers/
    ├── __init__.py
    ├── knowledge_graph.py   # SQLite knowledge graph
    ├── error_classifier.py  # classifies Docker/pip error logs
    ├── multi_agent.py       # Proposer, Critic, Decider LLM roles
    └── docker_runner.py     # builds and runs Docker containers
```

## Prerequisites

- Docker installed and running
- Ollama installed (https://ollama.com/) with `gemma2` pulled
- The hard-gists dataset extracted

### Install and start Ollama

If Ollama is not already running on the machine, start it and pull the model:

```bash
ollama serve &
ollama pull gemma2
```

If Ollama is already running (check with `ps aux | grep ollama`), find which port it is on:

```bash
ss -tlnp | grep ollama
```

Use that port in the `-b` flag when running snippets.

### Extract the hard-gists dataset

The snippets live in `hard-gists.tar.gz` at the repo root. Extract it:

```bash
cd ~//fse-aiware-python-dependencies
tar -xzf hard-gists.tar.gz
```

Also extract the training data archives (needed to build the knowledge graph):

```bash
cd pllm_results
for f in *.tar.gz; do tar -xzf "$f"; done

cd ../pyego-results
for f in *.tar.gz; do tar -xzf "$f"; done

cd ../readpy-results
for f in *.tar.gz; do tar -xzf "$f"; done
```

## Setup

### 1. Environment variables

Run this on the host machine before building:

```bash
echo "USER=$(whoami)" > .env
echo "UID=$(id -u)" >> .env
echo "GID=$(id -g)" >> .env
```

### 2. Build and start the container

```bash
docker-compose build
docker-compose up -d
```

### 3. Build the knowledge graph (run once)

This extracts the training data archives and populates the SQLite DB.
The DB is saved to `/gists/knowledge_graph.db` on the mounted volume
so it persists across container restarts.

```bash
docker exec -it pllm-plus-test python build_kg.py
```

To check what's in the DB without rebuilding:

```bash
docker exec -it pllm-plus-test python build_kg.py --stats
```

To force a full rebuild:

```bash
docker exec -it pllm-plus-test python build_kg.py --force
```

## Running

### Single snippet

```bash
docker exec -it pllm-plus-test python run.py \
    -f /gists/<gist_id>/snippet.py \
    -m gemma2 \
    -b http://<ollama_host>:<port> \
    -l 5
```

### All snippets (bash loop)

```bash
for snippet in ~/path/to/hard-gists/*/snippet.py; do
    gist_id=$(basename $(dirname "$snippet"))
    docker exec pllm-plus-test python run.py \
        -f "/gists/$gist_id/snippet.py" \
        -m gemma2 \
        -b http://<ollama_host>:<port> \
        -l 5 \
        -o /gists/pllm-plus-results.csv
done
```

## Arguments

| Flag | Description                  | Default                  |
| ---- | ---------------------------- | ------------------------ |
| `-f` | Path to the snippet file     | required                 |
| `-m` | Ollama model name            | `gemma2`                 |
| `-b` | Ollama base URL              | `http://localhost:11434` |
| `-l` | Max LLM retry iterations     | `5`                      |
| `-p` | Initial Python version guess | `3.8`                    |
| `-o` | Path to output CSV file      | `/gists/results.csv`     |
| `-v` | Verbose output               | off                      |

## Output

Results are appended to a single CSV file. Each row is one snippet:

| Column                 | Description                    |
| ---------------------- | ------------------------------ |
| `snippet_path`         | path to the snippet            |
| `python_version`       | Python version used            |
| `status`               | `success` or `failed`          |
| `iterations`           | how many LLM loops were needed |
| `elapsed_seconds`      | total time taken               |
| `used_knowledge_graph` | whether KG skipped the LLM     |
| `requirements`         | final package versions as JSON |
| `final_error_type`     | last error category seen       |
