# FSE-AIWare Python Dependency Resolver

## Requirements

- Docker + Docker Compose
- [Ollama](https://ollama.com) running locally with `gemma2` pulled
- The PLLM result data (tars + CSVs) see Data Setup below
- Python 3.10+ on the host 

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/vpoweska/fse-aiware-python-dependencies.git
cd fse-aiware-python-dependencies
```

### 2. Get the data

Extract hard-gists.tar.gz

### 3. Start Ollama with Gemma 2

```bash
ollama pull gemma2
OLLAMA_HOST=0.0.0.0:11437 ollama serve &
```

### 4. Open a new terminal and start the container

```bash
cd tools/our_tool
docker-compose up -d --build
```

### 5. Build the solutions DB

```bash
docker exec -it enhanced-resolver-test python helpers/build_solution_db_v2.py
```

## Running

### Run all 2,891 snippets

```bash
docker exec -it enhanced-resolver-test python run_all.py \
  -g '/gists' \
  -m 'gemma2' \
  -b 'http://172.18.0.1:11437/' \
  -l 5 \
  -r 0
```

### Run a limited batch

```bash
docker exec -it enhanced-resolver-test python run_all.py \
  -g '/gists' \
  -m 'gemma2' \
  -b 'http://172.18.0.1:11437/' \
  -l 5 \
  -r 0 \
  --limit 50
```

### Run a single snippet

```bash
docker exec -it enhanced-resolver-test python test_executor.py \
  -f '/gists/SNIPPET_ID/snippet.py' \
  -m 'gemma2' \
  -b 'http://172.18.0.1:11437/' \
  -l 5 \
  -r 0
```

### Resume from a specific snippet

```bash
docker exec -it enhanced-resolver-test python run_all.py \
  -g '/gists' \
  -m 'gemma2' \
  -b 'http://172.18.0.1:11437/' \
  -l 5 \
  -r 0 \
  --start-from SNIPPET_ID
```

### flags
description - default value

 `-l` Max LLM loops per snippet - 10 
 `-r` - Python version range (0 = single version) - 0 
 `--limit N` - Only process first N snippets - all 
 `--no-skip` - Re-run already-solved snippets - skip by default
 `--start-from ID` - Resume from a specific snippet ID - start from beginning


Results are written to `hard-gists/results_summary.csv` NOTE: this will override previous runs if you are doing multiple sessions so be sure to save your csvs after each run.
