FROM python:3.11

# ── System packages ──────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y gosu && rm -rf /var/lib/apt/lists/*

# ── User setup (mirrors PLLM Dockerfile for consistency) ────────────────────
ARG UNAME=appuser
ARG UID=1000
ARG GID=1000

RUN groupadd -g $GID -o $UNAME && \
    useradd  -m -u $UID -g $GID -o -s /bin/bash $UNAME

ENV UNAME=$UNAME

# ── Python dependencies ──────────────────────────────────────────────────────
RUN pip install --upgrade pip && \
    pip install \
        # LangGraph — our agent framework
        langgraph==0.1.5 \
        langchain==0.2.1 \
        langchain-community==0.2.1 \
        # Docker SDK — for Docker-in-Docker builds
        docker==7.1.0 \
        # HTTP calls to Ollama
        requests==2.31.0 \
        # YAML parsing (for KG ingestion of result files)
        pyyaml==6.0.1 \
        # Utilities
        python-dateutil==2.9.0.post0

# ── App directory ────────────────────────────────────────────────────────────
RUN mkdir -p /app && chown $UID:$GID /app

COPY . /app
RUN chown -R $UID:$GID /app

WORKDIR /app

# Default: keep the container alive for interactive use
CMD ["tail", "-f", "/dev/null"]
