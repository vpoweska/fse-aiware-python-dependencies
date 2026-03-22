FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    docker.io \
    gosu \
    && rm -rf /var/lib/apt/lists/*

ARG UNAME=appuser
ARG UID=1000
ARG GID=1000
ARG DOCKER_GID=999
ENV UNAME=${UNAME}

RUN groupadd -g $GID $UNAME 2>/dev/null || true && \
    useradd -u $UID -g $GID -s /bin/bash -m $UNAME 2>/dev/null || true && \
    groupadd -g $DOCKER_GID dockerhost 2>/dev/null || true && \
    usermod -aG dockerhost $UNAME 2>/dev/null || true

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python", "run.py"]
