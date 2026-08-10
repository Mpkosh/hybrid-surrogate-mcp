FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
COPY README.md hybrid_surrogate_server.py ./
COPY hybrid_surr ./hybrid_surr

RUN uv sync --frozen --no-dev

EXPOSE 7331

CMD ["uv", "run", "python", "hybrid_surrogate_server.py"]
