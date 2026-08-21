FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*


COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-install-project

COPY README.md hybrid_surrogate_server.py ./
COPY hybrid_surr ./hybrid_surr


EXPOSE 7331

CMD ["uv", "run", "python", "hybrid_surrogate_server.py"]
