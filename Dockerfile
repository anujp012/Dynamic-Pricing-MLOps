# ── Stage 1: Install dependencies ─────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ── Stage 2: Lean final image ──────────────────────────────
FROM python:3.10-slim

WORKDIR /app

# Copy only installed packages from builder (NOT .venv)
COPY --from=builder /root/.local /root/.local

# Copy only files the app actually needs
COPY api.py .
COPY main.py .
COPY model.pkl .
COPY mlflow.db .
COPY data.db .
COPY uber.csv .

ENV PATH=/root/.local/bin:$PATH

EXPOSE 8000

# Health check so Kubernetes knows when pod is truly ready
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8000/ || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]