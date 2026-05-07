# syntax=docker/dockerfile:1.7

# ---------------------------------------------------------------------------
# Stage 1 — frontend build
#
# Vite emits the static SPA into ../jobai/api/static/ (relative to
# frontend/) which the FastAPI app mounts at /. We build it in a
# Node-only stage so the runtime image doesn't carry Node, npm, or
# the 200MB-ish node_modules tree.
# ---------------------------------------------------------------------------
FROM node:23-alpine AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
# vite.config.ts emits to ../jobai/api/static/, which when run with
# the working directory being /build resolves to /jobai/api/static/.
RUN mkdir -p /jobai/api && npm run build


# ---------------------------------------------------------------------------
# Stage 2 — runtime
#
# python:3.12-slim is the smallest official image that satisfies
# pyproject.toml's `requires-python = ">=3.12"`. Microsoft's
# playwright/python:*-jammy image ships Python 3.10 (Ubuntu 22.04
# default) so it isn't usable here. We install the Chromium that
# Playwright + Patchright drive ourselves via `playwright install`.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    JOBAI_DB_PATH=/data/jobai.db \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

# System libs Chromium links against on Debian slim. Playwright's own
# `--with-deps` would pull these in but also installs build-essential
# we don't need at runtime; listing them here keeps the image lean.
# We also install Node 20 + the ``claude`` CLI so the subscription
# agent backend (claude-agent-sdk) has the binary it spawns. The
# image stays usable in API-key mode regardless — claude is only
# invoked when JOBAI_AGENT_BACKEND=subscription.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl tini gnupg util-linux \
      libglib2.0-0 libnss3 libnspr4 libdbus-1-3 libatk1.0-0 \
      libatk-bridge2.0-0 libcups2 libdrm2 libxcomposite1 libxdamage1 \
      libxext6 libxfixes3 libxrandr2 libgbm1 libxkbcommon0 libpango-1.0-0 \
      libcairo2 libasound2 libatspi2.0-0 libx11-xcb1 \
 && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
 && apt-get install -y --no-install-recommends nodejs \
 && npm install -g @anthropic-ai/claude-code \
 && apt-get purge -y --auto-remove gnupg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps. Editable install keeps /app/jobai as the
# active package so the COPY below lands inside the importable
# `jobai.api.static` path the FastAPI app resolves at request time.
# (A non-editable install would copy jobai/ into site-packages, and
# the SPA bundle COPY'd into /app afterwards wouldn't be visible.)
COPY pyproject.toml LICENSE README.md ./
COPY jobai ./jobai
RUN pip install --no-cache-dir -e . \
 && playwright install chromium \
 && patchright install chromium \
 && rm -rf /root/.cache/pip

# Copy the built SPA from stage 1 into the editable package.
COPY --from=frontend /jobai/api/static /app/jobai/api/static

# /data is the persistent volume mount point; the SQLite file lives
# here so it survives container restarts. Compose mounts a named
# volume to this path.
RUN mkdir -p /data

# Run as a non-root user. The ``claude`` CLI's
# ``--dangerously-skip-permissions`` flag (used by the subscription
# backend) refuses to run as root for security reasons; without this
# the entire subscription path bombs with "cannot be used with
# root/sudo privileges for security reasons" before the first
# message ever streams. The entrypoint stays root just long enough
# to fix volume ownership, then drops to ``jobai`` for the actual
# server process.
RUN useradd --system --create-home --home-dir /home/jobai --shell /bin/bash jobai \
 && chown -R jobai:jobai /app /home/jobai
ENV HOME=/home/jobai

COPY docker-entrypoint.sh /usr/local/bin/jobai-entrypoint
RUN chmod +x /usr/local/bin/jobai-entrypoint

VOLUME ["/data"]

EXPOSE 8421

# tini is PID 1 so signals propagate cleanly to the API + scheduler.
# The entrypoint script chowns /data to jobai (handles existing
# volumes from previous root-only image versions) and then exec's
# the CMD as the unprivileged jobai user.
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/jobai-entrypoint"]
CMD ["sh", "-c", "jobai migrate && jobai serve --host 0.0.0.0 --port 8421"]
