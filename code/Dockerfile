# ═══════════════════════════════════════════════════════════════════════════
# Multi-stage Dockerfile for Multiturn Workflow Agent
# Stage 1: Build React frontend
# Stage 2: Create runtime container with FastAPI backend + nginx
# ═══════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Frontend Builder (Node 20 Alpine)
# ─────────────────────────────────────────────────────────────────────────────
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy package files
COPY frontend/package*.json ./

# Install dependencies
RUN npm install --silent

# Copy frontend source
COPY frontend/ .

# Build argument for API URL
ARG REACT_APP_API_URL=/api
ENV REACT_APP_API_URL=$REACT_APP_API_URL

# Build production bundle
RUN npm run build

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Runtime Container (UBI 9 Python 3.12)
# ─────────────────────────────────────────────────────────────────────────────
FROM registry.access.redhat.com/ubi9/python-312:latest

USER root

# Install system dependencies
RUN dnf install -y \
    nginx \
    && dnf clean all

# Install supervisor for process management
RUN pip install --no-cache-dir supervisor

# ─────────────────────────────────────────────────────────────────────────────
# Backend Setup
# ─────────────────────────────────────────────────────────────────────────────
WORKDIR /app/backend

# Copy backend requirements and install
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY backend/ .

# ─────────────────────────────────────────────────────────────────────────────
# Frontend Setup
# ─────────────────────────────────────────────────────────────────────────────
# Copy built frontend from builder stage
COPY --from=frontend-builder /app/frontend/build /usr/share/nginx/html

# ─────────────────────────────────────────────────────────────────────────────
# Nginx Configuration
# ─────────────────────────────────────────────────────────────────────────────
# Remove default nginx config
RUN rm -f /etc/nginx/conf.d/default.conf /etc/nginx/nginx.conf

# Copy custom nginx configuration
COPY nginx-combined.conf /etc/nginx/nginx.conf

# Create nginx directories and set permissions
RUN mkdir -p /var/log/nginx /var/lib/nginx /run \
    && chown -R 1001:0 /var/log/nginx /var/lib/nginx /run /usr/share/nginx/html \
    && chmod -R g+rwX /var/log/nginx /var/lib/nginx /run /usr/share/nginx/html

# ─────────────────────────────────────────────────────────────────────────────
# Supervisord Configuration
# ─────────────────────────────────────────────────────────────────────────────
RUN mkdir -p /etc/supervisor/conf.d /var/log/supervisor
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Set permissions for supervisor
RUN chown -R 1001:0 /etc/supervisor /var/log/supervisor \
    && chmod -R g+rwX /etc/supervisor /var/log/supervisor

# ─────────────────────────────────────────────────────────────────────────────
# Application Setup
# ─────────────────────────────────────────────────────────────────────────────
# Create data directory for workflow generation
RUN mkdir -p /app/data \
    && chown -R 1001:0 /app \
    && chmod -R g+rwX /app

# Switch to non-root user
USER 1001

# Expose port
EXPOSE 9000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:9000/api/health || exit 1

# Start supervisord (manages nginx + uvicorn)
CMD ["/opt/app-root/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]