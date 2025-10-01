#!/bin/sh
set -e

# Versi Docker Compose yang mau di-install
DOCKER_COMPOSE_VERSION=v2.29.2

echo "📥 Downloading Docker Compose ${DOCKER_COMPOSE_VERSION}..."
curl -L "https://github.com/docker/compose/releases/download/${DOCKER_COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" \
    -o /usr/local/bin/docker-compose

echo "🔧 Setting execute permissions..."
chmod +x /usr/local/bin/docker-compose

echo "✅ Docker Compose installed successfully!"
docker-compose version
