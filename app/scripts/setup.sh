#!/bin/bash

# Movie Recommendation System Setup Script
# This script initializes the entire system

set -e

echo "=========================================="
echo "Movie Recommendation System Setup"
echo "=========================================="

# Check if .env exists
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "Please edit .env and add your TMDB_API_KEY"
    echo "Then run this script again."
    exit 1
fi

# Check if TMDB_API_KEY is set
if grep -q "your_tmdb_api_key_here" .env; then
    echo "ERROR: Please set your TMDB_API_KEY in .env file"
    exit 1
fi

echo ""
echo "Step 1: Building Docker images..."
docker-compose build

echo ""
echo "Step 2: Starting services..."
docker-compose up -d

echo ""
echo "Step 3: Waiting for services to be ready..."
sleep 10

# Wait for PostgreSQL
echo "Waiting for PostgreSQL..."
until docker-compose exec -T db pg_isready -U postgres > /dev/null 2>&1; do
    sleep 2
done
echo "PostgreSQL is ready!"

# Wait for Qdrant
echo "Waiting for Qdrant..."
until curl -s http://localhost:6333/health > /dev/null 2>&1; do
    sleep 2
done
echo "Qdrant is ready!"

echo ""
echo "Step 4: Running database migrations..."
docker-compose exec -T backend python manage.py migrate

echo ""
echo "Step 5: Loading embeddings into Qdrant..."
echo "This may take a few minutes for ~700K movie embeddings..."
docker-compose exec -T backend python manage.py load_embeddings

echo ""
echo "Step 6: Importing sample users..."
docker-compose exec -T backend python manage.py sync_users --limit-users 100

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Access the application at:"
echo "  - Frontend:        http://localhost:3000"
echo "  - Backend API:     http://localhost:8000/api/"
echo "  - Qdrant Dashboard: http://localhost:6333/dashboard"
echo ""
echo "To stop the services:"
echo "  docker-compose down"
echo ""
echo "To view logs:"
echo "  docker-compose logs -f"
echo ""
