# Setup Guide

This guide provides detailed instructions for setting up and running the movie recommendation system.

## Quick Start

### 1. Clone Repository

```bash
git clone git@github.com:birzyk6/bachelors.git
cd bachelors
```

### 2. Start Services

Start all services using Docker Compose:

```bash
docker-compose up -d
```

This will start:

-   PostgreSQL database (port 5432)
-   Qdrant vector database (ports 6333, 6334)
-   TensorFlow Serving (ports 8500, 8501)
-   Django backend (port 8000)
-   Next.js frontend (port 3000)

### 3. Initialize Database

Wait for services to be healthy, then run migrations:

```bash
docker-compose exec backend python manage.py migrate
```

### 4. Load Movie Embeddings

Load pre-computed movie embeddings into Qdrant:

```bash
docker-compose exec backend python manage.py load_embeddings
```

This step is required for recommendations to work. It may take several minutes depending on the number of embeddings.

### 5. Import Sample Users (Optional)

For testing user impersonation, import sample users from MovieLens:

```bash
docker-compose exec backend python manage.py sync_users --limit-users 100
```

This imports 100 users with their ratings. Adjust `--limit-users` as needed.
