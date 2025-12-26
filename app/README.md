# Movie Recommendation System

A full-stack movie recommendation application powered by a Two-Tower neural network architecture, designed as part of a bachelor's thesis on modern recommendation systems.

## System Architecture

The application follows a microservices architecture with the following components:

```
+-------------------+     +------------------+     +------------------+
|                   |     |                  |     |                  |
|   Next.js         |---->|   Django         |---->|   PostgreSQL     |
|   Frontend        |     |   REST API       |     |   Database       |
|   (Port 3000)     |     |   (Port 8000)    |     |   (Port 5432)    |
|                   |     |                  |     |                  |
+-------------------+     +--------+---------+     +------------------+
                                  |
                    +-------------+-------------+
                    |                           |
          +---------v---------+       +---------v---------+
          |                   |       |                   |
          |   Qdrant          |       |   TensorFlow      |
          |   Vector DB       |       |   Serving         |
          |   (Port 6333)     |       |   (Port 8501)     |
          |                   |       |                   |
          +-------------------+       +-------------------+
```

### Components

| Service | Technology | Purpose |
|---------|-----------|---------|
| Frontend | Next.js 14, React, Tailwind CSS | User interface for browsing and discovering movies |
| Backend | Django 4.2, Django REST Framework | REST API, business logic, TMDB integration |
| Database | PostgreSQL 15 | User data, ratings, cached movie metadata |
| Vector DB | Qdrant | Movie embeddings for similarity search (ANN) |
| Model Serving | TensorFlow Serving | User embedding generation from Two-Tower model |

## Two-Tower Model Integration

### Architecture Overview

The recommendation system uses a **Two-Tower (Dual Encoder)** architecture:

1. **User Tower**: Generates user embeddings from user ID
   - Input: User ID (integer)
   - Output: 128-dimensional L2-normalized embedding
   - Served via TensorFlow Serving for real-time inference

2. **Movie Tower**: Generates movie embeddings from movie ID + BERT text features
   - Input: Movie ID + 512-dim BERT embedding of movie overview
   - Output: 128-dimensional L2-normalized embedding
   - Pre-computed and stored in Qdrant for efficient retrieval

3. **Content-Only Tower**: For cold-start movies
   - Input: 512-dim BERT embedding only
   - Output: 128-dimensional embedding
   - Enables recommendations for new movies without interaction history

### Recommendation Flow

```
User Request
     |
     v
+----+----+
|  User   |
|  Tower  |  (TensorFlow Serving)
+---------+
     |
     v
User Embedding (128-dim)
     |
     v
+----+----+
| Qdrant  |  Approximate Nearest Neighbor Search
|   ANN   |
+---------+
     |
     v
Top-K Similar Movie IDs
     |
     v
+----+----+
|  TMDB   |  Fetch movie metadata
|   API   |
+---------+
     |
     v
Recommendations with Metadata
```

## Features

### User Modes

1. **User Impersonation**: Select an existing MovieLens user to see their personalized recommendations
2. **Cold-Start Onboarding**: New users can:
   - Select preferred genres
   - Rate seed movies
   - Get instant recommendations based on initial preferences

### Recommendation Types

- **For You**: Personalized recommendations based on user embedding similarity
- **Because You Watched**: Movies similar to recently rated titles
- **By Genre**: Genre-filtered recommendations
- **Trending**: Popular movies across all users

### Movie Details

Each movie displays:
- Poster and backdrop images (from TMDB)
- Title, release year, runtime
- Genre tags
- Vote average and count
- Full overview/synopsis
- Similar movies carousel
- User rating input (1-5 stars)

## Quick Start

### Prerequisites

- Docker and Docker Compose
- TMDB API key (get one at https://www.themoviedb.org/settings/api)

### Setup

1. **Clone and navigate to the app directory**:
   ```bash
   cd app
   ```

2. **Create environment file**:
   ```bash
   cp .env.example .env
   # Edit .env and add your TMDB_API_KEY
   ```

3. **Start all services**:
   ```bash
   docker-compose up -d
   ```

4. **Load movie embeddings into Qdrant** (first time only):
   ```bash
   docker-compose exec backend python manage.py load_embeddings
   ```

5. **Import sample users** (optional, for impersonation):
   ```bash
   docker-compose exec backend python manage.py sync_users --limit-users 100
   ```

6. **Access the application**:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000/api/
   - Qdrant Dashboard: http://localhost:6333/dashboard

## API Endpoints

### Users
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/users/` | List users for impersonation |
| POST | `/api/users/cold_start/` | Create cold-start user |
| GET | `/api/users/{id}/ratings/` | Get user's ratings |
| POST | `/api/users/{id}/rate/` | Rate a movie |

### Recommendations
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/recommendations/for-you/` | Personalized recommendations |
| GET | `/api/recommendations/similar/` | Similar movies |
| GET | `/api/recommendations/by-genre/` | Genre-filtered recommendations |
| GET | `/api/recommendations/trending/` | Trending movies |
| GET | `/api/recommendations/seed-movies/` | Onboarding seed movies |

### Movies
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/movies/{tmdb_id}/` | Get movie details |
| GET | `/api/movies/search/` | Search movies |

## Data Pipeline

### Embeddings

The system uses pre-computed embeddings from the trained Two-Tower model:

- **MovieLens Movies** (~80K): Full Two-Tower embeddings (user interactions + content)
- **TMDB Movies** (~616K): Content-only tower embeddings (cold-start)
- **Total**: ~696K movies with 128-dimensional embeddings

### Movie Metadata

Movie metadata is fetched on-demand from TMDB API and cached in PostgreSQL:
- Poster and backdrop images
- Title, overview, tagline
- Genres, keywords
- Runtime, release date
- Vote statistics

## Technical Highlights

### Vector Similarity Search

- Uses **Qdrant** for Approximate Nearest Neighbor (ANN) search
- **Dot product** distance metric (embeddings are L2-normalized)
- Sub-millisecond query times for 700K+ vectors

### Cold-Start Handling

For new users without rating history:
1. Collect genre preferences
2. Rate 5+ seed movies
3. Compute weighted average of rated movie embeddings
4. Use as pseudo-user embedding for Qdrant search

### Scalability

- TensorFlow Serving enables horizontal scaling of inference
- Qdrant supports distributed deployment
- Stateless backend allows multiple replicas

## Development

### Local Development (without Docker)

```bash
# Backend
cd backend
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver

# Frontend
cd frontend
npm install
npm run dev
```

### Running Tests

```bash
# Backend tests
docker-compose exec backend python manage.py test

# Frontend tests
docker-compose exec frontend npm test
```

## Files Structure

```
app/
├── docker-compose.yml          # Service orchestration
├── .env.example                # Environment template
├── frontend/                   # Next.js application
│   ├── src/
│   │   ├── app/               # Next.js App Router pages
│   │   ├── components/        # React components
│   │   └── lib/               # API client, utilities
│   └── Dockerfile
├── backend/                    # Django application
│   ├── config/                # Django settings
│   ├── core/                  # Movie models, views
│   ├── users/                 # User models, auth
│   ├── recommendations/       # Rec engine, services
│   │   └── services/
│   │       ├── qdrant_service.py
│   │       ├── tf_serving_client.py
│   │       ├── tmdb_client.py
│   │       └── recommendation_engine.py
│   └── Dockerfile
├── tf-serving/                # TensorFlow Serving config
└── scripts/                   # Database init scripts
```

## Thesis Context

This application demonstrates several key concepts in modern recommendation systems:

1. **Two-Tower Architecture**: Efficient retrieval through separate user/item encoders
2. **Cold-Start Problem**: Content-based embeddings for new users/items
3. **Approximate Nearest Neighbor**: Sub-linear time complexity for similarity search
4. **Microservices**: Scalable, maintainable system design
5. **Real-time ML Serving**: TensorFlow Serving for production inference

## License

This project is part of a bachelor's thesis and is provided for educational purposes.

## Acknowledgments

- MovieLens dataset by GroupLens Research
- TMDB API for movie metadata
- TensorFlow Recommenders library
