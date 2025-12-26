# Movie Recommendation App - Implementation Plan

## Overview

A full-stack movie recommendation application using your trained Two-Tower model, containerized with Docker. The app allows users to either impersonate existing users or go through a cold-start onboarding flow.

## Architecture

```
                                    +------------------+
                                    |   Next.js App    |
                                    |   (Frontend)     |
                                    +--------+---------+
                                             |
                                             v
+------------------+              +------------------+              +------------------+
|  TensorFlow      |<----------->|   Django API     |<----------->|   PostgreSQL     |
|  Serving         |              |   (Backend)      |              |   (Database)     |
|  (User Tower)    |              +--------+---------+              +------------------+
+------------------+                       |
                                           v
                                +------------------+              +------------------+
                                |   Qdrant         |              |   TMDB API       |
                                |   (Vectors)      |              |   (Metadata)     |
                                +------------------+              +------------------+
```

## Services (Docker Compose)

| Service | Port | Description |
|---------|------|-------------|
| `frontend` | 3000 | Next.js React application |
| `backend` | 8000 | Django REST API |
| `db` | 5432 | PostgreSQL database |
| `qdrant` | 6333 | Qdrant vector database |
| `tf-serving` | 8501 | TensorFlow Serving (user tower) |

## Data Flow

### 1. Cold-Start User Flow
1. User visits app -> No user selected
2. Onboarding wizard:
   - Select favorite genres (Action, Comedy, Drama, etc.)
   - Rate 5-10 seed movies
3. Backend creates temporary user profile
4. Generate user embedding using content-only tower
5. Query Qdrant for similar movies

### 2. Existing User Flow
1. User selects from dropdown (user impersonation)
2. Load user's rating history from PostgreSQL
3. Generate user embedding via TensorFlow Serving
4. Query Qdrant for recommendations

### 3. Recommendation Types
- **For You**: User embedding -> Qdrant ANN search
- **Similar Users**: Find users with similar embeddings, get their top movies
- **By Genre/Keyword**: Filter Qdrant results by metadata
- **Recently Popular**: Aggregate recent ratings
- **Because You Watched X**: Movie embedding -> Qdrant similarity

## Project Structure

```
app/
├── docker-compose.yml
├── .env.example
├── frontend/
│   ├── Dockerfile
│   ├── package.json
│   ├── next.config.js
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx
│   │   │   ├── page.tsx
│   │   │   └── globals.css
│   │   ├── components/
│   │   │   ├── MovieCard.tsx
│   │   │   ├── MovieModal.tsx
│   │   │   ├── MovieGrid.tsx
│   │   │   ├── UserSelector.tsx
│   │   │   ├── OnboardingWizard.tsx
│   │   │   ├── RecommendationSection.tsx
│   │   │   └── Header.tsx
│   │   ├── hooks/
│   │   │   └── useRecommendations.ts
│   │   └── lib/
│   │       └── api.ts
│   └── tailwind.config.js
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── manage.py
│   ├── config/
│   │   ├── settings.py
│   │   ├── urls.py
│   │   └── wsgi.py
│   ├── core/
│   │   ├── models.py
│   │   ├── serializers.py
│   │   ├── views.py
│   │   └── urls.py
│   ├── recommendations/
│   │   ├── models.py
│   │   ├── views.py
│   │   ├── urls.py
│   │   ├── services/
│   │   │   ├── qdrant_service.py
│   │   │   ├── tf_serving_client.py
│   │   │   ├── tmdb_client.py
│   │   │   └── recommendation_engine.py
│   │   └── management/
│   │       └── commands/
│   │           ├── load_embeddings.py
│   │           └── sync_movies.py
│   └── users/
│       ├── models.py
│       ├── views.py
│       └── urls.py
├── qdrant/
│   └── init-collections.sh
├── tf-serving/
│   ├── Dockerfile
│   └── models.config
└── scripts/
    ├── init-db.sql
    └── load-data.py
```

## Database Schema (PostgreSQL)

```sql
-- Users table (for cold-start and impersonation)
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    movielens_user_id INTEGER UNIQUE,  -- NULL for cold-start users
    created_at TIMESTAMP DEFAULT NOW(),
    is_cold_start BOOLEAN DEFAULT FALSE
);

-- User preferences (genres, keywords)
CREATE TABLE user_preferences (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    preference_type VARCHAR(50),  -- 'genre', 'keyword'
    preference_value VARCHAR(255),
    weight FLOAT DEFAULT 1.0
);

-- Ratings
CREATE TABLE ratings (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    tmdb_id INTEGER NOT NULL,
    rating FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Movies cache (metadata from TMDB)
CREATE TABLE movies (
    tmdb_id INTEGER PRIMARY KEY,
    title VARCHAR(500),
    overview TEXT,
    poster_path VARCHAR(255),
    backdrop_path VARCHAR(255),
    release_date DATE,
    vote_average FLOAT,
    genres JSONB,
    keywords JSONB,
    runtime INTEGER,
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Watch history
CREATE TABLE watch_history (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    tmdb_id INTEGER NOT NULL,
    watched_at TIMESTAMP DEFAULT NOW()
);
```

## API Endpoints (Django)

### Users
- `GET /api/users/` - List available users for impersonation
- `POST /api/users/cold-start/` - Create cold-start user with preferences
- `GET /api/users/{id}/` - Get user details

### Recommendations
- `GET /api/recommendations/for-you/?user_id=X` - Personalized recommendations
- `GET /api/recommendations/similar-users/?user_id=X` - Based on similar users
- `GET /api/recommendations/by-genre/?genre=Action` - Filter by genre
- `GET /api/recommendations/because-watched/?movie_id=X` - Similar to movie
- `GET /api/recommendations/trending/` - Recently popular

### Movies
- `GET /api/movies/{tmdb_id}/` - Get movie details (fetches from TMDB API)
- `GET /api/movies/search/?q=inception` - Search movies
- `POST /api/movies/{tmdb_id}/rate/` - Rate a movie

### Onboarding
- `GET /api/onboarding/genres/` - Get available genres
- `GET /api/onboarding/seed-movies/` - Get popular movies for initial rating
- `POST /api/onboarding/complete/` - Submit onboarding preferences

## Key Implementation Details

### 1. Qdrant Collection Setup
```python
# Collection for movie embeddings
client.create_collection(
    collection_name="movies",
    vectors_config=VectorParams(size=128, distance=Distance.DOT),
)

# Payload includes: tmdb_id, title, genres, keywords, year
```

### 2. TensorFlow Serving Configuration
```
model_config_list {
  config {
    name: 'user_tower'
    base_path: '/models/user_tower'
    model_platform: 'tensorflow'
  }
}
```

### 3. User Embedding Generation
- **Existing users**: Call TF Serving with user_id index
- **Cold-start users**:
  1. Get BERT embeddings for preferred genres/movies
  2. Average them to create pseudo-user profile
  3. Use content-only tower to project to embedding space

### 4. TMDB API Integration
- Fetch movie metadata on-demand
- Cache in PostgreSQL to reduce API calls
- Use backdrop/poster images for UI

## Environment Variables

```env
# Database
POSTGRES_DB=movies_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

# TMDB
TMDB_API_KEY=your_tmdb_api_key

# Services
QDRANT_HOST=qdrant
QDRANT_PORT=6333
TF_SERVING_HOST=tf-serving
TF_SERVING_PORT=8501

# Django
SECRET_KEY=your_secret_key
DEBUG=False
```

## Frontend Features

### Pages/Views
1. **Home** - User selector or onboarding prompt
2. **Browse** - Movie grid with recommendation sections
3. **Movie Modal** - Detailed view with:
   - Poster, backdrop, title, year
   - Overview, genres, keywords
   - Rating input
   - Similar movies carousel

### UI Components
- Movie cards with hover effects
- Horizontal carousels for recommendation sections
- Modal overlay for movie details
- Genre filter chips
- User dropdown selector
- Onboarding stepper wizard

## Implementation Order

1. **Phase 1: Infrastructure**
   - Docker Compose setup
   - PostgreSQL initialization
   - Qdrant collection creation
   - TensorFlow Serving configuration

2. **Phase 2: Backend Core**
   - Django project setup
   - Database models
   - Qdrant service (load embeddings)
   - TF Serving client
   - Basic recommendation API

3. **Phase 3: TMDB Integration**
   - TMDB API client
   - Movie metadata caching
   - Search functionality

4. **Phase 4: Frontend**
   - Next.js project setup
   - Component library
   - API integration
   - Recommendation sections

5. **Phase 5: Cold-Start**
   - Onboarding wizard
   - Preference-based embedding
   - Cold-start recommendations

6. **Phase 6: Polish**
   - Error handling
   - Loading states
   - Responsive design
   - Documentation

## Data Loading Strategy

1. Load `combined_movie_embeddings.npy` into Qdrant with metadata
2. Import MovieLens users (sample of 1000 for demo)
3. Import ratings for those users
4. Pre-cache popular movie metadata from TMDB

## Notes for Thesis

This application demonstrates:
- **Two-Tower Architecture**: Separate user/item embeddings for efficient retrieval
- **Cold-Start Problem**: Content-only tower enables recommendations for new users
- **Approximate Nearest Neighbor**: Qdrant enables sub-millisecond retrieval from 700K+ movies
- **Microservices Architecture**: Each component is independently scalable
- **Real-world Integration**: TMDB API provides rich movie metadata
