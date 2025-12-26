-- Initialize database schema for movie recommendation app

-- Users table (for cold-start and impersonation)
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    movielens_user_id INTEGER UNIQUE,
    username VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    is_cold_start BOOLEAN DEFAULT FALSE
);

-- User preferences (genres, keywords)
CREATE TABLE IF NOT EXISTS user_preferences (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    preference_type VARCHAR(50),
    preference_value VARCHAR(255),
    weight FLOAT DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Ratings
CREATE TABLE IF NOT EXISTS ratings (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    tmdb_id INTEGER NOT NULL,
    rating FLOAT NOT NULL CHECK (rating >= 0.5 AND rating <= 5.0),
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(user_id, tmdb_id)
);

-- Movies cache (metadata from TMDB)
CREATE TABLE IF NOT EXISTS movies (
    tmdb_id INTEGER PRIMARY KEY,
    title VARCHAR(500) NOT NULL,
    original_title VARCHAR(500),
    overview TEXT,
    poster_path VARCHAR(255),
    backdrop_path VARCHAR(255),
    release_date DATE,
    vote_average FLOAT,
    vote_count INTEGER,
    genres JSONB DEFAULT '[]',
    keywords JSONB DEFAULT '[]',
    runtime INTEGER,
    tagline TEXT,
    popularity FLOAT,
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Watch history
CREATE TABLE IF NOT EXISTS watch_history (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    tmdb_id INTEGER NOT NULL,
    watched_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_ratings_user_id ON ratings(user_id);
CREATE INDEX IF NOT EXISTS idx_ratings_tmdb_id ON ratings(tmdb_id);
CREATE INDEX IF NOT EXISTS idx_watch_history_user_id ON watch_history(user_id);
CREATE INDEX IF NOT EXISTS idx_user_preferences_user_id ON user_preferences(user_id);
CREATE INDEX IF NOT EXISTS idx_movies_vote_average ON movies(vote_average DESC);
CREATE INDEX IF NOT EXISTS idx_movies_popularity ON movies(popularity DESC);
