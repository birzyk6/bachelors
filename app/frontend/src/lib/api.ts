import axios from 'axios';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const client = axios.create({
  baseURL: `${API_BASE}/api`,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const api = {
  // Users
  async getUsers(page = 1, limit = 50, search = '', minRatings = 0, sortField = 'ratings_count', sortDirection: 'asc' | 'desc' = 'desc') {
    const response = await client.get('/users/', {
      params: { page, limit, search, min_ratings: minRatings, sort_by: sortField, sort_dir: sortDirection },
    });
    return response.data;
  },

  async getUser(userId: number) {
    const response = await client.get(`/users/${userId}/`);
    return response.data;
  },

  async createColdStartUser(genres: string[], initialRatings: {tmdb_id: number; rating: number}[]) {
    const response = await client.post('/users/cold_start/', {
      genres,
      initial_ratings: initialRatings,
    });
    return response.data;
  },

  async getUserRatings(userId: number, page = 1, limit = 50, includeMovies = false) {
    const response = await client.get(`/users/${userId}/ratings/`, {
      params: { page, limit, include_movies: includeMovies },
    });
    return response.data;
  },

  async getUserStatistics(userId: number) {
    const response = await client.get(`/users/${userId}/statistics/`);
    return response.data;
  },

  async getUserRating(userId: number, tmdbId: number) {
    const response = await client.get(`/users/${userId}/rating/${tmdbId}/`);
    return response.data;
  },

  async getMoviesBatch(tmdbIds: number[]) {
    const response = await client.post('/movies/batch/', { ids: tmdbIds });
    return response.data;
  },

  async rateMovie(userId: number, tmdbId: number, rating: number) {
    const response = await client.post(`/users/${userId}/rate/`, {
      tmdb_id: tmdbId,
      rating,
    });
    return response.data;
  },

  // Movies
  async getMovie(tmdbId: number) {
    const response = await client.get(`/movies/${tmdbId}/`);
    return response.data;
  },

  async searchMovies(query: string) {
    const response = await client.get('/movies/search/', {
      params: { q: query },
    });
    return response.data;
  },

  async getPopularMovies() {
    const response = await client.get('/movies/popular/');
    return response.data;
  },

  // Recommendations
  async getForYou(userId: number, limit = 20) {
    const response = await client.get('/recommendations/for-you/', {
      params: { user_id: userId, limit },
    });
    return response.data;
  },

  async getSimilarMovies(movieId: number, limit = 10) {
    const response = await client.get('/recommendations/similar/', {
      params: { movie_id: movieId, limit },
    });
    return response.data;
  },

  async getBecauseYouWatched(userId: number, limit = 10) {
    const response = await client.get('/recommendations/because-watched/', {
      params: { user_id: userId, limit },
    });
    return response.data;
  },

  async getByGenre(genre: string, userId?: number, limit = 20) {
    const response = await client.get('/recommendations/by-genre/', {
      params: { genre, user_id: userId, limit },
    });
    return response.data;
  },

  async getTrending(limit = 20) {
    const response = await client.get('/recommendations/trending/', {
      params: { limit },
    });
    return response.data;
  },

  async getSeedMovies(count = 20) {
    const response = await client.get('/recommendations/seed-movies/', {
      params: { count },
    });
    return response.data;
  },

  // System
  async getGenres() {
    const response = await client.get('/genres/list/');
    return response.data;
  },

  async getSystemStatus() {
    const response = await client.get('/recommendations/status/');
    return response.data;
  },
};

export default api;
