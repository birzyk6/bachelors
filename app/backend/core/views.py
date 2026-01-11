"""Core views for movie recommendation app."""

from recommendations.services.tmdb_client import TMDBClient
from rest_framework import status, viewsets
from rest_framework.decorators import action, api_view
from rest_framework.response import Response

from .models import Genre, Movie
from .serializers import GenreSerializer, MovieListSerializer, MovieSerializer


class MovieViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for movie operations."""

    queryset = Movie.objects.all()
    serializer_class = MovieSerializer
    lookup_field = "tmdb_id"

    def get_serializer_class(self):
        if self.action == "list":
            return MovieListSerializer
        return MovieSerializer

    def retrieve(self, request, tmdb_id=None):
        """Get movie details, fetching from TMDB if not cached."""
        try:
            movie = Movie.objects.get(tmdb_id=tmdb_id)
            serializer = self.get_serializer(movie)
            return Response(serializer.data)
        except Movie.DoesNotExist:
            tmdb = TMDBClient()
            movie_data = tmdb.get_movie(int(tmdb_id))
            if movie_data:
                movie, _ = Movie.objects.get_or_create(
                    tmdb_id=movie_data["id"],
                    defaults={
                        "title": movie_data.get("title", ""),
                        "original_title": movie_data.get("original_title"),
                        "overview": movie_data.get("overview"),
                        "poster_path": movie_data.get("poster_path"),
                        "backdrop_path": movie_data.get("backdrop_path"),
                        "release_date": movie_data.get("release_date") or None,
                        "vote_average": movie_data.get("vote_average", 0),
                        "vote_count": movie_data.get("vote_count", 0),
                        "genres": [g["name"] for g in movie_data.get("genres", [])],
                        "keywords": [k["name"] for k in movie_data.get("keywords", {}).get("keywords", [])],
                        "runtime": movie_data.get("runtime"),
                        "tagline": movie_data.get("tagline"),
                        "popularity": movie_data.get("popularity", 0),
                    },
                )
                serializer = self.get_serializer(movie)
                return Response(serializer.data)
            return Response({"error": "Movie not found"}, status=status.HTTP_404_NOT_FOUND)

    @action(detail=False, methods=["get"])
    def search(self, request):
        """Search movies via TMDB."""
        query = request.query_params.get("q", "")
        if not query:
            return Response({"results": []})

        tmdb = TMDBClient()
        results = tmdb.search_movies(query)
        return Response({"results": results})

    @action(detail=False, methods=["get"])
    def popular(self, request):
        """Get popular movies."""
        movies = Movie.objects.order_by("-popularity")[:20]
        serializer = MovieListSerializer(movies, many=True)
        return Response({"movies": serializer.data})

    @action(detail=False, methods=["post"])
    def batch(self, request):
        """
        Get multiple movies by TMDB IDs in a single request.

        POST body: {"ids": [123, 456, 789]}
        Returns cached movies only - does not fetch from TMDB.
        """
        tmdb_ids = request.data.get("ids", [])
        if not tmdb_ids:
            return Response({"movies": []})

        tmdb_ids = tmdb_ids[:100]

        movies = Movie.objects.filter(tmdb_id__in=tmdb_ids)
        serializer = MovieListSerializer(movies, many=True)

        movies_dict = {m["tmdb_id"]: m for m in serializer.data}

        return Response({"movies": movies_dict})


class GenreViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for genres."""

    queryset = Genre.objects.all()
    serializer_class = GenreSerializer


@api_view(["GET"])
def health_check(request):
    """Health check endpoint."""
    return Response({"status": "healthy"})


@api_view(["GET"])
def available_genres(request):
    """Get all available genres for filtering."""
    genres = [
        {"id": 28, "name": "Action"},
        {"id": 12, "name": "Adventure"},
        {"id": 16, "name": "Animation"},
        {"id": 35, "name": "Comedy"},
        {"id": 80, "name": "Crime"},
        {"id": 99, "name": "Documentary"},
        {"id": 18, "name": "Drama"},
        {"id": 10751, "name": "Family"},
        {"id": 14, "name": "Fantasy"},
        {"id": 36, "name": "History"},
        {"id": 27, "name": "Horror"},
        {"id": 10402, "name": "Music"},
        {"id": 9648, "name": "Mystery"},
        {"id": 10749, "name": "Romance"},
        {"id": 878, "name": "Science Fiction"},
        {"id": 10770, "name": "TV Movie"},
        {"id": 53, "name": "Thriller"},
        {"id": 10752, "name": "War"},
        {"id": 37, "name": "Western"},
    ]
    return Response({"genres": genres})
