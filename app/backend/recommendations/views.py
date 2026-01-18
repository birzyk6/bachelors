"""Recommendation views."""

import logging

from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from users.models import AppUser

from .services import QdrantService, RecommendationEngine

logger = logging.getLogger(__name__)


def get_engine():
    """Get or create recommendation engine instance."""
    return RecommendationEngine()


@api_view(["GET"])
def for_you(request):
    """
    Get personalized recommendations for a user.

    Query params:
    - user_id: User ID (required)
    - limit: Number of recommendations (default 20)
    """
    user_id = request.query_params.get("user_id")
    limit = int(request.query_params.get("limit", 20))

    if not user_id:
        return Response({"error": "user_id is required"}, status=status.HTTP_400_BAD_REQUEST)

    try:
        user = AppUser.objects.get(id=user_id)
    except AppUser.DoesNotExist:
        return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)

    engine = get_engine()
    recommendations = engine.get_for_you_recommendations(user, limit=limit)

    return Response(
        {
            "user_id": user_id,
            "recommendations": recommendations,
        }
    )


@api_view(["GET"])
def similar_movies(request):
    """
    Get movies similar to a given movie.

    Query params:
    - movie_id: TMDB movie ID (required)
    - limit: Number of results (default 10)
    """
    movie_id = request.query_params.get("movie_id")
    limit = int(request.query_params.get("limit", 10))

    if not movie_id:
        return Response({"error": "movie_id is required"}, status=status.HTTP_400_BAD_REQUEST)

    engine = get_engine()
    similar = engine.get_similar_movies(int(movie_id), limit=limit)

    return Response(
        {
            "movie_id": movie_id,
            "similar": similar,
        }
    )


@api_view(["GET"])
def because_you_watched(request):
    """
    Get recommendations based on user's recently watched movies.

    Query params:
    - user_id: User ID (required)
    - limit: Number of results (default 10)
    """
    user_id = request.query_params.get("user_id")
    limit = int(request.query_params.get("limit", 10))

    if not user_id:
        return Response({"error": "user_id is required"}, status=status.HTTP_400_BAD_REQUEST)

    try:
        user = AppUser.objects.get(id=user_id)
    except AppUser.DoesNotExist:
        return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)

    engine = get_engine()
    result = engine.get_because_you_watched(user, limit=limit)

    return Response(result)


@api_view(["GET"])
def by_genre(request):
    """
    Get recommendations filtered by genre.

    Query params:
    - genre: Genre name (required)
    - user_id: Optional user ID for exclusions
    - limit: Number of results (default 20)
    """
    genre = request.query_params.get("genre")
    user_id = request.query_params.get("user_id")
    limit = int(request.query_params.get("limit", 20))

    if not genre:
        return Response({"error": "genre is required"}, status=status.HTTP_400_BAD_REQUEST)

    user = None
    if user_id:
        try:
            user = AppUser.objects.get(id=user_id)
        except AppUser.DoesNotExist:
            pass

    engine = get_engine()
    recommendations = engine.get_by_genre(genre, user=user, limit=limit)

    return Response(
        {
            "genre": genre,
            "recommendations": recommendations,
        }
    )


@api_view(["GET"])
def trending(request):
    """Get trending/popular movies."""
    limit = int(request.query_params.get("limit", 20))

    engine = get_engine()
    movies = engine.get_trending(limit=limit)

    return Response(
        {
            "trending": movies,
        }
    )


@api_view(["GET"])
def seed_movies(request):
    """
    Get seed movies for onboarding new users.

    Returns diverse, popular movies for users to rate during setup.
    """
    count = int(request.query_params.get("count", 20))

    engine = get_engine()
    movies = engine.get_seed_movies_for_onboarding(count=count)

    return Response(
        {
            "movies": movies,
        }
    )


@api_view(["GET"])
def system_status(request):
    """Get recommendation system status."""
    try:
        qdrant = QdrantService()
        qdrant_info = qdrant.get_collection_info()
    except Exception as e:
        qdrant_info = {"error": str(e)}

    from .services import TFServingClient

    tf_client = TFServingClient()
    tf_status = tf_client.get_model_status()

    return Response(
        {
            "qdrant": qdrant_info,
            "tf_serving": tf_status,
        }
    )


@api_view(["POST"])
def load_embeddings(request):
    """
    Load embeddings into Qdrant.

    This is an admin endpoint to initialize the vector database.
    """
    from pathlib import Path

    from django.conf import settings

    embeddings_path = request.data.get("embeddings_path")
    if not embeddings_path:
        embeddings_path = Path(settings.EMBEDDINGS_PATH) / "combined_movie_embeddings.npy"
    else:
        embeddings_path = Path(embeddings_path)

    if not embeddings_path.exists():
        return Response({"error": f"Embeddings file not found: {embeddings_path}"}, status=status.HTTP_404_NOT_FOUND)

    try:
        qdrant = QdrantService()
        count = qdrant.load_embeddings(str(embeddings_path))
        return Response(
            {
                "status": "success",
                "loaded": count,
            }
        )
    except Exception as e:
        logger.exception("Failed to load embeddings")
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
