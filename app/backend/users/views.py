"""User views for movie recommendation app."""

from django.db.models import Q
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from .models import AppUser, Rating, UserPreference, WatchHistory
from .serializers import (
    AppUserListSerializer,
    AppUserSerializer,
    ColdStartUserCreateSerializer,
    RateMovieSerializer,
    RatingSerializer,
    WatchHistorySerializer,
)


class UserViewSet(viewsets.ModelViewSet):
    """ViewSet for user operations."""

    queryset = AppUser.objects.all()
    serializer_class = AppUserSerializer

    def get_serializer_class(self):
        if self.action == "list":
            return AppUserListSerializer
        return AppUserSerializer

    def list(self, request):
        """List users available for impersonation with pagination, search, filters, and sorting."""
        from django.core.cache import cache
        from django.db import connection

        page = int(request.query_params.get("page", 1))
        limit = min(int(request.query_params.get("limit", 50)), 100)
        search = request.query_params.get("search", "").strip()
        min_ratings = int(request.query_params.get("min_ratings", 0))
        sort_by = request.query_params.get("sort_by", "ratings_count")
        sort_dir = request.query_params.get("sort_dir", "desc")

        cache_key = f"users_list_{min_ratings}_{sort_by}_{sort_dir}_{search}_{page}_{limit}"
        cached = cache.get(cache_key)
        if cached:
            return Response(cached)

        offset = (page - 1) * limit
        order_dir = "DESC" if sort_dir == "desc" else "ASC"

        search_condition = ""
        search_params = []
        if search:
            try:
                user_id = int(search)
                search_condition = "AND (u.movielens_user_id = %s OR (u.movielens_user_id >= %s AND u.movielens_user_id < %s))"
                search_params = [user_id, user_id, user_id + 100]
            except ValueError:
                pass

        sql = f"""
            SELECT u.id, u.movielens_user_id, u.username, rc.rating_count
            FROM app_users u
            INNER JOIN (
                SELECT user_id, COUNT(*) as rating_count
                FROM user_ratings
                GROUP BY user_id
                HAVING COUNT(*) >= %s
            ) rc ON u.id = rc.user_id
            WHERE u.movielens_user_id IS NOT NULL
            {search_condition}
            ORDER BY rc.rating_count {order_dir}
            LIMIT %s OFFSET %s
        """

        count_cache_key = f"users_count_{min_ratings}_{search}"
        total = cache.get(count_cache_key)
        if total is None:
            count_sql = f"""
                SELECT COUNT(*) FROM app_users u
                INNER JOIN (
                    SELECT user_id FROM user_ratings
                    GROUP BY user_id HAVING COUNT(*) >= %s
                ) rc ON u.id = rc.user_id
                WHERE u.movielens_user_id IS NOT NULL
                {search_condition}
            """
            with connection.cursor() as cursor:
                cursor.execute(count_sql, [min_ratings] + search_params)
                total = cursor.fetchone()[0]
            cache.set(count_cache_key, total, 300)

        with connection.cursor() as cursor:
            cursor.execute(sql, [min_ratings] + search_params + [limit, offset])
            rows = cursor.fetchall()

        users_data = []
        for row in rows:
            user_id, movielens_id, username, rating_count = row
            if username:
                display_name = username
            elif movielens_id:
                display_name = f"User #{movielens_id}"
            else:
                display_name = f"Guest #{user_id}"

            users_data.append(
                {
                    "id": user_id,
                    "movielens_user_id": movielens_id,
                    "display_name": display_name,
                    "ratings_count": rating_count,
                }
            )

        result = {
            "users": users_data,
            "total": total,
            "page": page,
            "limit": limit,
            "has_more": offset + limit < total,
        }

        cache.set(cache_key, result, 60)

        return Response(result)

    @action(detail=False, methods=["post"])
    def cold_start(self, request):
        """Create a new cold-start user with preferences."""
        serializer = ColdStartUserCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        user = AppUser.objects.create(is_cold_start=True)

        for genre in serializer.validated_data.get("genres", []):
            UserPreference.objects.create(user=user, preference_type="genre", preference_value=genre, weight=1.0)

        for rating_data in serializer.validated_data.get("initial_ratings", []):
            Rating.objects.create(user=user, tmdb_id=rating_data["tmdb_id"], rating=rating_data["rating"])

        return Response(AppUserSerializer(user).data, status=status.HTTP_201_CREATED)

    @action(detail=True, methods=["get"])
    def ratings(self, request, pk=None):
        """Get user's ratings with optional pagination and movie info."""
        from core.models import Movie

        user = self.get_object()

        page = int(request.query_params.get("page", 1))
        limit = min(int(request.query_params.get("limit", 50)), 100)
        include_movies = request.query_params.get("include_movies", "false").lower() == "true"

        ratings = user.ratings.all().order_by("-created_at")
        total = ratings.count()

        offset = (page - 1) * limit
        paginated_ratings = ratings[offset : offset + limit]

        serializer = RatingSerializer(paginated_ratings, many=True)
        ratings_data = serializer.data

        if include_movies:
            tmdb_ids = [r["tmdb_id"] for r in ratings_data]
            movies = Movie.objects.filter(tmdb_id__in=tmdb_ids)
            movies_dict = {
                m.tmdb_id: {
                    "title": m.title,
                    "poster_url": f"https://image.tmdb.org/t/p/w342{m.poster_path}" if m.poster_path else None,
                    "year": m.release_date.year if m.release_date else None,
                    "genres": m.genres or [],
                }
                for m in movies
            }
            for rating in ratings_data:
                rating["movie"] = movies_dict.get(rating["tmdb_id"])

        return Response(
            {
                "ratings": ratings_data,
                "total": total,
                "page": page,
                "limit": limit,
                "has_more": offset + limit < total,
            }
        )

    @action(detail=True, methods=["get"], url_path="rating/(?P<tmdb_id>[0-9]+)")
    def get_rating(self, request, pk=None, tmdb_id=None):
        """Get user's rating for a specific movie."""
        user = self.get_object()
        try:
            rating = user.ratings.get(tmdb_id=tmdb_id)
            return Response({"rating": rating.rating, "created_at": rating.created_at})
        except Rating.DoesNotExist:
            return Response({"rating": None})

    @action(detail=True, methods=["post"])
    def rate(self, request, pk=None):
        """Rate a movie."""
        user = self.get_object()
        tmdb_id = request.data.get("tmdb_id")

        if not tmdb_id:
            return Response({"error": "tmdb_id is required"}, status=status.HTTP_400_BAD_REQUEST)

        serializer = RateMovieSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        rating, created = Rating.objects.update_or_create(
            user=user, tmdb_id=tmdb_id, defaults={"rating": serializer.validated_data["rating"]}
        )

        return Response(RatingSerializer(rating).data)

    @action(detail=True, methods=["get"])
    def history(self, request, pk=None):
        """Get user's watch history."""
        user = self.get_object()
        history = user.watch_history.all()[:50]
        serializer = WatchHistorySerializer(history, many=True)
        return Response({"history": serializer.data})

    @action(detail=True, methods=["post"])
    def add_to_history(self, request, pk=None):
        """Add movie to watch history."""
        user = self.get_object()
        tmdb_id = request.data.get("tmdb_id")

        if not tmdb_id:
            return Response({"error": "tmdb_id is required"}, status=status.HTTP_400_BAD_REQUEST)

        watch = WatchHistory.objects.create(user=user, tmdb_id=tmdb_id)
        return Response(WatchHistorySerializer(watch).data)

    @action(detail=True, methods=["get"])
    def preferences(self, request, pk=None):
        """Get user's preferences."""
        user = self.get_object()
        preferences = user.preferences.all()

        result = {
            "genres": [],
            "keywords": [],
        }

        for pref in preferences:
            if pref.preference_type == "genre":
                result["genres"].append({"value": pref.preference_value, "weight": pref.weight})
            elif pref.preference_type == "keyword":
                result["keywords"].append({"value": pref.preference_value, "weight": pref.weight})

        return Response(result)

    @action(detail=True, methods=["get"])
    def statistics(self, request, pk=None):
        """Get statistics for a user based on their ratings."""
        import datetime
        from collections import Counter

        from core.models import Movie
        from django.db.models import Avg, Count, Max, Min
        from django.db.models.functions import ExtractYear, TruncMonth

        user = self.get_object()
        ratings = user.ratings.all()

        total_ratings = ratings.count()
        if total_ratings == 0:
            return Response(
                {
                    "total_ratings": 0,
                    "average_rating": 0,
                    "rating_distribution": [],
                    "genre_distribution": [],
                    "decade_distribution": [],
                    "monthly_activity": [],
                    "yearly_activity": [],
                    "insights": {
                        "favorite_genre": None,
                        "avg_movie_year": None,
                        "most_active_month": None,
                        "rating_style": "No ratings yet",
                        "liked_count": 0,
                        "disliked_count": 0,
                        "neutral_count": 0,
                    },
                }
            )

        stats = ratings.aggregate(
            avg_rating=Avg("rating"),
            min_rating=Min("rating"),
            max_rating=Max("rating"),
        )

        rating_counts = ratings.values("rating").annotate(count=Count("rating")).order_by("rating")
        rating_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        rating_distribution = [{"rating": str(val), "count": 0} for val in rating_values]
        for rc in rating_counts:
            rating_val = round(rc["rating"] * 2) / 2
            if 0.5 <= rating_val <= 5.0:
                idx = int((rating_val - 0.5) * 2)
                if 0 <= idx < len(rating_distribution):
                    rating_distribution[idx]["count"] = rc["count"]

        tmdb_ids = list(ratings.values_list("tmdb_id", flat=True))
        movies = Movie.objects.filter(tmdb_id__in=tmdb_ids)
        movies_dict = {m.tmdb_id: m for m in movies}

        genre_counter = Counter()
        genre_ratings = {}
        decade_counter = Counter()
        years = []

        for rating in ratings:
            movie = movies_dict.get(rating.tmdb_id)
            if movie:
                if movie.genres:
                    for genre in movie.genres:
                        genre_counter[genre] += 1
                        if genre not in genre_ratings:
                            genre_ratings[genre] = []
                        genre_ratings[genre].append(rating.rating)

                if movie.release_date:
                    year = movie.release_date.year
                    years.append(year)
                    decade = (year // 10) * 10
                    decade_counter[decade] += 1

        genre_distribution = [
            {
                "name": genre,
                "count": count,
                "avg_rating": round(sum(genre_ratings[genre]) / len(genre_ratings[genre]), 2) if genre in genre_ratings else 0,
            }
            for genre, count in genre_counter.most_common(8)
        ]

        decade_distribution = [{"decade": f"{decade}s", "count": count} for decade, count in sorted(decade_counter.items())]

        twelve_months_ago = datetime.datetime.now() - datetime.timedelta(days=365)
        monthly_activity = list(
            ratings.filter(created_at__gte=twelve_months_ago)
            .annotate(month=TruncMonth("created_at"))
            .values("month")
            .annotate(count=Count("id"), avg_rating=Avg("rating"))
            .order_by("month")
        )
        monthly_activity = [
            {
                "month": item["month"].strftime("%Y-%m"),
                "count": item["count"],
                "avg_rating": round(item["avg_rating"], 2) if item["avg_rating"] else 0,
            }
            for item in monthly_activity
        ]

        yearly_activity = list(
            ratings.annotate(year=ExtractYear("created_at"))
            .values("year")
            .annotate(count=Count("id"), avg_rating=Avg("rating"))
            .order_by("year")
        )
        yearly_activity = [
            {
                "year": item["year"],
                "count": item["count"],
                "avg_rating": round(item["avg_rating"], 2) if item["avg_rating"] else 0,
            }
            for item in yearly_activity
        ]

        liked_count = ratings.filter(rating__gte=4).count()
        disliked_count = ratings.filter(rating__lte=2).count()
        neutral_count = ratings.filter(rating__gt=2, rating__lt=4).count()

        avg = stats["avg_rating"] or 0
        if avg >= 4:
            rating_style = "Generous Critic"
        elif avg >= 3.5:
            rating_style = "Balanced Viewer"
        elif avg >= 3:
            rating_style = "Selective Critic"
        else:
            rating_style = "Tough Critic"

        favorite_genre = None
        if genre_distribution:
            best_score = 0
            for g in genre_distribution:
                if g["count"] >= 5:
                    score = g["count"] * g["avg_rating"]
                    if score > best_score:
                        best_score = score
                        favorite_genre = g["name"]
            if not favorite_genre:
                favorite_genre = genre_distribution[0]["name"]

        avg_movie_year = round(sum(years) / len(years)) if years else None

        most_active_month = None
        if monthly_activity:
            most_active = max(monthly_activity, key=lambda x: x["count"])
            most_active_month = most_active["month"]

        top_rated_genres = sorted(
            [g for g in genre_distribution if g["count"] >= 3], key=lambda x: x["avg_rating"], reverse=True
        )[:3]

        lowest_rated_genres = sorted([g for g in genre_distribution if g["count"] >= 3], key=lambda x: x["avg_rating"])[:3]

        return Response(
            {
                "total_ratings": total_ratings,
                "average_rating": round(stats["avg_rating"], 2) if stats["avg_rating"] else 0,
                "rating_distribution": rating_distribution,
                "genre_distribution": genre_distribution,
                "decade_distribution": decade_distribution,
                "monthly_activity": monthly_activity,
                "yearly_activity": yearly_activity,
                "insights": {
                    "favorite_genre": favorite_genre,
                    "avg_movie_year": avg_movie_year,
                    "most_active_month": most_active_month,
                    "rating_style": rating_style,
                    "liked_count": liked_count,
                    "disliked_count": disliked_count,
                    "neutral_count": neutral_count,
                    "liked_percentage": round((liked_count / total_ratings) * 100, 1) if total_ratings > 0 else 0,
                    "top_rated_genres": top_rated_genres,
                    "lowest_rated_genres": lowest_rated_genres,
                },
            }
        )
