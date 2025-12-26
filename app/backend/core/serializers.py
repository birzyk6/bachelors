"""Serializers for core models."""

from rest_framework import serializers

from .models import Genre, Movie


class GenreSerializer(serializers.ModelSerializer):
    class Meta:
        model = Genre
        fields = ["tmdb_id", "name"]


class MovieSerializer(serializers.ModelSerializer):
    poster_url = serializers.SerializerMethodField()
    backdrop_url = serializers.SerializerMethodField()
    year = serializers.SerializerMethodField()

    class Meta:
        model = Movie
        fields = [
            "tmdb_id",
            "title",
            "original_title",
            "overview",
            "poster_path",
            "backdrop_path",
            "poster_url",
            "backdrop_url",
            "release_date",
            "year",
            "vote_average",
            "vote_count",
            "genres",
            "keywords",
            "runtime",
            "tagline",
            "popularity",
        ]

    def get_poster_url(self, obj):
        if obj.poster_path:
            return f"https://image.tmdb.org/t/p/w500{obj.poster_path}"
        return None

    def get_backdrop_url(self, obj):
        if obj.backdrop_path:
            return f"https://image.tmdb.org/t/p/original{obj.backdrop_path}"
        return None

    def get_year(self, obj):
        if not obj.release_date:
            return None
        if isinstance(obj.release_date, str):
            try:
                return int(obj.release_date[:4])
            except (ValueError, IndexError):
                return None
        if hasattr(obj.release_date, "year"):
            return obj.release_date.year
        return None


class MovieListSerializer(serializers.ModelSerializer):
    """Lightweight serializer for movie lists."""

    poster_url = serializers.SerializerMethodField()
    year = serializers.SerializerMethodField()

    class Meta:
        model = Movie
        fields = ["tmdb_id", "title", "poster_url", "year", "vote_average", "genres"]

    def get_poster_url(self, obj):
        if obj.poster_path:
            return f"https://image.tmdb.org/t/p/w342{obj.poster_path}"
        return None

    def get_year(self, obj):
        if not obj.release_date:
            return None
        if isinstance(obj.release_date, str):
            try:
                return int(obj.release_date[:4])
            except (ValueError, IndexError):
                return None
        if hasattr(obj.release_date, "year"):
            return obj.release_date.year
        return None
