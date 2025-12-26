"""Serializers for user models."""

from rest_framework import serializers

from .models import AppUser, Rating, UserPreference, WatchHistory


class UserPreferenceSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserPreference
        fields = ["id", "preference_type", "preference_value", "weight"]


class RatingSerializer(serializers.ModelSerializer):
    class Meta:
        model = Rating
        fields = ["id", "tmdb_id", "rating", "created_at"]


class WatchHistorySerializer(serializers.ModelSerializer):
    class Meta:
        model = WatchHistory
        fields = ["id", "tmdb_id", "watched_at"]


class AppUserSerializer(serializers.ModelSerializer):
    preferences = UserPreferenceSerializer(many=True, read_only=True)
    ratings_count = serializers.SerializerMethodField()
    display_name = serializers.SerializerMethodField()

    class Meta:
        model = AppUser
        fields = [
            "id",
            "movielens_user_id",
            "username",
            "is_cold_start",
            "created_at",
            "preferences",
            "ratings_count",
            "display_name",
        ]

    def get_ratings_count(self, obj):
        return obj.ratings.count()

    def get_display_name(self, obj):
        if obj.username:
            return obj.username
        if obj.movielens_user_id:
            return f"User #{obj.movielens_user_id}"
        return f"Guest #{obj.id}"


class AppUserListSerializer(serializers.ModelSerializer):
    """Lightweight serializer for user lists."""

    ratings_count = serializers.SerializerMethodField()
    display_name = serializers.SerializerMethodField()

    class Meta:
        model = AppUser
        fields = ["id", "movielens_user_id", "display_name", "ratings_count"]

    def get_ratings_count(self, obj):
        # Use annotated field if available (from list view), otherwise count
        return getattr(obj, "rating_count", obj.ratings.count())

    def get_display_name(self, obj):
        if obj.username:
            return obj.username
        if obj.movielens_user_id:
            return f"User #{obj.movielens_user_id}"
        return f"Guest #{obj.id}"


class ColdStartUserCreateSerializer(serializers.Serializer):
    """Serializer for creating cold-start users."""

    genres = serializers.ListField(child=serializers.CharField(), required=False, default=list)
    initial_ratings = serializers.ListField(child=serializers.DictField(), required=False, default=list)


class RateMovieSerializer(serializers.Serializer):
    """Serializer for rating a movie."""

    rating = serializers.FloatField(min_value=0.5, max_value=5.0)
