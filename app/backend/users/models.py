"""User models for movie recommendation app."""

from django.db import models


class AppUser(models.Model):
    """Application user (for impersonation or cold-start)."""

    movielens_user_id = models.IntegerField(unique=True, null=True, blank=True)
    username = models.CharField(max_length=255, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    is_cold_start = models.BooleanField(default=False)

    class Meta:
        db_table = "app_users"

    def __str__(self):
        if self.movielens_user_id:
            return f"User {self.movielens_user_id}"
        return f"ColdStart User {self.id}"


class UserPreference(models.Model):
    """User preferences for genres, keywords, etc."""

    PREFERENCE_TYPES = [
        ("genre", "Genre"),
        ("keyword", "Keyword"),
        ("actor", "Actor"),
        ("director", "Director"),
    ]

    user = models.ForeignKey(AppUser, on_delete=models.CASCADE, related_name="preferences")
    preference_type = models.CharField(max_length=50, choices=PREFERENCE_TYPES)
    preference_value = models.CharField(max_length=255)
    weight = models.FloatField(default=1.0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "user_preferences"

    def __str__(self):
        return f"{self.user} - {self.preference_type}: {self.preference_value}"


class Rating(models.Model):
    """User movie ratings."""

    user = models.ForeignKey(AppUser, on_delete=models.CASCADE, related_name="ratings")
    tmdb_id = models.IntegerField()
    rating = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "user_ratings"
        unique_together = ["user", "tmdb_id"]
        indexes = [
            models.Index(fields=["user"], name="rating_user_idx"),
        ]

    def __str__(self):
        return f"{self.user} rated {self.tmdb_id}: {self.rating}"


class WatchHistory(models.Model):
    """User watch history."""

    user = models.ForeignKey(AppUser, on_delete=models.CASCADE, related_name="watch_history")
    tmdb_id = models.IntegerField()
    watched_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "watch_history"
        ordering = ["-watched_at"]

    def __str__(self):
        return f"{self.user} watched {self.tmdb_id}"
