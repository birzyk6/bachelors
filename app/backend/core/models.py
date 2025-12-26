"""Core models for movie recommendation app."""

from django.db import models


class Movie(models.Model):
    """Cached movie metadata from TMDB."""

    tmdb_id = models.IntegerField(primary_key=True)
    title = models.CharField(max_length=500)
    original_title = models.CharField(max_length=500, blank=True, null=True)
    overview = models.TextField(blank=True, null=True)
    poster_path = models.CharField(max_length=255, blank=True, null=True)
    backdrop_path = models.CharField(max_length=255, blank=True, null=True)
    release_date = models.DateField(blank=True, null=True)
    vote_average = models.FloatField(default=0)
    vote_count = models.IntegerField(default=0)
    genres = models.JSONField(default=list)
    keywords = models.JSONField(default=list)
    runtime = models.IntegerField(blank=True, null=True)
    tagline = models.TextField(blank=True, null=True)
    popularity = models.FloatField(default=0)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "movies"
        ordering = ["-popularity"]

    def __str__(self):
        return f"{self.title} ({self.tmdb_id})"


class Genre(models.Model):
    """Movie genres for filtering."""

    tmdb_id = models.IntegerField(primary_key=True)
    name = models.CharField(max_length=100)

    class Meta:
        db_table = "genres"

    def __str__(self):
        return self.name
