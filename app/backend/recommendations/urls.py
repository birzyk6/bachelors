"""URL configuration for recommendations app."""

from django.urls import path

from . import views

urlpatterns = [
    path("for-you/", views.for_you, name="for-you"),
    path("similar/", views.similar_movies, name="similar-movies"),
    path("because-watched/", views.because_you_watched, name="because-watched"),
    path("by-genre/", views.by_genre, name="by-genre"),
    path("trending/", views.trending, name="trending"),
    path("seed-movies/", views.seed_movies, name="seed-movies"),
    path("status/", views.system_status, name="system-status"),
    path("load-embeddings/", views.load_embeddings, name="load-embeddings"),
]
