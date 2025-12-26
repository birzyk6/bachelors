"""URL configuration for core app."""

from django.urls import include, path
from rest_framework.routers import DefaultRouter

from . import views

router = DefaultRouter()
router.register(r"movies", views.MovieViewSet, basename="movie")
router.register(r"genres", views.GenreViewSet, basename="genre")

urlpatterns = [
    path("", include(router.urls)),
    path("health/", views.health_check, name="health-check"),
    path("genres/list/", views.available_genres, name="genres-list"),
]
