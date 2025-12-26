"""URL configuration for movie recommendation app."""

from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("core.urls")),
    path("api/users/", include("users.urls")),
    path("api/recommendations/", include("recommendations.urls")),
]
