"""Management command to sync MovieLens users to the database."""

import csv
import logging
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand
from django.db import transaction
from users.models import AppUser, Rating

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Sync MovieLens users and ratings to the database"

    def add_arguments(self, parser):
        parser.add_argument(
            "--ratings-path",
            type=str,
            help="Path to ratings CSV file",
        )
        parser.add_argument(
            "--links-path",
            type=str,
            help="Path to links CSV file (for MovieLens to TMDB mapping)",
        )
        parser.add_argument(
            "--limit-users",
            type=int,
            default=0,
            help="Maximum number of users to import (0 = unlimited, default: 0)",
        )
        parser.add_argument(
            "--min-ratings",
            type=int,
            default=20,
            help="Minimum ratings per user to include (default: 20)",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=1000,
            help="Batch size for database operations (default: 1000)",
        )

    def handle(self, *args, **options):
        model_data_path = Path(settings.MODEL_DATA_PATH) if hasattr(settings, "MODEL_DATA_PATH") else Path("/data/model_data")

        ratings_path = options.get("ratings_path")
        if not ratings_path:
            for dataset in ["ml-32m", "ml-latest-small"]:
                path = model_data_path / "raw" / dataset / "ratings.csv"
                if path.exists():
                    ratings_path = path
                    break

        links_path = options.get("links_path")
        if not links_path:
            for dataset in ["ml-32m", "ml-latest-small"]:
                path = model_data_path / "raw" / dataset / "links.csv"
                if path.exists():
                    links_path = path
                    break

        if not ratings_path or not Path(ratings_path).exists():
            self.stderr.write(self.style.ERROR(f"Ratings file not found: {ratings_path}"))
            return

        limit_users = options["limit_users"]
        min_ratings = options["min_ratings"]
        batch_size = options["batch_size"]

        limit_str = "unlimited" if limit_users == 0 else str(limit_users)
        self.stdout.write(f"Loading ratings from: {ratings_path}")
        self.stdout.write(f"Limit users: {limit_str}, Min ratings: {min_ratings}")

        ml_to_tmdb = {}
        if links_path and Path(links_path).exists():
            self.stdout.write(f"Loading ID mappings from: {links_path}")
            with open(links_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    movie_id = int(row["movieId"])
                    tmdb_id = row.get("tmdbId", "")
                    if tmdb_id:
                        ml_to_tmdb[movie_id] = int(tmdb_id)
            self.stdout.write(f"Loaded {len(ml_to_tmdb)} movie ID mappings")

        self.stdout.write("Counting ratings per user...")
        user_rating_counts = {}
        total_ratings = 0
        with open(ratings_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                user_id = int(row["userId"])
                user_rating_counts[user_id] = user_rating_counts.get(user_id, 0) + 1
                total_ratings += 1

        self.stdout.write(f"Found {len(user_rating_counts)} unique users with {total_ratings} total ratings")

        eligible_users = [uid for uid, count in user_rating_counts.items() if count >= min_ratings]
        eligible_users = sorted(eligible_users)

        if limit_users > 0:
            eligible_users = eligible_users[:limit_users]

        self.stdout.write(f"Selected {len(eligible_users)} eligible users (min {min_ratings} ratings)")

        self.stdout.write("Loading ratings...")
        user_ratings = {uid: [] for uid in eligible_users}
        eligible_set = set(eligible_users)
        ratings_loaded = 0

        with open(ratings_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                user_id = int(row["userId"])
                if user_id not in eligible_set:
                    continue

                movie_id = int(row["movieId"])
                tmdb_id = ml_to_tmdb.get(movie_id)
                if not tmdb_id:
                    continue

                rating = float(row["rating"])
                user_ratings[user_id].append((tmdb_id, rating))
                ratings_loaded += 1

                if ratings_loaded % 1000000 == 0:
                    self.stdout.write(f"  Loaded {ratings_loaded:,} ratings...")

        self.stdout.write(f"Loaded {ratings_loaded:,} ratings for import")

        self.stdout.write("Importing to database...")

        users_created = 0
        users_updated = 0
        ratings_created = 0

        for i in range(0, len(eligible_users), batch_size):
            batch_users = eligible_users[i : i + batch_size]

            with transaction.atomic():
                for user_id in batch_users:
                    if not user_ratings[user_id]:
                        continue

                    user, created = AppUser.objects.get_or_create(movielens_user_id=user_id, defaults={"is_cold_start": False})

                    if created:
                        users_created += 1
                    else:
                        users_updated += 1

                    existing_tmdb_ids = set(user.ratings.values_list("tmdb_id", flat=True))

                    new_ratings = []
                    for tmdb_id, rating in user_ratings[user_id]:
                        if tmdb_id not in existing_tmdb_ids:
                            new_ratings.append(Rating(user=user, tmdb_id=tmdb_id, rating=rating))

                    if new_ratings:
                        Rating.objects.bulk_create(new_ratings, ignore_conflicts=True)
                        ratings_created += len(new_ratings)

            processed = min(i + batch_size, len(eligible_users))
            self.stdout.write(f"  Processed {processed:,}/{len(eligible_users):,} users...")

        self.stdout.write(
            self.style.SUCCESS(f"Imported {users_created:,} new users, updated {users_updated:,} existing users")
        )
        self.stdout.write(self.style.SUCCESS(f"Created {ratings_created:,} new ratings"))

        total_users = AppUser.objects.count()
        total_ratings = Rating.objects.count()
        self.stdout.write(f"Total users in DB: {total_users:,}")
        self.stdout.write(f"Total ratings in DB: {total_ratings:,}")
