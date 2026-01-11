"""Management command to load movie embeddings into Qdrant."""

import json
import logging
from pathlib import Path

import numpy as np
from django.conf import settings
from django.core.management.base import BaseCommand
from recommendations.services.qdrant_service import QdrantService

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Load movie embeddings into Qdrant vector database"

    def add_arguments(self, parser):
        parser.add_argument(
            "--embeddings-path",
            type=str,
            help="Path to embeddings file (NPY or NPZ format)",
        )
        parser.add_argument(
            "--clear",
            action="store_true",
            help="Clear existing collection before loading",
        )

    def handle(self, *args, **options):
        embeddings_path = options.get("embeddings_path")

        if not embeddings_path:
            embeddings_path = Path(settings.EMBEDDINGS_PATH) / "combined_movie_embeddings.npy"
        else:
            embeddings_path = Path(embeddings_path)

        self.stdout.write(f"Loading embeddings from: {embeddings_path}")

        if not embeddings_path.exists():
            self.stderr.write(self.style.ERROR(f"Embeddings file not found: {embeddings_path}"))
            return

        try:
            qdrant = QdrantService()
        except Exception as e:
            self.stderr.write(self.style.ERROR(f"Failed to connect to Qdrant: {e}"))
            return

        if options.get("clear"):
            self.stdout.write("Clearing existing collection...")
            try:
                qdrant.client.delete_collection(QdrantService.COLLECTION_NAME)
                qdrant._ensure_collection()
            except Exception:
                pass

        self.stdout.write("Loading embeddings into Qdrant...")
        try:
            count = qdrant.load_embeddings(str(embeddings_path))
            self.stdout.write(self.style.SUCCESS(f"Successfully loaded {count} embeddings"))
        except Exception as e:
            self.stderr.write(self.style.ERROR(f"Failed to load embeddings: {e}"))
            raise

        info = qdrant.get_collection_info()
        self.stdout.write(f"Collection status: {info}")
