"""
Main CLI application for Two-Tower movie recommendations.

Interactive command-line interface for querying movie recommendations.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from .display import (
    Colors,
    print_error,
    print_genres,
    print_header,
    print_help,
    print_info,
    print_recommendations,
    print_search_results,
    print_success,
    print_warning,
    print_welcome,
    prompt,
)
from .embedding_store import EmbeddingStore
from .engine import RecommendationEngine
from .movie_catalog import MovieCatalog


def get_default_paths() -> tuple[Path, Path, Path]:
    """Get default paths for data files."""
    # Try to import config for paths
    try:
        from config import MODELS_DIR, PROCESSED_DIR

        embeddings_path = MODELS_DIR / "movie_embeddings.npy"
        metadata_path = MODELS_DIR / "movie_embeddings_metadata.json"
        movies_path = PROCESSED_DIR / "movies.parquet"
    except ImportError:
        # Fall back to relative paths
        project_root = Path(__file__).parent.parent.parent.parent.parent
        embeddings_path = project_root / "model" / "saved_models" / "movie_embeddings.npy"
        metadata_path = project_root / "model" / "saved_models" / "movie_embeddings_metadata.json"
        movies_path = project_root / "model" / "data" / "processed" / "movies.parquet"

    return embeddings_path, metadata_path, movies_path


class MovieRecommenderCLI:
    """
    Interactive CLI application for movie recommendations.

    Commands:
        genre <genres>  - Find movies by genre(s)
        similar <id>    - Find similar movies
        search <title>  - Search by title
        info <id>       - Show movie info
        favorites       - Recommendations from favorites
        genres          - List available genres
        help            - Show help
        quit            - Exit
    """

    def __init__(self, embeddings_path: Path, metadata_path: Path, movies_path: Path, top_k: int = 10):
        """
        Initialize CLI application.

        Args:
            embeddings_path: Path to movie embeddings
            metadata_path: Path to embeddings metadata
            movies_path: Path to movies parquet file
            top_k: Default number of recommendations
        """
        self.top_k = top_k

        # Initialize stores
        self.embedding_store = EmbeddingStore(embeddings_path, metadata_path)
        self.movie_catalog = MovieCatalog(movies_path)

        # Will be initialized after loading
        self.engine: Optional[RecommendationEngine] = None

    def load(self) -> bool:
        """
        Load data files.

        Returns:
            True if successful, False otherwise
        """
        try:
            print_info("Loading movie embeddings...")
            self.embedding_store.load()
            print_success(f"Loaded {self.embedding_store.num_movies:,} movie embeddings")

            print_info("Loading movie catalog...")
            self.movie_catalog.load()
            print_success(f"Loaded {self.movie_catalog.num_movies:,} movies")

            # Initialize engine
            self.engine = RecommendationEngine(self.embedding_store, self.movie_catalog)

            return True
        except FileNotFoundError as e:
            print_error(f"File not found: {e}")
            return False
        except Exception as e:
            print_error(f"Error loading data: {e}")
            return False

    def cmd_genre(self, args: str) -> None:
        """Handle genre command."""
        if not args:
            print_warning("Please specify genre(s). Example: genre action sci-fi")
            return

        genres = self.engine.parse_genre_query(args)

        if not genres:
            print_warning(f'No matching genres found for "{args}"')
            print_info("Use 'genres' command to see available genres")
            return

        print_info(f"Searching for genres: {', '.join(genres)}")

        results = self.engine.recommend_by_genres(genres, top_k=self.top_k)
        print_recommendations(results, f"Movies similar to {', '.join(genres)}")

    def cmd_similar(self, args: str) -> None:
        """Handle similar command."""
        if not args:
            print_warning("Please specify a movie ID. Example: similar 260")
            return

        try:
            movie_id = int(args.split()[0])
        except ValueError:
            print_error(f'Invalid movie ID: "{args}"')
            return

        # Check if movie exists
        title = self.movie_catalog.get_title(movie_id)
        if title.startswith("Unknown Movie"):
            print_error(f"Movie ID {movie_id} not found")
            return

        print_info(f"Finding movies similar to: {title}")

        results = self.engine.recommend_similar_to_movie(movie_id, top_k=self.top_k)

        if not results:
            print_warning(f"No embedding found for movie ID {movie_id}")
            return

        print_recommendations(results, f"Movies similar to: {title}")

    def cmd_search(self, args: str) -> None:
        """Handle search command."""
        if not args:
            print_warning("Please specify a search term. Example: search star wars")
            return

        results = self.engine.search_movies(args, limit=20)
        print_search_results(results, args)

    def cmd_info(self, args: str) -> None:
        """Handle info command."""
        if not args:
            print_warning("Please specify a movie ID. Example: info 260")
            return

        try:
            movie_id = int(args.split()[0])
        except ValueError:
            print_error(f'Invalid movie ID: "{args}"')
            return

        title = self.movie_catalog.get_title(movie_id)
        if title.startswith("Unknown Movie"):
            print_error(f"Movie ID {movie_id} not found")
            return

        print_header(f"Movie Info: {title}")

        genres = self.movie_catalog.get_genres(movie_id)
        year = self.movie_catalog.get_year(movie_id)
        overview = self.movie_catalog.get_overview(movie_id)
        has_embedding = movie_id in self.embedding_store.embeddings

        print(f"  {Colors.BOLD}ID:{Colors.END}       {movie_id}")
        print(f"  {Colors.BOLD}Title:{Colors.END}    {title}")
        print(f"  {Colors.BOLD}Year:{Colors.END}     {year or 'Unknown'}")
        print(f"  {Colors.BOLD}Genres:{Colors.END}   {genres or 'None'}")
        print(f"  {Colors.BOLD}Embedding:{Colors.END} {'Yes ✓' if has_embedding else 'No ✗'}")

        if overview:
            print(f"  {Colors.BOLD}Overview:{Colors.END}")
            # Word wrap overview
            words = overview.split()
            line = "    "
            for word in words:
                if len(line) + len(word) > 70:
                    print(line)
                    line = "    "
                line += word + " "
            if line.strip():
                print(line)

        print()

    def cmd_favorites(self, args: str) -> None:
        """Handle favorites command - interactive mode."""
        print_header("Favorites-Based Recommendations")
        print("Enter movie IDs of movies you like (comma or space separated).")
        print("Then I'll find similar movies you might enjoy!")
        print()

        user_input = prompt()
        if user_input.lower() in ("quit", "q", "exit"):
            return

        # Parse movie IDs
        ids_str = user_input.replace(",", " ").split()
        movie_ids = []

        for s in ids_str:
            try:
                movie_id = int(s)
                title = self.movie_catalog.get_title(movie_id)
                if not title.startswith("Unknown Movie"):
                    movie_ids.append(movie_id)
                    print_success(f"Added: {title}")
                else:
                    print_warning(f"Movie ID {movie_id} not found, skipping")
            except ValueError:
                print_warning(f'Invalid ID "{s}", skipping')

        if not movie_ids:
            print_error("No valid movie IDs provided")
            return

        print()
        print_info(f"Finding recommendations based on {len(movie_ids)} favorite(s)...")

        results = self.engine.recommend_similar_to_movies(movie_ids, top_k=self.top_k)
        print_recommendations(results, "Recommendations Based on Your Favorites")

    def cmd_genres(self, args: str) -> None:
        """Handle genres command."""
        print_genres(self.movie_catalog.genres)

    def process_command(self, user_input: str) -> bool:
        """
        Process a user command.

        Args:
            user_input: Raw user input

        Returns:
            True to continue, False to quit
        """
        if not user_input:
            return True

        # Parse command and arguments
        parts = user_input.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        # Dispatch commands
        if cmd in ("quit", "q", "exit"):
            print_info("Goodbye!")
            return False
        elif cmd in ("help", "h", "?"):
            print_help()
        elif cmd == "genres":
            self.cmd_genres(args)
        elif cmd in ("similar", "like") and args:
            self.cmd_similar(args)
        elif cmd == "info" and args:
            self.cmd_info(args)
        elif cmd in ("favorites", "favs", "fav"):
            self.cmd_favorites(args)
        else:
            # Default: Smart search - treat any input as a query
            self.cmd_smart_search(user_input)

        return True

    def cmd_smart_search(self, query: str) -> None:
        """Handle smart search - the default for any query."""
        results, explanation = self.engine.smart_search(query, top_k=self.top_k)

        print_info(explanation)
        print_recommendations(results, f'Results for "{query}"')

    def run_interactive(self) -> None:
        """Run interactive REPL loop."""
        print_welcome(self.movie_catalog.num_movies, self.embedding_store.num_movies)

        while True:
            try:
                user_input = prompt()
                if not self.process_command(user_input):
                    break
            except KeyboardInterrupt:
                print()
                print_info("Use 'quit' to exit")
            except Exception as e:
                print_error(f"Error: {e}")

    def run_single_command(self, command: str) -> None:
        """Run a single command and exit."""
        self.process_command(command)


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Two-Tower Movie Recommendation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Interactive mode
  %(prog)s -c "genre action sci-fi" # Single command
  %(prog)s -c "similar 260"         # Find similar to Star Wars
  %(prog)s -c "search matrix"       # Search for movies
        """,
    )

    parser.add_argument("-c", "--command", help="Run a single command and exit")
    parser.add_argument("-k", "--top-k", type=int, default=10, help="Number of recommendations (default: 10)")
    parser.add_argument("--embeddings", type=Path, help="Path to movie embeddings file")
    parser.add_argument("--metadata", type=Path, help="Path to embeddings metadata file")
    parser.add_argument("--movies", type=Path, help="Path to movies parquet file")

    args = parser.parse_args()

    # Get paths
    default_embeddings, default_metadata, default_movies = get_default_paths()

    embeddings_path = args.embeddings or default_embeddings
    metadata_path = args.metadata or default_metadata
    movies_path = args.movies or default_movies

    # Initialize CLI
    cli = MovieRecommenderCLI(
        embeddings_path=embeddings_path, metadata_path=metadata_path, movies_path=movies_path, top_k=args.top_k
    )

    # Load data
    if not cli.load():
        sys.exit(1)

    # Run
    if args.command:
        cli.run_single_command(args.command)
    else:
        cli.run_interactive()


if __name__ == "__main__":
    main()
