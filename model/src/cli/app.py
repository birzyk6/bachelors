"""
Main CLI application for Two-Tower movie recommendations.

Interactive command-line interface for querying movie recommendations.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Set

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
from .user_store import UserStore


def get_default_paths() -> tuple[Path, Path, Path, Path, Path]:
    """Get default paths for data files."""
    # Try to import config for paths
    try:
        from config import MODELS_DIR, PROCESSED_DIR

        # Prefer combined embeddings (MovieLens + TMDB cold-start)
        combined_embeddings = MODELS_DIR / "combined_movie_embeddings.npy"
        combined_metadata = MODELS_DIR / "combined_movie_embeddings_metadata.json"

        # Fall back to MovieLens-only embeddings if combined don't exist
        if combined_embeddings.exists() and combined_metadata.exists():
            embeddings_path = combined_embeddings
            metadata_path = combined_metadata
        else:
            embeddings_path = MODELS_DIR / "movie_embeddings.npy"
            metadata_path = MODELS_DIR / "movie_embeddings_metadata.json"

        # Prefer combined movie catalog (MovieLens + TMDB)
        combined_movies = PROCESSED_DIR / "movies_combined.parquet"
        if combined_movies.exists():
            movies_path = combined_movies
        else:
            movies_path = PROCESSED_DIR / "movies.parquet"

        # User data paths
        ratings_path = PROCESSED_DIR / "train.parquet"
        user_tower_path = MODELS_DIR / "user_tower"

    except ImportError:
        # Fall back to relative paths
        project_root = Path(__file__).parent.parent.parent.parent.parent
        embeddings_path = project_root / "model" / "saved_models" / "combined_movie_embeddings.npy"
        metadata_path = project_root / "model" / "saved_models" / "combined_movie_embeddings_metadata.json"
        movies_path = project_root / "model" / "data" / "processed" / "movies_combined.parquet"
        ratings_path = project_root / "model" / "data" / "processed" / "train.parquet"
        user_tower_path = project_root / "model" / "saved_models" / "user_tower"

    return embeddings_path, metadata_path, movies_path, ratings_path, user_tower_path


class MovieRecommenderCLI:
    """
    Interactive CLI application for movie recommendations.

    Commands:
        genre <genres>  - Find movies by genre(s)
        similar <id>    - Find similar movies
        search <title>  - Search by title
        info <id>       - Show movie info
        favorites       - Recommendations from favorites
        user <id>       - Impersonate a user
        recommend       - Get personalized recommendations (when impersonating)
        history         - Show user's watch history (when impersonating)
        whoami          - Show current user info
        logout          - Stop impersonating user
        genres          - List available genres
        help            - Show help
        quit            - Exit
    """

    def __init__(
        self,
        embeddings_path: Path,
        metadata_path: Path,
        movies_path: Path,
        ratings_path: Optional[Path] = None,
        user_tower_path: Optional[Path] = None,
        top_k: int = 10,
    ):
        """
        Initialize CLI application.

        Args:
            embeddings_path: Path to movie embeddings
            metadata_path: Path to embeddings metadata
            movies_path: Path to movies parquet file
            ratings_path: Path to ratings parquet file (for user data)
            user_tower_path: Path to saved user tower model
            top_k: Default number of recommendations
        """
        self.top_k = top_k

        # Initialize stores
        self.embedding_store = EmbeddingStore(embeddings_path, metadata_path)
        self.movie_catalog = MovieCatalog(movies_path)

        # User store (optional)
        self.user_store: Optional[UserStore] = None
        if ratings_path and user_tower_path:
            self.user_store = UserStore(ratings_path, user_tower_path)

        # Current impersonated user
        self._current_user_id: Optional[int] = None
        self._current_user_watched: Set[int] = set()

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

            # Load user data if available
            if self.user_store:
                try:
                    self.user_store.load()
                    self.user_store.movies_catalog = self.movie_catalog
                    print_success(f"Loaded {self.user_store.num_users:,} users")
                except Exception as e:
                    print_warning(f"Could not load user data: {e}")
                    self.user_store = None

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
        elif cmd in ("user", "login", "impersonate"):
            self.cmd_user(args)
        elif cmd in ("recommend", "recommendations", "recs", "rec"):
            self.cmd_recommend(args)
        elif cmd in ("history", "watched"):
            self.cmd_history(args)
        elif cmd in ("whoami", "me"):
            self.cmd_whoami(args)
        elif cmd in ("logout", "logoff"):
            self.cmd_logout(args)
        elif cmd == "users":
            self.cmd_users(args)
        else:
            # Default: Smart search - treat any input as a query
            self.cmd_smart_search(user_input)

        return True

    def cmd_user(self, args: str) -> None:
        """Handle user impersonation command."""
        if not self.user_store:
            print_error("User data not available. Make sure train.parquet and user_tower model exist.")
            return

        if not args:
            # Show sample users
            print_header("User Impersonation")
            print("Enter a user ID to impersonate them and get personalized recommendations.")
            print()
            print_info("Sample active users:")
            sample_users = self.user_store.get_sample_users(n=5, min_ratings=100)
            for uid in sample_users:
                stats = self.user_store.get_user_stats(uid)
                print(f"  {Colors.CYAN}User {uid}{Colors.END}: {stats['num_ratings']} ratings, avg {stats['avg_rating']:.1f}★")
            print()
            print("Usage: user <id>")
            return

        try:
            user_id = int(args.split()[0])
        except ValueError:
            print_error(f'Invalid user ID: "{args}"')
            return

        if not self.user_store.user_exists(user_id):
            print_error(f"User {user_id} not found in dataset")
            return

        # Impersonate user
        self._current_user_id = user_id

        # Get user's watched movies
        history = self.user_store.get_user_history(user_id)
        self._current_user_watched = {mid for mid, _, _ in history}

        stats = self.user_store.get_user_stats(user_id)
        top_genres = self.user_store.get_top_genres_for_user(user_id, top_k=3)

        print_success(f"Now impersonating User {user_id}")
        print()
        print(f"  {Colors.BOLD}Ratings:{Colors.END}      {stats['num_ratings']}")
        print(f"  {Colors.BOLD}Avg Rating:{Colors.END}   {stats['avg_rating']:.1f}★")
        if top_genres:
            genres_str = ", ".join(f"{g} ({c})" for g, c in top_genres)
            print(f"  {Colors.BOLD}Top Genres:{Colors.END}   {genres_str}")
        print()
        print_info(
            "Commands: 'recommend' for personalized picks, 'recommend popular', " "'recommend <theme>', 'history', 'logout'"
        )

    def cmd_recommend(self, args: str) -> None:
        """
        Handle personalized recommendations command.

        Modes:
            recommend              - General personalized picks
            recommend <theme>      - Themed recommendations (e.g., "recommend vampires")
            recommend popular      - Popular movies matching your taste
            recommend recent       - Based on your most recent watch
            recommend discover     - Movies outside your comfort zone
            recommend gems         - Hidden gems you might like
        """
        if not self._current_user_id:
            print_warning("Not impersonating any user. Use 'user <id>' first.")
            print_info("Or use 'favorites' to get recommendations from movies you like.")
            return

        if not self.user_store:
            print_error("User data not available.")
            return

        # Get user embedding from the Two-Tower model
        user_embedding = self.user_store.get_user_embedding(self._current_user_id)

        if user_embedding is None:
            print_warning("Could not generate user embedding. Falling back to history-based recommendations.")
            history = self.user_store.get_user_history(self._current_user_id, min_rating=4.0, limit=10)
            if history:
                movie_ids = [mid for mid, _, _ in history]
                results = self.engine.recommend_similar_to_movies(movie_ids, top_k=self.top_k)
                print_recommendations(results, f"Recommendations for User {self._current_user_id}")
            else:
                print_error("No high-rated movies in history to base recommendations on.")
            return

        # Parse recommendation mode
        mode = args.strip().lower() if args else ""

        if not mode:
            # Default: general personalized picks
            print_info(f"Generating personalized picks for User {self._current_user_id}...")
            results = self.engine.recommend_for_user(
                user_embedding, exclude_movie_ids=self._current_user_watched, top_k=self.top_k
            )
            print_recommendations(results, f"🎬 Personalized Picks for User {self._current_user_id}")
            self._print_recommend_modes()

        elif mode == "popular":
            # Popular movies matching user taste
            print_info(f"Finding popular movies for User {self._current_user_id}...")
            results = self.engine.recommend_popular_for_user(
                user_embedding, exclude_movie_ids=self._current_user_watched, top_k=self.top_k
            )
            print_recommendations(results, f"🌟 Popular Picks for User {self._current_user_id}")

        elif mode == "recent":
            # Based on most recently watched movie
            history = self.user_store.get_user_history(self._current_user_id, limit=1)
            if not history:
                print_error("No watch history available.")
                return

            recent_movie_id, rating, _ = history[0]
            recent_title = self.movie_catalog.get_title(recent_movie_id)
            print_info(f"Finding movies similar to your recent watch: {recent_title}")

            results = self.engine.recommend_based_on_movie(
                recent_movie_id,
                user_embedding=user_embedding,
                exclude_movie_ids=self._current_user_watched,
                top_k=self.top_k,
            )
            print_recommendations(results, f"📺 Because You Watched: {recent_title}")

        elif mode == "discover":
            # Movies outside comfort zone
            print_info(f"Finding new discoveries for User {self._current_user_id}...")
            top_genres = self.user_store.get_top_genres_for_user(self._current_user_id, top_k=3)
            genre_names = [g for g, _ in top_genres]

            results = self.engine.recommend_discover(
                user_embedding,
                user_top_genres=genre_names,
                exclude_movie_ids=self._current_user_watched,
                top_k=self.top_k,
            )
            print_recommendations(results, f"🔍 Discover Something New (outside {', '.join(genre_names)})")

        elif mode in ("gems", "hidden", "hidden gems", "underrated"):
            # Hidden gems
            print_info(f"Finding hidden gems for User {self._current_user_id}...")
            results = self.engine.recommend_hidden_gems(
                user_embedding,
                exclude_movie_ids=self._current_user_watched,
                top_k=self.top_k,
            )
            print_recommendations(results, f"💎 Hidden Gems for User {self._current_user_id}")

        else:
            # Themed recommendations (e.g., "vampires", "space", "90s comedy")
            print_info(f"Finding '{mode}' movies for User {self._current_user_id}...")
            results = self.engine.recommend_for_user_with_theme(
                user_embedding,
                theme_query=mode,
                exclude_movie_ids=self._current_user_watched,
                top_k=self.top_k,
            )
            print_recommendations(results, f"🎭 '{mode.title()}' Picks for User {self._current_user_id}")

    def _print_recommend_modes(self) -> None:
        """Print available recommendation modes."""
        print(f"\n{Colors.CYAN}Other recommendation modes:{Colors.END}")
        print(f"  recommend {Colors.BOLD}popular{Colors.END}     - Popular movies you'd enjoy")
        print(f"  recommend {Colors.BOLD}recent{Colors.END}      - Based on your last watched movie")
        print(f"  recommend {Colors.BOLD}discover{Colors.END}    - Explore outside your comfort zone")
        print(f"  recommend {Colors.BOLD}gems{Colors.END}        - Hidden gems you might love")
        print(f"  recommend {Colors.BOLD}<theme>{Colors.END}     - E.g., 'recommend vampires', 'recommend 90s action'")
        print()

    def cmd_history(self, args: str) -> None:
        """Handle watch history command."""
        if not self._current_user_id:
            print_warning("Not impersonating any user. Use 'user <id>' first.")
            return

        limit = 20
        if args:
            try:
                limit = int(args.split()[0])
            except ValueError:
                pass

        history = self.user_store.get_user_history(self._current_user_id, limit=limit)

        if not history:
            print_info(f"User {self._current_user_id} has no watch history.")
            return

        print_header(f"Recent Watch History for User {self._current_user_id}")

        for i, (movie_id, rating, timestamp) in enumerate(history[:limit], 1):
            title = self.movie_catalog.get_title(movie_id)
            genres = self.movie_catalog.get_genres(movie_id)
            year = self.movie_catalog.get_year(movie_id)

            # Format date
            try:
                date_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
            except:
                date_str = "Unknown"

            # Rating stars
            stars = "★" * int(rating) + "☆" * (5 - int(rating))

            year_str = f"({year})" if year else ""
            genres_str = f"[{genres}]" if genres else ""

            print(f"  {i:2}. {Colors.BOLD}{title}{Colors.END} {year_str} {genres_str}")
            print(f"      {Colors.YELLOW}{stars}{Colors.END} ({rating:.1f}) - {date_str}")

        print()
        print_info(f"Showing {min(limit, len(history))} of {len(self._current_user_watched)} total ratings")

    def cmd_whoami(self, args: str) -> None:
        """Handle whoami command."""
        if not self._current_user_id:
            print_info("Not impersonating any user.")
            print_info("Use 'user <id>' to impersonate a user.")
            return

        stats = self.user_store.get_user_stats(self._current_user_id)
        top_genres = self.user_store.get_top_genres_for_user(self._current_user_id, top_k=5)

        print_header(f"Current User: {self._current_user_id}")
        print(f"  {Colors.BOLD}Total Ratings:{Colors.END}  {stats['num_ratings']}")
        print(f"  {Colors.BOLD}Avg Rating:{Colors.END}     {stats['avg_rating']:.1f}★")
        print(f"  {Colors.BOLD}Rating Range:{Colors.END}   {stats['min_rating']:.1f} - {stats['max_rating']:.1f}")

        if top_genres:
            print(f"  {Colors.BOLD}Top Genres:{Colors.END}")
            for genre, count in top_genres:
                bar_len = min(20, count // 5)
                bar = "█" * bar_len
                print(f"      {genre:15} {bar} ({count})")
        print()

    def cmd_logout(self, args: str) -> None:
        """Handle logout command."""
        if not self._current_user_id:
            print_info("Not impersonating any user.")
            return

        print_info(f"Logged out from User {self._current_user_id}")
        self._current_user_id = None
        self._current_user_watched = set()

    def cmd_users(self, args: str) -> None:
        """Show sample users."""
        if not self.user_store:
            print_error("User data not available.")
            return

        n = 10
        if args:
            try:
                n = int(args.split()[0])
            except ValueError:
                pass

        print_header(f"Sample Active Users (min 100 ratings)")
        sample_users = self.user_store.get_sample_users(n=n, min_ratings=100)

        for uid in sample_users:
            stats = self.user_store.get_user_stats(uid)
            top_genres = self.user_store.get_top_genres_for_user(uid, top_k=2)
            genres_str = ", ".join(g for g, _ in top_genres) if top_genres else "Various"
            print(
                f"  {Colors.CYAN}User {uid:6}{Colors.END}: {stats['num_ratings']:4} ratings, avg {stats['avg_rating']:.1f}★ | {genres_str}"
            )

        print()
        print_info(f"Total users: {self.user_store.num_users:,}")
        print_info("Use 'user <id>' to impersonate a user")

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
  %(prog)s -c "user 12345"          # Impersonate a user
        """,
    )

    parser.add_argument("-c", "--command", help="Run a single command and exit")
    parser.add_argument("-k", "--top-k", type=int, default=10, help="Number of recommendations (default: 10)")
    parser.add_argument("--embeddings", type=Path, help="Path to movie embeddings file")
    parser.add_argument("--metadata", type=Path, help="Path to embeddings metadata file")
    parser.add_argument("--movies", type=Path, help="Path to movies parquet file")

    args = parser.parse_args()

    # Get paths
    default_embeddings, default_metadata, default_movies, default_ratings, default_user_tower = get_default_paths()

    embeddings_path = args.embeddings or default_embeddings
    metadata_path = args.metadata or default_metadata
    movies_path = args.movies or default_movies

    # Initialize CLI
    cli = MovieRecommenderCLI(
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
        movies_path=movies_path,
        ratings_path=default_ratings,
        user_tower_path=default_user_tower,
        top_k=args.top_k,
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
