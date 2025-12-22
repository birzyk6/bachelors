"""
Display utilities for CLI output.

Provides colored and formatted output for the terminal.
"""

import sys
from typing import List, Optional, Tuple


class Colors:
    """ANSI color codes for terminal output."""

    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"

    @classmethod
    def disable(cls):
        """Disable colors (for non-TTY output)."""
        cls.HEADER = ""
        cls.BLUE = ""
        cls.CYAN = ""
        cls.GREEN = ""
        cls.YELLOW = ""
        cls.RED = ""
        cls.BOLD = ""
        cls.UNDERLINE = ""
        cls.END = ""


# Disable colors if not a TTY
if not sys.stdout.isatty():
    Colors.disable()


def print_header(text: str) -> None:
    """Print a bold header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{text}{Colors.END}")
    print("─" * len(text))


def print_success(text: str) -> None:
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_error(text: str) -> None:
    """Print error message."""
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_warning(text: str) -> None:
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


def print_info(text: str) -> None:
    """Print info message."""
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")


def print_movie_result(rank: int, movie_info: str, score: Optional[float] = None, highlight: bool = False) -> None:
    """
    Print a single movie result.

    Args:
        rank: Result rank (1-indexed)
        movie_info: Formatted movie info string
        score: Optional similarity score
        highlight: Whether to highlight this result
    """
    color = Colors.YELLOW if highlight else ""
    end = Colors.END if highlight else ""

    score_str = f" {Colors.CYAN}[{score:.4f}]{Colors.END}" if score is not None else ""

    print(f"  {color}{rank:2d}. {movie_info}{end}{score_str}")


def print_recommendations(results: List[Tuple[int, float, str]], title: str = "Recommendations") -> None:
    """
    Print a list of movie recommendations.

    Args:
        results: List of (movie_id, score, formatted_info) tuples
        title: Section title
    """
    print_header(title)

    if not results:
        print_warning("No recommendations found.")
        return

    for i, (movie_id, score, info) in enumerate(results, 1):
        # Negative score indicates a title match (sorted by popularity)
        if score < 0:
            # It's a title match - show ★ MATCH instead of similarity score
            popularity = -score  # We stored -popularity
            print(f"  {i:2d}. {info} {Colors.GREEN}★ MATCH{Colors.END}")
        else:
            # It's a similar movie - show similarity percentage
            similarity_pct = score * 100
            print(f"  {i:2d}. {info} {Colors.CYAN}[{similarity_pct:.0f}% similar]{Colors.END}")

    print()


def print_search_results(results: List[Tuple[int, str]], query: str) -> None:
    """
    Print movie search results.

    Args:
        results: List of (movie_id, formatted_info) tuples
        query: Original search query
    """
    print_header(f'Search Results for "{query}"')

    if not results:
        print_warning(f'No movies found matching "{query}"')
        return

    for i, (movie_id, info) in enumerate(results, 1):
        print(f"  {i:2d}. [{movie_id}] {info}")

    print()


def print_genres(genres: List[str], columns: int = 4) -> None:
    """
    Print available genres in columns.

    Args:
        genres: List of genre names
        columns: Number of columns
    """
    print_header("Available Genres")

    # Calculate column width
    max_len = max(len(g) for g in genres) + 2

    for i, genre in enumerate(genres):
        end = "\n" if (i + 1) % columns == 0 else ""
        print(f"  {genre:<{max_len}}", end=end)

    print("\n")


def print_help() -> None:
    """Print help message with available commands."""
    print_header("Two-Tower Movie Recommender - Help")

    print(
        f"""
{Colors.BOLD}Just type anything!{Colors.END} This works like a search engine.

{Colors.CYAN}Examples:{Colors.END}
  Star Wars              → Find Star Wars movies and similar sci-fi
  romantic comedy        → Find romantic comedies
  action thriller 90s    → Action thrillers from the 90s
  Matrix                 → Movies like The Matrix
  scary movies           → Horror recommendations

{Colors.BOLD}Special Commands:{Colors.END}
  {Colors.CYAN}similar <id>{Colors.END}        Find movies similar to a specific movie ID
                       Example: similar 260

  {Colors.CYAN}info <id>{Colors.END}           Show detailed info about a movie
                       Example: info 260

  {Colors.CYAN}favorites{Colors.END}           Get recommendations based on your favorites
                       (interactive - enter multiple movie IDs)

  {Colors.CYAN}genres{Colors.END}              List all available genres

{Colors.BOLD}User Impersonation:{Colors.END}
  {Colors.CYAN}users{Colors.END}               Show sample active users
  {Colors.CYAN}user <id>{Colors.END}           Impersonate a user (e.g., user 12345)
  {Colors.CYAN}history{Colors.END}             Show user's watch history
  {Colors.CYAN}whoami{Colors.END}              Show current user info
  {Colors.CYAN}logout{Colors.END}              Stop impersonating user

{Colors.BOLD}Personalized Recommendations (when impersonating):{Colors.END}
  {Colors.CYAN}recommend{Colors.END}             General personalized picks
  {Colors.CYAN}recommend popular{Colors.END}     Popular movies matching your taste
  {Colors.CYAN}recommend recent{Colors.END}      Based on your most recent watch
  {Colors.CYAN}recommend discover{Colors.END}    Explore outside your comfort zone
  {Colors.CYAN}recommend gems{Colors.END}        Hidden gems you might love
  {Colors.CYAN}recommend <theme>{Colors.END}     Themed picks (e.g., 'recommend vampires')
                       Examples: recommend space, recommend 90s comedy

{Colors.BOLD}Other:{Colors.END}
  {Colors.CYAN}help{Colors.END}                Show this help message
  {Colors.CYAN}quit{Colors.END}                Exit the program

{Colors.BOLD}How it works:{Colors.END}
  The system uses Two-Tower neural network embeddings to understand
  movie similarity. When you search:

  1. It finds movies matching your text (titles, genres)
  2. Uses their embeddings to find similar movies
  3. Combines results to give you the best recommendations

  When impersonating a user, it uses their learned embedding from
  the Two-Tower model to find movies they would likely enjoy!
"""
    )


def print_welcome(num_movies: int, num_embeddings: int) -> None:
    """Print welcome message."""
    print(
        f"""
{Colors.BOLD}{Colors.CYAN}╔══════════════════════════════════════════════════════════════╗
║           Two-Tower Movie Recommendation System               ║
╚══════════════════════════════════════════════════════════════╝{Colors.END}

{Colors.GREEN}Loaded:{Colors.END} {num_embeddings:,} movie embeddings from {num_movies:,} total movies

{Colors.BOLD}Just type anything to search!{Colors.END} Examples:
  • {Colors.CYAN}Star Wars{Colors.END}           - Find Star Wars and similar movies
  • {Colors.CYAN}romantic comedy{Colors.END}    - Browse romantic comedies
  • {Colors.CYAN}scary thriller{Colors.END}     - Horror and thriller recommendations

Type {Colors.CYAN}help{Colors.END} for more options.
"""
    )


def prompt() -> str:
    """Display prompt and get user input."""
    try:
        return input(f"{Colors.BOLD}>>> {Colors.END}").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return "quit"
