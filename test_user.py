from model.src.cli.app import MovieRecommenderCLI, get_default_paths

paths = get_default_paths()
print("Paths:", [str(p) for p in paths])

cli = MovieRecommenderCLI(
    embeddings_path=paths[0],
    metadata_path=paths[1],
    movies_path=paths[2],
    ratings_path=paths[3],
    user_tower_path=paths[4],
    top_k=10,
)

if cli.load():
    print()
    print("=== Testing User Commands ===")

    cli.cmd_users("")

    print()
    print("--- Impersonating User 1 ---")
    cli.cmd_user("1")

    print()
    print("--- User History ---")
    cli.cmd_history("10")

    print()
    print("--- Personalized Recommendations ---")
    cli.cmd_recommend("")
