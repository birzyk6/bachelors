"use client";

import { useState, useEffect, useCallback } from "react";
import Header from "@/components/Header";
import UserSelector from "@/components/UserSelector";
import OnboardingWizard from "@/components/OnboardingWizard";
import RecommendationSection from "@/components/RecommendationSection";
import MovieModal from "@/components/MovieModal";
import HeroSection from "@/components/HeroSection";
import UserRatingsHistory from "@/components/UserRatingsHistory";
import UserSettings from "@/components/UserSettings";
import { api } from "@/lib/api";

interface User {
    id: number;
    movielens_user_id: number | null;
    display_name: string;
    ratings_count: number;
}

interface Movie {
    tmdb_id: number;
    title: string;
    poster_url: string | null;
    year: number | null;
    vote_average: number;
    genres: string[];
    score?: number;
    overview?: string;
    backdrop_url?: string;
}

interface BecauseWatched {
    based_on: number;
    based_on_title?: string;
    movies: Movie[];
}

const GENRES = ["Action", "Comedy", "Drama", "Horror", "Science Fiction", "Romance", "Thriller", "Animation"];

export default function Home() {
    const [currentUser, setCurrentUser] = useState<User | null>(null);
    const [showOnboarding, setShowOnboarding] = useState(false);
    const [showUserSettings, setShowUserSettings] = useState(false);
    const [selectedMovie, setSelectedMovie] = useState<number | null>(null);

    // Recommendation states
    const [forYouMovies, setForYouMovies] = useState<Movie[]>([]);
    const [trendingMovies, setTrendingMovies] = useState<Movie[]>([]);
    const [topRatedMovies, setTopRatedMovies] = useState<Movie[]>([]);
    const [becauseWatched, setBecauseWatched] = useState<BecauseWatched | null>(null);
    const [genreRecommendations, setGenreRecommendations] = useState<{ [key: string]: Movie[] }>({});
    const [heroMovie, setHeroMovie] = useState<Movie | null>(null);

    const [loading, setLoading] = useState(false);

    // Load recommendations when user changes
    useEffect(() => {
        if (currentUser) {
            loadRecommendations();
        } else {
            setForYouMovies([]);
            setBecauseWatched(null);
            setGenreRecommendations({});
        }
    }, [currentUser]);

    // Load trending and top rated on mount
    useEffect(() => {
        loadTrending();
        loadTopRated();
    }, []);

    const loadRecommendations = async () => {
        if (!currentUser) return;

        setLoading(true);
        try {
            // Load all recommendations in parallel
            const [forYouData, becauseData, ...genreData] = await Promise.all([
                api.getForYou(currentUser.id, 30),
                api.getBecauseYouWatched(currentUser.id, 15),
                ...GENRES.slice(0, 4).map((genre) =>
                    api
                        .getByGenre(genre, currentUser.id, 15)
                        .then((data) => ({ genre, movies: data.recommendations || [] }))
                ),
            ]);

            setForYouMovies(forYouData.recommendations || []);

            // Set hero movie from top recommendation
            if (forYouData.recommendations && forYouData.recommendations.length > 0) {
                const topRec = forYouData.recommendations[0];
                if (topRec.backdrop_url || topRec.poster_url) {
                    setHeroMovie(topRec);
                }
            }

            if (becauseData.movies && becauseData.movies.length > 0) {
                // Try to get the title of the based_on movie
                try {
                    const baseMovie = await api.getMovie(becauseData.based_on);
                    setBecauseWatched({
                        ...becauseData,
                        based_on_title: baseMovie.title,
                    });
                } catch {
                    setBecauseWatched(becauseData);
                }
            }

            // Set genre recommendations
            const genres: { [key: string]: Movie[] } = {};
            genreData.forEach((result: { genre: string; movies: Movie[] }) => {
                if (result.movies && result.movies.length > 0) {
                    genres[result.genre] = result.movies;
                }
            });
            setGenreRecommendations(genres);
        } catch (error) {
            console.error("Failed to load recommendations:", error);
        } finally {
            setLoading(false);
        }
    };

    const loadTrending = async () => {
        try {
            const data = await api.getTrending(20);
            setTrendingMovies(data.trending || []);

            // Set hero movie from trending if no user
            if (!heroMovie && data.trending && data.trending.length > 0) {
                setHeroMovie(data.trending[0]);
            }
        } catch (error) {
            console.error("Failed to load trending:", error);
        }
    };

    const loadTopRated = async () => {
        try {
            const data = await api.getPopularMovies();
            setTopRatedMovies(data.movies || []);
        } catch (error) {
            console.error("Failed to load top rated:", error);
        }
    };

    const handleUserSelect = (user: User | null) => {
        setCurrentUser(user);
        setShowOnboarding(false);
    };

    const handleStartOnboarding = () => {
        setShowOnboarding(true);
    };

    const handleOnboardingComplete = (user: User) => {
        setCurrentUser(user);
        setShowOnboarding(false);
    };

    const handleMovieClick = useCallback((tmdbId: number) => {
        setSelectedMovie(tmdbId);
    }, []);

    const handleCloseModal = () => {
        setSelectedMovie(null);
    };

    const handleRateMovie = async (tmdbId: number, rating: number) => {
        if (!currentUser) return;

        try {
            await api.rateMovie(currentUser.id, tmdbId, rating);
            // Refresh recommendations after rating
            loadRecommendations();
        } catch (error) {
            console.error("Failed to rate movie:", error);
        }
    };

    return (
        <main className="min-h-screen">
            <Header
                currentUser={currentUser}
                onUserSelect={handleUserSelect}
                onStartOnboarding={handleStartOnboarding}
                onMovieClick={handleMovieClick}
                onOpenSettings={() => setShowUserSettings(true)}
            />

            {showOnboarding ? (
                <div className="container mx-auto px-4 py-8">
                    <OnboardingWizard onComplete={handleOnboardingComplete} onCancel={() => setShowOnboarding(false)} />
                </div>
            ) : (
                <>
                    {/* Hero Section */}
                    {heroMovie && (
                        <HeroSection movie={heroMovie} onMovieClick={handleMovieClick} isPersonalized={!!currentUser} />
                    )}

                    <div className="container mx-auto px-4 py-8 space-y-12">
                        {/* Welcome message for non-logged in users */}
                        {!currentUser && (
                            <div className="text-center py-12">
                                <div className="relative inline-block mb-6">
                                    <div className="absolute -inset-1 bg-gradient-to-r from-primary/50 via-primary to-primary/50 rounded-full blur-xl opacity-50 animate-pulse" />
                                    <span className="relative px-4 py-2 rounded-full bg-primary/10 border border-primary/30 text-primary text-sm font-medium">
                                        AI-Powered Recommendations
                                    </span>
                                </div>
                                <h2 className="text-5xl md:text-6xl font-bold mb-6 gradient-text">
                                    Discover Your Next
                                    <br />
                                    Favorite Movie
                                </h2>
                                <p className="text-muted-foreground mb-10 max-w-2xl mx-auto text-lg leading-relaxed">
                                    Get personalized movie recommendations powered by advanced AI. Select an existing
                                    user to explore their taste, or create your own profile.
                                </p>
                                <UserSelector
                                    onUserSelect={handleUserSelect}
                                    onStartOnboarding={handleStartOnboarding}
                                />
                            </div>
                        )}

                        {/* User's Rating History */}
                        {currentUser && <UserRatingsHistory userId={currentUser.id} onMovieClick={handleMovieClick} />}

                        {/* Personalized: For You */}
                        {currentUser && forYouMovies.length > 0 && (
                            <RecommendationSection
                                title="For You"
                                subtitle="Personalized picks based on your taste"
                                movies={forYouMovies}
                                onMovieClick={handleMovieClick}
                                loading={loading}
                                icon="sparkles"
                            />
                        )}

                        {/* Because You Watched */}
                        {currentUser && becauseWatched && becauseWatched.movies.length > 0 && (
                            <RecommendationSection
                                title={`Because You Watched ${becauseWatched.based_on_title || "Recently"}`}
                                subtitle="Similar movies you might enjoy"
                                movies={becauseWatched.movies}
                                onMovieClick={handleMovieClick}
                                icon="film"
                            />
                        )}

                        {/* Trending Now */}
                        {trendingMovies.length > 0 && (
                            <RecommendationSection
                                title="Trending Now"
                                subtitle="What everyone's watching"
                                movies={trendingMovies}
                                onMovieClick={handleMovieClick}
                                icon="trending"
                            />
                        )}

                        {/* Genre Recommendations */}
                        {currentUser && Object.keys(genreRecommendations).length > 0 && (
                            <>
                                {Object.entries(genreRecommendations).map(
                                    ([genre, movies]) =>
                                        movies.length > 0 && (
                                            <RecommendationSection
                                                key={genre}
                                                title={`Top ${genre}`}
                                                subtitle={`Best ${genre.toLowerCase()} picks for you`}
                                                movies={movies}
                                                onMovieClick={handleMovieClick}
                                                icon="genre"
                                            />
                                        )
                                )}
                            </>
                        )}

                        {/* Top Rated */}
                        {topRatedMovies.length > 0 && (
                            <RecommendationSection
                                title="Top Rated"
                                subtitle="Critically acclaimed movies"
                                movies={topRatedMovies}
                                onMovieClick={handleMovieClick}
                                icon="star"
                            />
                        )}
                    </div>
                </>
            )}

            {/* Movie Modal */}
            {selectedMovie && (
                <MovieModal
                    tmdbId={selectedMovie}
                    userId={currentUser?.id}
                    onClose={handleCloseModal}
                    onRate={currentUser ? handleRateMovie : undefined}
                    onMovieClick={handleMovieClick}
                />
            )}

            {/* User Settings Modal */}
            {showUserSettings && currentUser && (
                <UserSettings
                    userId={currentUser.id}
                    userName={currentUser.display_name}
                    onClose={() => setShowUserSettings(false)}
                    onMovieClick={handleMovieClick}
                />
            )}
        </main>
    );
}
