"use client";

import MovieCard, { MovieCardSkeleton } from "./MovieCard";

interface Movie {
    tmdb_id: number;
    title: string;
    poster_url: string | null;
    year: number | null;
    vote_average: number;
}

interface MovieGridProps {
    movies: Movie[];
    onMovieClick: (tmdbId: number) => void;
    loading?: boolean;
}

export default function MovieGrid({ movies, onMovieClick, loading }: MovieGridProps) {
    if (loading) {
        return (
            <div className="scroll-container">
                <div className="flex gap-4 pb-4">
                    {Array.from({ length: 8 }).map((_, i) => (
                        <MovieCardSkeleton key={i} />
                    ))}
                </div>
            </div>
        );
    }

    if (movies.length === 0) {
        return <div className="text-center py-8 text-muted-foreground">No movies found</div>;
    }

    return (
        <div className="movie-scroll-container">
            <div className="flex gap-4">
                {movies.map((movie) => (
                    <MovieCard
                        key={movie.tmdb_id}
                        tmdbId={movie.tmdb_id}
                        title={movie.title}
                        posterUrl={movie.poster_url}
                        year={movie.year}
                        voteAverage={movie.vote_average}
                        onClick={onMovieClick}
                    />
                ))}
            </div>
        </div>
    );
}
