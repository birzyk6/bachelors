"use client";

import { useState, useEffect, useRef } from "react";
import { X, Star, Clock, Calendar, ExternalLink } from "lucide-react";
import Image from "next/image";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import MovieGrid from "./MovieGrid";
import gsap from "gsap";

interface MovieDetails {
    tmdb_id: number;
    title: string;
    original_title: string;
    overview: string;
    poster_url: string | null;
    backdrop_url: string | null;
    release_date: string;
    year: number | null;
    vote_average: number;
    vote_count: number;
    genres: string[];
    keywords: string[];
    runtime: number | null;
    tagline: string | null;
}

interface SimilarMovie {
    tmdb_id: number;
    title: string;
    poster_url: string | null;
    year: number | null;
    vote_average: number;
}

interface MovieModalProps {
    tmdbId: number;
    userId?: number;
    onClose: () => void;
    onRate?: (tmdbId: number, rating: number) => void;
    onMovieClick: (tmdbId: number) => void;
}

export default function MovieModal({ tmdbId, userId, onClose, onRate, onMovieClick }: MovieModalProps) {
    const [movie, setMovie] = useState<MovieDetails | null>(null);
    const [similarMovies, setSimilarMovies] = useState<SimilarMovie[]>([]);
    const [loading, setLoading] = useState(true);
    const [userRating, setUserRating] = useState<number | null>(null);
    const [hoverRating, setHoverRating] = useState<number | null>(null);

    const modalRef = useRef<HTMLDivElement>(null);
    const backdropRef = useRef<HTMLDivElement>(null);
    const contentRef = useRef<HTMLDivElement>(null);

    // GSAP entrance animation
    useEffect(() => {
        const ctx = gsap.context(() => {
            // Animate backdrop
            gsap.fromTo(backdropRef.current, { opacity: 0 }, { opacity: 1, duration: 0.3, ease: "power2.out" });

            // Animate modal content
            gsap.fromTo(
                contentRef.current,
                { opacity: 0, y: 50, scale: 0.95 },
                {
                    opacity: 1,
                    y: 0,
                    scale: 1,
                    duration: 0.4,
                    ease: "power3.out",
                    delay: 0.1,
                }
            );
        });

        return () => ctx.revert();
    }, []);

    // Animate content when movie loads
    useEffect(() => {
        if (movie && contentRef.current) {
            const ctx = gsap.context(() => {
                // Animate poster
                gsap.fromTo(
                    ".modal-poster",
                    { opacity: 0, x: -30 },
                    { opacity: 1, x: 0, duration: 0.5, ease: "power2.out" }
                );

                // Animate details with stagger
                gsap.fromTo(
                    ".modal-detail",
                    { opacity: 0, y: 20 },
                    {
                        opacity: 1,
                        y: 0,
                        duration: 0.4,
                        stagger: 0.08,
                        ease: "power2.out",
                        delay: 0.2,
                    }
                );

                // Animate genres
                gsap.fromTo(
                    ".modal-genre",
                    { opacity: 0, scale: 0.8 },
                    {
                        opacity: 1,
                        scale: 1,
                        duration: 0.3,
                        stagger: 0.05,
                        ease: "back.out(1.7)",
                        delay: 0.4,
                    }
                );
            }, contentRef);

            return () => ctx.revert();
        }
    }, [movie]);

    useEffect(() => {
        loadMovie();
        loadSimilar();
        if (userId) {
            loadUserRating();
        }
    }, [tmdbId, userId]);

    const loadUserRating = async () => {
        if (!userId) return;
        try {
            const data = await api.getUserRating(userId, tmdbId);
            if (data.rating) {
                setUserRating(data.rating);
            }
        } catch (error) {
            console.error("Failed to load user rating:", error);
        }
    };

    const loadMovie = async () => {
        setLoading(true);
        try {
            const data = await api.getMovie(tmdbId);
            setMovie(data);
        } catch (error) {
            console.error("Failed to load movie:", error);
        } finally {
            setLoading(false);
        }
    };

    const loadSimilar = async () => {
        try {
            const data = await api.getSimilarMovies(tmdbId);
            setSimilarMovies(data.similar || []);
        } catch (error) {
            console.error("Failed to load similar movies:", error);
        }
    };

    const handleRate = (rating: number) => {
        setUserRating(rating);
        if (onRate) {
            onRate(tmdbId, rating);
        }
    };

    const handleSimilarClick = (id: number) => {
        onMovieClick(id);
    };

    useEffect(() => {
        const handleEscape = (e: KeyboardEvent) => {
            if (e.key === "Escape") onClose();
        };
        window.addEventListener("keydown", handleEscape);
        return () => window.removeEventListener("keydown", handleEscape);
    }, [onClose]);

    const formatRuntime = (minutes: number) => {
        const hours = Math.floor(minutes / 60);
        const mins = minutes % 60;
        return `${hours}h ${mins}m`;
    };

    return (
        <div ref={modalRef} className="fixed inset-0 z-50 overflow-y-auto">
            {/* Backdrop */}
            <div ref={backdropRef} className="fixed inset-0 bg-black/80 modal-backdrop opacity-0" onClick={onClose} />

            {/* Modal */}
            <div className="relative min-h-screen flex items-start justify-center p-4 pt-20">
                <div
                    ref={contentRef}
                    className="relative w-full max-w-4xl bg-background border border-border rounded-xl overflow-hidden shadow-2xl opacity-0"
                >
                    {/* Close button */}
                    <Button
                        variant="ghost"
                        size="icon"
                        onClick={onClose}
                        className="absolute top-4 right-4 z-10 bg-black/50 hover:bg-black/70 rounded-full"
                    >
                        <X className="w-5 h-5" />
                    </Button>

                    {loading ? (
                        <div className="p-8 text-center">
                            <div className="animate-spin w-8 h-8 border-2 border-primary border-t-transparent rounded-full mx-auto" />
                            <p className="mt-4 text-muted-foreground">Loading...</p>
                        </div>
                    ) : movie ? (
                        <>
                            {/* Backdrop */}
                            {movie.backdrop_url && (
                                <div className="relative h-64 md:h-80">
                                    <Image src={movie.backdrop_url} alt={movie.title} fill className="object-cover" />
                                    <div className="absolute inset-0 bg-gradient-to-t from-background via-background/50 to-transparent" />
                                </div>
                            )}

                            {/* Content */}
                            <div className="relative p-6 md:p-8">
                                <div className="flex flex-col md:flex-row gap-6">
                                    {/* Poster */}
                                    <div className="modal-poster flex-shrink-0 w-48 mx-auto md:mx-0 opacity-0">
                                        <div className="relative aspect-[2/3] rounded-lg overflow-hidden shadow-lg">
                                            {movie.poster_url ? (
                                                <Image
                                                    src={movie.poster_url}
                                                    alt={movie.title}
                                                    fill
                                                    className="object-cover"
                                                />
                                            ) : (
                                                <div className="absolute inset-0 bg-secondary flex items-center justify-center">
                                                    <span className="text-6xl">🎬</span>
                                                </div>
                                            )}
                                        </div>
                                    </div>

                                    {/* Details */}
                                    <div className="flex-1 space-y-4">
                                        {/* Title */}
                                        <div className="modal-detail opacity-0">
                                            <h2 className="text-3xl font-bold">{movie.title}</h2>
                                            {movie.tagline && (
                                                <p className="text-muted-foreground italic mt-1">{movie.tagline}</p>
                                            )}
                                        </div>

                                        {/* Meta info */}
                                        <div className="modal-detail flex flex-wrap items-center gap-4 text-sm opacity-0">
                                            {movie.year && (
                                                <span className="flex items-center gap-1 text-muted-foreground">
                                                    <Calendar className="w-4 h-4" />
                                                    {movie.year}
                                                </span>
                                            )}
                                            {movie.runtime && (
                                                <span className="flex items-center gap-1 text-muted-foreground">
                                                    <Clock className="w-4 h-4" />
                                                    {formatRuntime(movie.runtime)}
                                                </span>
                                            )}
                                            {movie.vote_average > 0 && (
                                                <span className="flex items-center gap-1">
                                                    <Star className="w-4 h-4 text-yellow-400 fill-yellow-400" />
                                                    <span className="font-medium">{movie.vote_average.toFixed(1)}</span>
                                                    <span className="text-muted-foreground">
                                                        ({movie.vote_count.toLocaleString()})
                                                    </span>
                                                </span>
                                            )}
                                        </div>

                                        {/* Genres */}
                                        {movie.genres && movie.genres.length > 0 && (
                                            <div className="modal-detail flex flex-wrap gap-2 opacity-0">
                                                {movie.genres.map((genre) => (
                                                    <Badge
                                                        key={genre}
                                                        variant="secondary"
                                                        className="modal-genre opacity-0"
                                                    >
                                                        {genre}
                                                    </Badge>
                                                ))}
                                            </div>
                                        )}

                                        {/* Overview */}
                                        <p className="modal-detail text-muted-foreground leading-relaxed opacity-0">
                                            {movie.overview}
                                        </p>

                                        {/* User rating */}
                                        {onRate && (
                                            <div className="pt-4 border-t border-border">
                                                <p className="text-sm text-muted-foreground mb-2">Your Rating</p>
                                                <div className="flex gap-0">
                                                    {[1, 2, 3, 4, 5].map((starPosition) => (
                                                        <div key={starPosition} className="relative w-8 h-8">
                                                            {/* Left half (0.5) */}
                                                            <button
                                                                onClick={() => handleRate(starPosition - 0.5)}
                                                                onMouseEnter={() => setHoverRating(starPosition - 0.5)}
                                                                onMouseLeave={() => setHoverRating(null)}
                                                                className="absolute left-0 top-0 w-4 h-8 z-10"
                                                            />
                                                            {/* Right half (1.0) */}
                                                            <button
                                                                onClick={() => handleRate(starPosition)}
                                                                onMouseEnter={() => setHoverRating(starPosition)}
                                                                onMouseLeave={() => setHoverRating(null)}
                                                                className="absolute right-0 top-0 w-4 h-8 z-10"
                                                            />
                                                            {/* Star visual */}
                                                            <Star
                                                                className={cn(
                                                                    "w-8 h-8 transition-colors pointer-events-none",
                                                                    (hoverRating || userRating || 0) >= starPosition
                                                                        ? "text-yellow-400 fill-yellow-400"
                                                                        : (hoverRating || userRating || 0) >=
                                                                            starPosition - 0.5
                                                                          ? "text-yellow-400 fill-yellow-400"
                                                                          : "text-muted-foreground"
                                                                )}
                                                                style={{
                                                                    clipPath:
                                                                        (hoverRating || userRating || 0) >= starPosition
                                                                            ? "none"
                                                                            : (hoverRating || userRating || 0) >=
                                                                                starPosition - 0.5
                                                                              ? "inset(0 50% 0 0)"
                                                                              : "none",
                                                                }}
                                                            />
                                                            {/* Background star (always gray) */}
                                                            <Star className="w-8 h-8 text-muted-foreground absolute top-0 left-0 -z-10" />
                                                        </div>
                                                    ))}
                                                </div>
                                            </div>
                                        )}

                                        {/* TMDB link */}
                                        <Button variant="link" asChild className="p-0 h-auto text-primary">
                                            <a
                                                href={`https://www.themoviedb.org/movie/${movie.tmdb_id}`}
                                                target="_blank"
                                                rel="noopener noreferrer"
                                            >
                                                View on TMDB
                                                <ExternalLink className="w-4 h-4 ml-2" />
                                            </a>
                                        </Button>
                                    </div>
                                </div>

                                {/* Similar Movies */}
                                {similarMovies.length > 0 && (
                                    <div className="mt-8 pt-6 border-t border-border">
                                        <h3 className="text-xl font-bold mb-4">Similar Movies</h3>
                                        <MovieGrid movies={similarMovies} onMovieClick={handleSimilarClick} />
                                    </div>
                                )}
                            </div>
                        </>
                    ) : (
                        <div className="p-8 text-center text-muted-foreground">Movie not found</div>
                    )}
                </div>
            </div>
        </div>
    );
}
