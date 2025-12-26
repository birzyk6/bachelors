"use client";

import { useState, useEffect, useRef } from "react";
import { Star, History, ChevronLeft, ChevronRight, X } from "lucide-react";
import Image from "next/image";
import { api } from "@/lib/api";
import gsap from "gsap";

interface Rating {
    tmdb_id: number;
    rating: number;
    created_at: string;
    movie?: {
        title: string;
        poster_url: string | null;
        year: number | null;
    };
}

interface UserRatingsHistoryProps {
    userId: number;
    onMovieClick: (tmdbId: number) => void;
}

export default function UserRatingsHistory({ userId, onMovieClick }: UserRatingsHistoryProps) {
    const [ratings, setRatings] = useState<Rating[]>([]);
    const [loading, setLoading] = useState(true);
    const [showModal, setShowModal] = useState(false);
    const scrollRef = useRef<HTMLDivElement>(null);
    const [showLeftArrow, setShowLeftArrow] = useState(false);
    const [showRightArrow, setShowRightArrow] = useState(true);

    const modalBackdropRef = useRef<HTMLDivElement>(null);
    const modalContentRef = useRef<HTMLDivElement>(null);

    // GSAP modal animation
    useEffect(() => {
        if (showModal && modalBackdropRef.current && modalContentRef.current) {
            const ctx = gsap.context(() => {
                gsap.fromTo(
                    modalBackdropRef.current,
                    { opacity: 0 },
                    { opacity: 1, duration: 0.3, ease: "power2.out" }
                );

                gsap.fromTo(
                    modalContentRef.current,
                    { opacity: 0, y: 50, scale: 0.95 },
                    { opacity: 1, y: 0, scale: 1, duration: 0.4, ease: "power3.out", delay: 0.1 }
                );

                // Animate grid items
                gsap.fromTo(
                    ".modal-movie-item",
                    { opacity: 0, y: 20, scale: 0.9 },
                    { opacity: 1, y: 0, scale: 1, duration: 0.3, stagger: 0.03, ease: "power2.out", delay: 0.3 }
                );
            });

            return () => ctx.revert();
        }
    }, [showModal]);

    useEffect(() => {
        loadRatings();
    }, [userId]);

    const loadRatings = async () => {
        setLoading(true);
        try {
            const data = await api.getUserRatings(userId);
            const ratingsWithMovies = data.ratings || [];

            // Load movie details in parallel (limit to 30 for performance)
            const enrichedRatings = await Promise.all(
                ratingsWithMovies.slice(0, 30).map(async (rating: Rating) => {
                    try {
                        const movie = await api.getMovie(rating.tmdb_id);
                        return {
                            ...rating,
                            movie: {
                                title: movie.title,
                                poster_url: movie.poster_url,
                                year: movie.year,
                            },
                        };
                    } catch {
                        return {
                            ...rating,
                            movie: {
                                title: `Movie #${rating.tmdb_id}`,
                                poster_url: null,
                                year: null,
                            },
                        };
                    }
                })
            );

            setRatings(enrichedRatings);
        } catch (error) {
            console.error("Failed to load ratings:", error);
        } finally {
            setLoading(false);
        }
    };

    const handleScroll = () => {
        if (scrollRef.current) {
            const { scrollLeft, scrollWidth, clientWidth } = scrollRef.current;
            setShowLeftArrow(scrollLeft > 0);
            setShowRightArrow(scrollLeft < scrollWidth - clientWidth - 10);
        }
    };

    const scroll = (direction: "left" | "right") => {
        if (scrollRef.current) {
            const scrollAmount = 400;
            scrollRef.current.scrollBy({
                left: direction === "left" ? -scrollAmount : scrollAmount,
                behavior: "smooth",
            });
        }
    };

    // Group ratings by rating value
    const likedMovies = ratings.filter((r) => r.rating >= 4);
    const displayMovies = likedMovies.slice(0, 15);

    if (loading) {
        return (
            <div className="bg-card/50 rounded-2xl border border-border p-5">
                <div className="flex items-center gap-3 mb-4">
                    <div className="p-2 bg-primary/20 rounded-lg">
                        <History className="w-5 h-5 text-primary" />
                    </div>
                    <div>
                        <h3 className="text-lg font-semibold">Your Liked Movies</h3>
                        <p className="text-sm text-muted-foreground">Loading...</p>
                    </div>
                </div>
                <div className="flex gap-3 overflow-hidden">
                    {[1, 2, 3, 4, 5, 6].map((i) => (
                        <div key={i} className="w-28 flex-shrink-0">
                            <div className="aspect-[2/3] rounded-lg skeleton" />
                        </div>
                    ))}
                </div>
            </div>
        );
    }

    if (likedMovies.length === 0) {
        return null;
    }

    return (
        <>
            {/* Inline Preview */}
            <div className="bg-card/50 rounded-2xl border border-border p-5 relative group/section">
                <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                        <div className="p-2 bg-primary/20 rounded-lg">
                            <History className="w-5 h-5 text-primary" />
                        </div>
                        <div>
                            <h3 className="text-lg font-semibold">Your Liked Movies</h3>
                            <p className="text-sm text-muted-foreground">{likedMovies.length} movies rated 4+ stars</p>
                        </div>
                    </div>
                    {likedMovies.length > 6 && (
                        <button
                            onClick={() => setShowModal(true)}
                            className="text-sm text-primary hover:text-primary/80 transition-colors flex items-center gap-1"
                        >
                            View All
                            <ChevronRight className="w-4 h-4" />
                        </button>
                    )}
                </div>

                {/* Horizontal scroll container */}
                <div className="relative -mx-2">
                    {/* Left Arrow */}
                    {showLeftArrow && (
                        <button
                            onClick={() => scroll("left")}
                            className="absolute left-0 top-1/2 -translate-y-1/2 z-10 w-10 h-10 flex items-center justify-center bg-black/80 hover:bg-black rounded-full shadow-lg opacity-0 group-hover/section:opacity-100 transition-opacity"
                        >
                            <ChevronLeft className="w-5 h-5" />
                        </button>
                    )}

                    {/* Right Arrow */}
                    {showRightArrow && displayMovies.length > 5 && (
                        <button
                            onClick={() => scroll("right")}
                            className="absolute right-0 top-1/2 -translate-y-1/2 z-10 w-10 h-10 flex items-center justify-center bg-black/80 hover:bg-black rounded-full shadow-lg opacity-0 group-hover/section:opacity-100 transition-opacity"
                        >
                            <ChevronRight className="w-5 h-5" />
                        </button>
                    )}

                    <div ref={scrollRef} onScroll={handleScroll} className="flex gap-3 movie-scroll-container px-2">
                        {displayMovies.map((rating) => (
                            <button
                                key={rating.tmdb_id}
                                onClick={() => onMovieClick(rating.tmdb_id)}
                                className="w-28 flex-shrink-0 group text-left"
                            >
                                <div className="relative aspect-[2/3] rounded-lg overflow-hidden bg-secondary">
                                    {rating.movie?.poster_url ? (
                                        <Image
                                            src={rating.movie.poster_url}
                                            alt={rating.movie?.title || ""}
                                            fill
                                            className="object-cover group-hover:scale-105 transition-transform"
                                            sizes="112px"
                                        />
                                    ) : (
                                        <div className="absolute inset-0 flex items-center justify-center text-3xl">
                                            🎬
                                        </div>
                                    )}
                                    {/* Rating badge */}
                                    <div className="absolute bottom-1.5 right-1.5 flex items-center gap-0.5 px-1.5 py-0.5 bg-black/80 rounded text-xs font-medium">
                                        <Star className="w-3 h-3 text-yellow-400 fill-yellow-400" />
                                        {rating.rating}
                                    </div>
                                    {/* Hover overlay */}
                                    <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                                        <span className="text-xs font-medium">View</span>
                                    </div>
                                </div>
                                <p className="mt-1.5 text-xs text-muted-foreground line-clamp-1 group-hover:text-foreground transition-colors">
                                    {rating.movie?.title}
                                </p>
                            </button>
                        ))}
                    </div>
                </div>
            </div>

            {/* Full Modal */}
            {showModal && (
                <div className="fixed inset-0 z-50 overflow-y-auto">
                    <div
                        ref={modalBackdropRef}
                        className="fixed inset-0 bg-black/80 modal-backdrop opacity-0"
                        onClick={() => setShowModal(false)}
                    />
                    <div className="relative min-h-screen flex items-start justify-center p-4 pt-20">
                        <div
                            ref={modalContentRef}
                            className="relative w-full max-w-5xl bg-card rounded-2xl overflow-hidden shadow-2xl border border-border opacity-0"
                        >
                            {/* Header */}
                            <div className="flex items-center justify-between p-6 border-b border-border">
                                <div className="flex items-center gap-3">
                                    <div className="p-2 bg-primary/20 rounded-lg">
                                        <History className="w-6 h-6 text-primary" />
                                    </div>
                                    <div>
                                        <h2 className="text-xl font-bold">Your Liked Movies</h2>
                                        <p className="text-sm text-muted-foreground">
                                            {likedMovies.length} movies rated 4+ stars
                                        </p>
                                    </div>
                                </div>
                                <button
                                    onClick={() => setShowModal(false)}
                                    className="p-2 hover:bg-secondary rounded-lg transition-colors"
                                >
                                    <X className="w-5 h-5" />
                                </button>
                            </div>

                            {/* Grid of movies */}
                            <div className="p-6 max-h-[70vh] overflow-y-auto">
                                <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-6 gap-4">
                                    {likedMovies.map((rating) => (
                                        <button
                                            key={rating.tmdb_id}
                                            onClick={() => {
                                                onMovieClick(rating.tmdb_id);
                                                setShowModal(false);
                                            }}
                                            className="modal-movie-item group text-left opacity-0"
                                        >
                                            <div className="relative aspect-[2/3] rounded-lg overflow-hidden bg-secondary">
                                                {rating.movie?.poster_url ? (
                                                    <Image
                                                        src={rating.movie.poster_url}
                                                        alt={rating.movie?.title || ""}
                                                        fill
                                                        className="object-cover group-hover:scale-105 transition-transform"
                                                        sizes="150px"
                                                    />
                                                ) : (
                                                    <div className="absolute inset-0 flex items-center justify-center text-4xl">
                                                        🎬
                                                    </div>
                                                )}
                                                <div className="absolute bottom-2 right-2 flex items-center gap-1 px-2 py-1 bg-black/80 rounded text-sm font-medium">
                                                    <Star className="w-3.5 h-3.5 text-yellow-400 fill-yellow-400" />
                                                    {rating.rating}
                                                </div>
                                                <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                                                    <span className="text-sm font-medium">View Details</span>
                                                </div>
                                            </div>
                                            <p className="mt-2 text-sm text-muted-foreground line-clamp-2 group-hover:text-foreground transition-colors">
                                                {rating.movie?.title}
                                            </p>
                                            {rating.movie?.year && (
                                                <p className="text-xs text-muted-foreground/70">{rating.movie.year}</p>
                                            )}
                                        </button>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </>
    );
}
