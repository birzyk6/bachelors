"use client";

import { useState, useEffect, useRef } from "react";
import { ChevronLeft, ChevronRight, Check, Star } from "lucide-react";
import Image from "next/image";
import { api } from "@/lib/api";
import gsap from "gsap";

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
    vote_average: number;
}

interface OnboardingWizardProps {
    onComplete: (user: User) => void;
    onCancel: () => void;
}

const GENRES = [
    "Action",
    "Adventure",
    "Animation",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Family",
    "Fantasy",
    "History",
    "Horror",
    "Music",
    "Mystery",
    "Romance",
    "Science Fiction",
    "Thriller",
    "War",
    "Western",
];

export default function OnboardingWizard({ onComplete, onCancel }: OnboardingWizardProps) {
    const [step, setStep] = useState(1);
    const [selectedGenres, setSelectedGenres] = useState<string[]>([]);
    const [seedMovies, setSeedMovies] = useState<Movie[]>([]);
    const [ratings, setRatings] = useState<Record<number, number>>({});
    const [loading, setLoading] = useState(false);
    const [loadingMovies, setLoadingMovies] = useState(false);

    const containerRef = useRef<HTMLDivElement>(null);

    // GSAP entrance animation
    useEffect(() => {
        if (!containerRef.current) return;

        const ctx = gsap.context(() => {
            gsap.fromTo(
                containerRef.current,
                { opacity: 0, y: 30 },
                { opacity: 1, y: 0, duration: 0.6, ease: "power3.out" }
            );

            // Animate progress indicator
            gsap.fromTo(
                ".progress-step",
                { opacity: 0, scale: 0.5 },
                {
                    opacity: 1,
                    scale: 1,
                    duration: 0.4,
                    stagger: 0.1,
                    ease: "back.out(1.7)",
                    delay: 0.2,
                }
            );
        });

        return () => ctx.revert();
    }, []);

    // Animate genre buttons on step 1
    useEffect(() => {
        if (step === 1 && containerRef.current) {
            const ctx = gsap.context(() => {
                gsap.fromTo(
                    ".genre-btn",
                    { opacity: 0, scale: 0.8, y: 20 },
                    {
                        opacity: 1,
                        scale: 1,
                        y: 0,
                        duration: 0.3,
                        stagger: 0.03,
                        ease: "power2.out",
                        delay: 0.3,
                    }
                );
            }, containerRef);

            return () => ctx.revert();
        }
    }, [step]);

    // Animate movie cards on step 2
    useEffect(() => {
        if (step === 2 && !loadingMovies && seedMovies.length > 0 && containerRef.current) {
            const ctx = gsap.context(() => {
                gsap.fromTo(
                    ".seed-movie",
                    { opacity: 0, y: 30, scale: 0.9 },
                    {
                        opacity: 1,
                        y: 0,
                        scale: 1,
                        duration: 0.4,
                        stagger: 0.05,
                        ease: "power2.out",
                    }
                );
            }, containerRef);

            return () => ctx.revert();
        }
    }, [step, loadingMovies, seedMovies.length]);

    // Load seed movies when entering step 2
    useEffect(() => {
        if (step === 2 && seedMovies.length === 0) {
            loadSeedMovies();
        }
    }, [step]);

    const loadSeedMovies = async () => {
        setLoadingMovies(true);
        try {
            const data = await api.getSeedMovies(20);
            setSeedMovies(data.movies || []);
        } catch (error) {
            console.error("Failed to load seed movies:", error);
        } finally {
            setLoadingMovies(false);
        }
    };

    const toggleGenre = (genre: string) => {
        setSelectedGenres((prev) => (prev.includes(genre) ? prev.filter((g) => g !== genre) : [...prev, genre]));
    };

    const handleRating = (movieId: number, rating: number) => {
        setRatings((prev) => ({
            ...prev,
            [movieId]: rating,
        }));
    };

    const handleComplete = async () => {
        setLoading(true);
        try {
            const initialRatings = Object.entries(ratings).map(([tmdbId, rating]) => ({
                tmdb_id: parseInt(tmdbId),
                rating,
            }));

            const user = await api.createColdStartUser(selectedGenres, initialRatings);
            onComplete(user);
        } catch (error) {
            console.error("Failed to create user:", error);
        } finally {
            setLoading(false);
        }
    };

    const canProceedFromStep1 = selectedGenres.length >= 3;
    const canComplete = Object.keys(ratings).length >= 5;

    return (
        <div ref={containerRef} className="max-w-2xl mx-auto opacity-0">
            {/* Progress indicator */}
            <div className="flex items-center justify-center gap-4 mb-8">
                <div
                    className={`progress-step flex items-center justify-center w-8 h-8 rounded-full ${step >= 1 ? "bg-primary text-primary-foreground" : "bg-secondary"}`}
                >
                    {step > 1 ? <Check className="w-4 h-4" /> : "1"}
                </div>
                <div className={`progress-step w-16 h-1 rounded ${step > 1 ? "bg-primary" : "bg-secondary"}`} />
                <div
                    className={`progress-step flex items-center justify-center w-8 h-8 rounded-full ${step >= 2 ? "bg-primary text-primary-foreground" : "bg-secondary"}`}
                >
                    2
                </div>
            </div>

            {/* Step 1: Genre Selection */}
            {step === 1 && (
                <div className="space-y-6">
                    <div className="text-center">
                        <h2 className="text-2xl font-bold">What do you like to watch?</h2>
                        <p className="text-muted-foreground mt-2">Select at least 3 genres you enjoy</p>
                    </div>

                    <div className="flex flex-wrap gap-3 justify-center">
                        {GENRES.map((genre) => (
                            <button
                                key={genre}
                                onClick={() => toggleGenre(genre)}
                                className={`genre-btn px-4 py-2 rounded-full transition-all opacity-0 ${
                                    selectedGenres.includes(genre)
                                        ? "bg-primary text-primary-foreground"
                                        : "bg-secondary text-muted-foreground hover:bg-secondary/80 hover:text-foreground"
                                }`}
                            >
                                {genre}
                            </button>
                        ))}
                    </div>

                    <div className="flex justify-between pt-4">
                        <button
                            onClick={onCancel}
                            className="px-6 py-2 text-muted-foreground hover:text-foreground transition-colors"
                        >
                            Cancel
                        </button>
                        <button
                            onClick={() => setStep(2)}
                            disabled={!canProceedFromStep1}
                            className="flex items-center gap-2 px-6 py-2 bg-primary text-primary-foreground hover:bg-primary/90 disabled:bg-secondary disabled:text-muted-foreground rounded-lg transition-colors"
                        >
                            Next
                            <ChevronRight className="w-4 h-4" />
                        </button>
                    </div>
                </div>
            )}

            {/* Step 2: Rate Movies */}
            {step === 2 && (
                <div className="space-y-6">
                    <div className="text-center">
                        <h2 className="text-2xl font-bold">Rate some movies</h2>
                        <p className="text-muted-foreground mt-2">
                            Rate at least 5 movies to help us understand your taste
                        </p>
                    </div>

                    {loadingMovies ? (
                        <div className="text-center py-8 text-muted-foreground">Loading movies...</div>
                    ) : (
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                            {seedMovies.map((movie) => (
                                <div key={movie.tmdb_id} className="seed-movie space-y-2 opacity-0">
                                    {/* Poster */}
                                    <div className="relative aspect-[2/3] rounded-lg overflow-hidden bg-secondary">
                                        {movie.poster_url ? (
                                            <Image
                                                src={movie.poster_url}
                                                alt={movie.title}
                                                fill
                                                className="object-cover"
                                                sizes="150px"
                                            />
                                        ) : (
                                            <div className="absolute inset-0 flex items-center justify-center">
                                                <span className="text-3xl">🎬</span>
                                            </div>
                                        )}
                                    </div>

                                    {/* Title */}
                                    <p className="text-sm font-medium line-clamp-1">{movie.title}</p>

                                    {/* Star rating */}
                                    <div className="star-rating flex gap-0 justify-center">
                                        {[1, 2, 3, 4, 5].map((starPosition) => (
                                            <div key={starPosition} className="relative w-6 h-6">
                                                {/* Left half (0.5) */}
                                                <button
                                                    onClick={() => handleRating(movie.tmdb_id, starPosition - 0.5)}
                                                    className="absolute left-0 top-0 w-3 h-6 z-10"
                                                />
                                                {/* Right half (1.0) */}
                                                <button
                                                    onClick={() => handleRating(movie.tmdb_id, starPosition)}
                                                    className="absolute right-0 top-0 w-3 h-6 z-10"
                                                />
                                                {/* Star visual */}
                                                <Star
                                                    className={`w-6 h-6 transition-colors pointer-events-none ${
                                                        (ratings[movie.tmdb_id] || 0) >= starPosition
                                                            ? "text-yellow-400 fill-yellow-400"
                                                            : (ratings[movie.tmdb_id] || 0) >= starPosition - 0.5
                                                              ? "text-yellow-400 fill-yellow-400"
                                                              : "text-muted-foreground"
                                                    }`}
                                                    style={{
                                                        clipPath:
                                                            (ratings[movie.tmdb_id] || 0) >= starPosition
                                                                ? "none"
                                                                : (ratings[movie.tmdb_id] || 0) >= starPosition - 0.5
                                                                  ? "inset(0 50% 0 0)"
                                                                  : "none",
                                                    }}
                                                />
                                                {/* Background star (always gray) */}
                                                <Star className="w-6 h-6 text-muted-foreground absolute top-0 left-0 -z-10" />
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}

                    <div className="flex justify-between pt-4">
                        <button
                            onClick={() => setStep(1)}
                            className="flex items-center gap-2 px-6 py-2 text-muted-foreground hover:text-foreground transition-colors"
                        >
                            <ChevronLeft className="w-4 h-4" />
                            Back
                        </button>
                        <button
                            onClick={handleComplete}
                            disabled={!canComplete || loading}
                            className="flex items-center gap-2 px-6 py-2 bg-primary text-primary-foreground hover:bg-primary/90 disabled:bg-secondary disabled:text-muted-foreground rounded-lg transition-colors"
                        >
                            {loading ? "Creating..." : "Get Recommendations"}
                            <Check className="w-4 h-4" />
                        </button>
                    </div>

                    <p className="text-center text-sm text-muted-foreground">
                        {Object.keys(ratings).length} of 5 minimum ratings
                    </p>
                </div>
            )}
        </div>
    );
}
