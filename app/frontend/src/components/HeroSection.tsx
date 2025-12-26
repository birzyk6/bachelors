"use client";

import { useEffect, useRef } from "react";
import Image from "next/image";
import { Play, Info, Star } from "lucide-react";
import gsap from "gsap";

interface Movie {
    tmdb_id: number;
    title: string;
    poster_url: string | null;
    backdrop_url?: string;
    year: number | null;
    vote_average: number;
    overview?: string;
    genres?: string[];
}

interface HeroSectionProps {
    movie: Movie;
    onMovieClick: (tmdbId: number) => void;
    isPersonalized?: boolean;
}

export default function HeroSection({ movie, onMovieClick, isPersonalized }: HeroSectionProps) {
    const containerRef = useRef<HTMLDivElement>(null);
    const hasAnimated = useRef(false);

    const backdropUrl =
        movie.backdrop_url ||
        (movie.poster_url ? movie.poster_url.replace("/w342/", "/original/").replace("/w500/", "/original/") : null);

    useEffect(() => {
        if (!containerRef.current) return;

        const ctx = gsap.context(() => {
            const tl = gsap.timeline();

            // Animate background with ken burns effect
            tl.fromTo(
                ".hero-bg",
                { scale: 1.1, autoAlpha: 0 },
                { scale: 1, autoAlpha: 1, duration: 1.5, ease: "power2.out" }
            );

            // Animate title
            tl.fromTo(
                ".hero-title",
                { autoAlpha: 0, y: 30 },
                { autoAlpha: 1, y: 0, duration: 0.8, ease: "power3.out" },
                "-=0.8"
            );

            // Animate meta info
            tl.fromTo(
                ".hero-meta",
                { autoAlpha: 0, y: 15 },
                { autoAlpha: 1, y: 0, duration: 0.5, ease: "power2.out" },
                "-=0.4"
            );

            // Animate overview
            tl.fromTo(
                ".hero-overview",
                { autoAlpha: 0, y: 20 },
                { autoAlpha: 1, y: 0, duration: 0.6, ease: "power2.out" },
                "-=0.3"
            );

            // Animate buttons
            tl.fromTo(
                ".hero-actions",
                { autoAlpha: 0, y: 20 },
                { autoAlpha: 1, y: 0, duration: 0.5, ease: "power2.out" },
                "-=0.3"
            );

            // Subtle continuous background animation
            gsap.to(".hero-bg", {
                scale: 1.05,
                duration: 20,
                ease: "none",
                repeat: -1,
                yoyo: true,
            });
        }, containerRef);

        return () => ctx.revert();
    }, [movie.tmdb_id]);

    return (
        <div ref={containerRef} className="relative h-[70vh] min-h-[500px] max-h-[800px] overflow-hidden">
            {/* Background Image */}
            {backdropUrl && (
                <div className="hero-bg absolute inset-0 invisible">
                    <Image src={backdropUrl} alt={movie.title} fill className="object-cover object-top" priority />
                    {/* Gradient overlays */}
                    <div className="absolute inset-0 bg-gradient-to-r from-background via-background/80 to-transparent" />
                    <div className="absolute inset-0 bg-gradient-to-t from-background via-transparent to-background/30" />
                </div>
            )}

            {/* Content */}
            <div className="relative h-full container mx-auto px-4 flex items-center">
                <div className="max-w-2xl space-y-6">
                    {/* Badge */}
                    {isPersonalized && (
                        <span className="inline-flex items-center gap-2 px-4 py-1.5 bg-primary/90 backdrop-blur-sm rounded-full text-sm font-medium text-primary-foreground shadow-lg shadow-primary/25">
                            <Star className="w-4 h-4" />
                            Top Pick For You
                        </span>
                    )}

                    {/* Title */}
                    <h1 className="hero-title text-5xl md:text-6xl lg:text-7xl font-bold leading-tight invisible drop-shadow-2xl">
                        {movie.title}
                    </h1>

                    {/* Meta info */}
                    <div className="hero-meta flex items-center gap-4 text-muted-foreground invisible">
                        {movie.year && (
                            <span className="font-medium text-foreground bg-secondary/50 px-3 py-1 rounded-full">
                                {movie.year}
                            </span>
                        )}
                        {movie.vote_average > 0 && (
                            <span className="flex items-center gap-1 bg-secondary/50 px-3 py-1 rounded-full">
                                <Star className="w-4 h-4 text-yellow-400 fill-yellow-400" />
                                {movie.vote_average.toFixed(1)}
                            </span>
                        )}
                        {movie.genres && movie.genres.length > 0 && (
                            <span className="hidden sm:inline">{movie.genres.slice(0, 3).join(" • ")}</span>
                        )}
                    </div>

                    {/* Overview */}
                    {movie.overview && (
                        <p className="hero-overview text-lg text-muted-foreground line-clamp-3 leading-relaxed invisible">
                            {movie.overview}
                        </p>
                    )}

                    {/* Actions */}
                    <div className="hero-actions flex items-center gap-4 pt-2 invisible">
                        <button
                            onClick={() => onMovieClick(movie.tmdb_id)}
                            className="flex items-center gap-2 px-8 py-3 bg-primary text-primary-foreground rounded-lg font-semibold hover:bg-primary/90 hover:scale-105 active:scale-95 transition-all shadow-lg shadow-primary/25"
                        >
                            <Play className="w-5 h-5 fill-current" />
                            View Details
                        </button>
                        <button
                            onClick={() => onMovieClick(movie.tmdb_id)}
                            className="flex items-center gap-2 px-8 py-3 bg-secondary/80 backdrop-blur-sm text-foreground rounded-lg font-semibold hover:bg-secondary hover:scale-105 active:scale-95 transition-all"
                        >
                            <Info className="w-5 h-5" />
                            More Info
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}
