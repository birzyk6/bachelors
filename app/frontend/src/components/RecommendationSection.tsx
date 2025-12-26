"use client";

import { useRef, useState, useEffect } from "react";
import { ChevronRight, ChevronLeft, Sparkles, Film, TrendingUp, Star, Layers } from "lucide-react";
import MovieCard, { MovieCardSkeleton } from "./MovieCard";
import { Button } from "@/components/ui/button";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

// Register plugin
if (typeof window !== "undefined") {
    gsap.registerPlugin(ScrollTrigger);
}

interface Movie {
    tmdb_id: number;
    title: string;
    poster_url: string | null;
    year: number | null;
    vote_average: number;
}

interface RecommendationSectionProps {
    title: string;
    subtitle?: string;
    movies: Movie[];
    onMovieClick: (tmdbId: number) => void;
    loading?: boolean;
    onViewAll?: () => void;
    onLoadMore?: () => void;
    hasMore?: boolean;
    icon?: "sparkles" | "film" | "trending" | "star" | "genre";
}

const iconMap = {
    sparkles: Sparkles,
    film: Film,
    trending: TrendingUp,
    star: Star,
    genre: Layers,
};

export default function RecommendationSection({
    title,
    subtitle,
    movies,
    onMovieClick,
    loading,
    onViewAll,
    onLoadMore,
    hasMore,
    icon,
}: RecommendationSectionProps) {
    const sectionRef = useRef<HTMLElement>(null);
    const scrollRef = useRef<HTMLDivElement>(null);
    const [showLeftArrow, setShowLeftArrow] = useState(false);
    const [showRightArrow, setShowRightArrow] = useState(true);

    const IconComponent = icon ? iconMap[icon] : null;

    // GSAP scroll animation
    useEffect(() => {
        if (!sectionRef.current || loading || movies.length === 0) return;

        const ctx = gsap.context(() => {
            // Animate header
            gsap.fromTo(
                sectionRef.current!.querySelector(".section-header"),
                { autoAlpha: 0, x: -20 },
                {
                    autoAlpha: 1,
                    x: 0,
                    duration: 0.5,
                    ease: "power2.out",
                    scrollTrigger: {
                        trigger: sectionRef.current,
                        start: "top 85%",
                        toggleActions: "play none none none",
                    },
                }
            );

            // Animate movie cards with stagger
            const cards = sectionRef.current!.querySelectorAll(".movie-card-item");
            if (cards.length > 0) {
                gsap.fromTo(
                    cards,
                    { autoAlpha: 0, y: 20 },
                    {
                        autoAlpha: 1,
                        y: 0,
                        duration: 0.4,
                        stagger: 0.05,
                        ease: "power2.out",
                        scrollTrigger: {
                            trigger: sectionRef.current,
                            start: "top 80%",
                            toggleActions: "play none none none",
                        },
                    }
                );
            }
        }, sectionRef);

        return () => ctx.revert();
    }, [loading, movies.length]);

    const handleScroll = () => {
        if (scrollRef.current) {
            const { scrollLeft, scrollWidth, clientWidth } = scrollRef.current;
            setShowLeftArrow(scrollLeft > 0);
            setShowRightArrow(scrollLeft < scrollWidth - clientWidth - 10);
        }
    };

    const scroll = (direction: "left" | "right") => {
        if (scrollRef.current) {
            const scrollAmount = 600;
            scrollRef.current.scrollBy({
                left: direction === "left" ? -scrollAmount : scrollAmount,
                behavior: "smooth",
            });
        }
    };

    return (
        <section ref={sectionRef} className="space-y-4 relative group/section">
            {/* Header */}
            <div className="section-header flex items-end justify-between invisible">
                <div className="flex items-center gap-3">
                    {IconComponent && (
                        <div className="p-2 bg-primary/20 rounded-lg">
                            <IconComponent className="w-5 h-5 text-primary" />
                        </div>
                    )}
                    <div>
                        <h2 className="text-2xl font-bold">{title}</h2>
                        {subtitle && <p className="text-muted-foreground text-sm mt-0.5">{subtitle}</p>}
                    </div>
                </div>

                {onViewAll && (
                    <button
                        onClick={onViewAll}
                        className="flex items-center gap-1 text-primary hover:text-primary/80 transition-colors text-sm font-medium"
                    >
                        View All
                        <ChevronRight className="w-4 h-4" />
                    </button>
                )}
            </div>

            {/* Movie Scroll Container */}
            <div className="relative -mx-4 px-4">
                {/* Left Arrow */}
                {showLeftArrow && (
                    <button
                        onClick={() => scroll("left")}
                        className="absolute left-0 top-1/2 -translate-y-1/2 z-10 w-12 h-12 flex items-center justify-center bg-black/80 hover:bg-black hover:scale-110 rounded-full shadow-lg opacity-0 group-hover/section:opacity-100 transition-all ml-2"
                    >
                        <ChevronLeft className="w-6 h-6" />
                    </button>
                )}

                {/* Right Arrow */}
                {showRightArrow && movies.length > 5 && (
                    <button
                        onClick={() => scroll("right")}
                        className="absolute right-0 top-1/2 -translate-y-1/2 z-10 w-12 h-12 flex items-center justify-center bg-black/80 hover:bg-black hover:scale-110 rounded-full shadow-lg opacity-0 group-hover/section:opacity-100 transition-all mr-2"
                    >
                        <ChevronRight className="w-6 h-6" />
                    </button>
                )}

                {/* Scrollable Content */}
                <div ref={scrollRef} onScroll={handleScroll} className="flex gap-4 movie-scroll-container">
                    {loading ? (
                        // Loading skeletons
                        Array.from({ length: 8 }).map((_, i) => <MovieCardSkeleton key={i} />)
                    ) : (
                        <>
                            {movies.map((movie) => (
                                <div key={movie.tmdb_id} className="movie-card-item invisible">
                                    <MovieCard
                                        tmdbId={movie.tmdb_id}
                                        title={movie.title}
                                        posterUrl={movie.poster_url}
                                        year={movie.year}
                                        voteAverage={movie.vote_average}
                                        onClick={onMovieClick}
                                    />
                                </div>
                            ))}
                            {hasMore && onLoadMore && (
                                <div className="flex-shrink-0 w-[180px] flex items-center justify-center">
                                    <Button
                                        variant="outline"
                                        onClick={onLoadMore}
                                        className="h-full min-h-[200px] w-full border-dashed hover:border-primary hover:text-primary hover:scale-105 transition-all"
                                    >
                                        Load More
                                    </Button>
                                </div>
                            )}
                        </>
                    )}
                </div>
            </div>
        </section>
    );
}
