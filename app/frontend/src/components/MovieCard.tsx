"use client";

import { useRef } from "react";
import Image from "next/image";
import { Star, Play } from "lucide-react";
import { cn } from "@/lib/utils";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface MovieCardProps {
    tmdbId: number;
    title: string;
    posterUrl: string | null;
    year: number | null;
    voteAverage: number;
    onClick: (tmdbId: number) => void;
}

export default function MovieCard({ tmdbId, title, posterUrl, year, voteAverage, onClick }: MovieCardProps) {
    const cardRef = useRef<HTMLDivElement>(null);

    return (
        <Card
            ref={cardRef}
            onClick={() => onClick(tmdbId)}
            className="movie-card flex-shrink-0 w-[180px] cursor-pointer group bg-transparent border-0 shadow-none"
        >
            {/* Poster */}
            <div className="relative aspect-[2/3] rounded-xl overflow-hidden bg-secondary shadow-lg group-hover:shadow-2xl group-hover:shadow-primary/10 transition-shadow duration-300">
                {posterUrl ? (
                    <Image
                        src={posterUrl}
                        alt={title}
                        fill
                        className="object-cover transition-transform duration-500 group-hover:scale-110"
                        sizes="180px"
                    />
                ) : (
                    <div className="absolute inset-0 flex items-center justify-center text-muted-foreground">
                        <span className="text-4xl">🎬</span>
                    </div>
                )}

                {/* Gradient overlay on hover */}
                <div className="absolute inset-0 bg-gradient-to-t from-black/90 via-black/20 to-transparent opacity-0 group-hover:opacity-100 transition-all duration-300" />

                {/* Play button on hover */}
                <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-all duration-300">
                    <div className="w-14 h-14 rounded-full bg-primary/90 flex items-center justify-center transform scale-50 group-hover:scale-100 transition-transform duration-300 shadow-lg">
                        <Play className="w-6 h-6 text-primary-foreground fill-current ml-1" />
                    </div>
                </div>

                {/* Bottom info on hover */}
                <div className="absolute bottom-0 left-0 right-0 p-3 transform translate-y-full group-hover:translate-y-0 transition-transform duration-300">
                    <p className="text-white text-sm font-medium line-clamp-2">{title}</p>
                    {year && <p className="text-white/70 text-xs mt-1">{year}</p>}
                </div>

                {/* Rating badge */}
                {voteAverage > 0 && (
                    <Badge className="absolute top-2 right-2 bg-black/70 hover:bg-black/70 gap-1 text-white backdrop-blur-sm">
                        <Star className="w-3 h-3 text-yellow-400 fill-yellow-400" />
                        {voteAverage.toFixed(1)}
                    </Badge>
                )}

                {/* Glow effect */}
                <div className="absolute inset-0 rounded-xl ring-2 ring-primary/0 group-hover:ring-primary/50 transition-all duration-300" />
            </div>

            {/* Info - visible when not hovering */}
            <div className="mt-3 group-hover:opacity-0 transition-opacity duration-200">
                <h3 className="font-medium text-sm line-clamp-2 group-hover:text-primary transition-colors">{title}</h3>
                {year && <p className="text-xs text-muted-foreground mt-1">{year}</p>}
            </div>
        </Card>
    );
}

// Skeleton loader for movie card
export function MovieCardSkeleton() {
    return (
        <div className="flex-shrink-0 w-[180px]">
            <div className="aspect-[2/3] rounded-lg skeleton" />
            <div className="mt-2 space-y-2">
                <div className="h-4 skeleton rounded w-3/4" />
                <div className="h-3 skeleton rounded w-1/4" />
            </div>
        </div>
    );
}
