"use client";

import { useState, useEffect, useRef } from "react";
import { Film, User, ChevronDown, Search, X, Star, BarChart3, UserPlus } from "lucide-react";
import Image from "next/image";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuLabel,
    DropdownMenuSeparator,
    DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

interface HeaderUser {
    id: number;
    movielens_user_id: number | null;
    display_name: string;
    ratings_count: number;
}

interface SearchResult {
    tmdb_id: number;
    title: string;
    poster_url: string | null;
    year: number | null;
    vote_average: number;
}

interface HeaderProps {
    currentUser: HeaderUser | null;
    onUserSelect: (user: HeaderUser | null) => void;
    onStartOnboarding: () => void;
    onMovieClick?: (tmdbId: number) => void;
    onOpenSettings?: () => void;
}

export default function Header({
    currentUser,
    onUserSelect,
    onStartOnboarding,
    onMovieClick,
    onOpenSettings,
}: HeaderProps) {
    const [showSearch, setShowSearch] = useState(false);
    const [searchQuery, setSearchQuery] = useState("");
    const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
    const [searching, setSearching] = useState(false);
    const searchRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLInputElement>(null);

    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (searchRef.current && !searchRef.current.contains(event.target as Node)) {
                setShowSearch(false);
                setSearchResults([]);
            }
        };
        document.addEventListener("mousedown", handleClickOutside);
        return () => document.removeEventListener("mousedown", handleClickOutside);
    }, []);

    useEffect(() => {
        if (showSearch && inputRef.current) {
            inputRef.current.focus();
        }
    }, [showSearch]);

    useEffect(() => {
        const timer = setTimeout(async () => {
            if (searchQuery.trim().length >= 2) {
                setSearching(true);
                try {
                    const data = await api.searchMovies(searchQuery);
                    setSearchResults(data.results?.slice(0, 8) || []);
                } catch (error) {
                    console.error("Search failed:", error);
                    setSearchResults([]);
                } finally {
                    setSearching(false);
                }
            } else {
                setSearchResults([]);
            }
        }, 300);

        return () => clearTimeout(timer);
    }, [searchQuery]);

    const handleMovieSelect = (tmdbId: number) => {
        if (onMovieClick) {
            onMovieClick(tmdbId);
        }
        setShowSearch(false);
        setSearchQuery("");
        setSearchResults([]);
    };

    return (
        <header className="bg-background/80 backdrop-blur-md border-b border-border sticky top-0 z-50">
            <div className="container mx-auto px-4 py-3">
                <div className="flex items-center justify-between gap-4">
                    {/* Logo */}
                    <div className="flex items-center gap-3 flex-shrink-0">
                        <div className="p-2 rounded-lg bg-primary/10">
                            <Film className="w-6 h-6 text-primary" />
                        </div>
                        <h1 className="text-xl font-bold hidden sm:block">
                            Movie<span className="text-primary">Recs</span>
                        </h1>
                    </div>

                    {/* Search Bar */}
                    <div ref={searchRef} className="flex-1 max-w-xl relative">
                        <div className="relative">
                            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                            <Input
                                ref={inputRef}
                                type="text"
                                placeholder="Search movies..."
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                                onFocus={() => setShowSearch(true)}
                                className="pl-10 pr-10 bg-secondary/50 border-border"
                            />
                            {searchQuery && (
                                <button
                                    onClick={() => {
                                        setSearchQuery("");
                                        setSearchResults([]);
                                    }}
                                    className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                                >
                                    <X className="w-4 h-4" />
                                </button>
                            )}
                        </div>

                        {/* Search Results Dropdown */}
                        {showSearch && (searchResults.length > 0 || searching || searchQuery.length >= 2) && (
                            <div className="absolute top-full left-0 right-0 mt-2 bg-popover border border-border rounded-xl shadow-2xl overflow-hidden z-50">
                                {searching ? (
                                    <div className="p-4 text-center text-muted-foreground">
                                        <div className="animate-spin w-5 h-5 border-2 border-primary border-t-transparent rounded-full mx-auto mb-2" />
                                        Searching...
                                    </div>
                                ) : searchResults.length === 0 && searchQuery.length >= 2 ? (
                                    <div className="p-4 text-center text-muted-foreground">
                                        No movies found for "{searchQuery}"
                                    </div>
                                ) : (
                                    <div className="max-h-96 overflow-y-auto">
                                        {searchResults.map((movie) => (
                                            <button
                                                key={movie.tmdb_id}
                                                onClick={() => handleMovieSelect(movie.tmdb_id)}
                                                className="w-full p-3 flex items-center gap-3 hover:bg-accent transition-colors text-left"
                                            >
                                                <div className="w-12 h-18 flex-shrink-0 rounded overflow-hidden bg-secondary">
                                                    {movie.poster_url ? (
                                                        <Image
                                                            src={movie.poster_url}
                                                            alt={movie.title}
                                                            width={48}
                                                            height={72}
                                                            className="object-cover w-full h-full"
                                                        />
                                                    ) : (
                                                        <div className="w-full h-full flex items-center justify-center text-2xl">
                                                            🎬
                                                        </div>
                                                    )}
                                                </div>
                                                <div className="flex-1 min-w-0">
                                                    <p className="font-medium truncate">{movie.title}</p>
                                                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                                                        {movie.year && <span>{movie.year}</span>}
                                                        {movie.vote_average > 0 && (
                                                            <span className="flex items-center gap-1">
                                                                <Star className="w-3 h-3 text-yellow-400 fill-yellow-400" />
                                                                {movie.vote_average.toFixed(1)}
                                                            </span>
                                                        )}
                                                    </div>
                                                </div>
                                            </button>
                                        ))}
                                    </div>
                                )}
                            </div>
                        )}
                    </div>

                    {/* User section */}
                    <div className="flex-shrink-0">
                        {currentUser ? (
                            <DropdownMenu>
                                <DropdownMenuTrigger asChild>
                                    <Button variant="ghost" className="flex items-center gap-2 px-3">
                                        <Avatar className="w-8 h-8">
                                            <AvatarFallback className="bg-primary text-primary-foreground text-sm font-bold">
                                                {(currentUser.display_name || "U").charAt(0).toUpperCase()}
                                            </AvatarFallback>
                                        </Avatar>
                                        <span className="max-w-[100px] truncate hidden sm:block">
                                            {currentUser.display_name || "New User"}
                                        </span>
                                        <ChevronDown className="w-4 h-4" />
                                    </Button>
                                </DropdownMenuTrigger>
                                <DropdownMenuContent align="end" className="w-56">
                                    <DropdownMenuLabel>
                                        <p className="text-xs text-muted-foreground font-normal">Impersonating</p>
                                        <p className="truncate">{currentUser.display_name || "New User"}</p>
                                        <p className="text-xs text-muted-foreground font-normal mt-1">
                                            {(currentUser.ratings_count || 0).toLocaleString()} ratings
                                        </p>
                                    </DropdownMenuLabel>
                                    <DropdownMenuSeparator />
                                    <DropdownMenuItem onClick={() => onOpenSettings?.()}>
                                        <BarChart3 className="w-4 h-4 mr-2" />
                                        My Stats & Ratings
                                    </DropdownMenuItem>
                                    <DropdownMenuItem onClick={() => onUserSelect(null)}>
                                        <User className="w-4 h-4 mr-2" />
                                        Switch User
                                    </DropdownMenuItem>
                                    <DropdownMenuItem onClick={() => onStartOnboarding()}>
                                        <UserPlus className="w-4 h-4 mr-2" />
                                        New Profile
                                    </DropdownMenuItem>
                                </DropdownMenuContent>
                            </DropdownMenu>
                        ) : (
                            <Button onClick={onStartOnboarding}>Get Started</Button>
                        )}
                    </div>
                </div>
            </div>
        </header>
    );
}
