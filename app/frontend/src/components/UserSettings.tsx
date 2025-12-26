"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import {
    X,
    Star,
    Film,
    Calendar,
    TrendingUp,
    BarChart3,
    Clock,
    Heart,
    ThumbsDown,
    Meh,
    Award,
    Sparkles,
    Target,
    ChevronDown,
} from "lucide-react";
import Image from "next/image";
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    PieChart,
    Pie,
    Cell,
    LineChart,
    Line,
    Area,
    AreaChart,
    RadarChart,
    Radar,
    PolarGrid,
    PolarAngleAxis,
    PolarRadiusAxis,
} from "recharts";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import gsap from "gsap";

interface Rating {
    tmdb_id: number;
    rating: number;
    created_at: string;
    movie?: {
        title: string;
        poster_url: string | null;
        year: number | null;
        genres?: string[];
    };
}

interface Statistics {
    total_ratings: number;
    average_rating: number;
    rating_distribution: { rating: string; count: number }[];
    genre_distribution: { name: string; count: number; avg_rating: number }[];
    decade_distribution: { decade: string; count: number }[];
    monthly_activity: { month: string; count: number; avg_rating: number }[];
    yearly_activity: { year: number; count: number; avg_rating: number }[];
    insights: {
        favorite_genre: string | null;
        avg_movie_year: number | null;
        most_active_month: string | null;
        rating_style: string;
        liked_count: number;
        disliked_count: number;
        neutral_count: number;
        liked_percentage: number;
        top_rated_genres: { name: string; count: number; avg_rating: number }[];
        lowest_rated_genres: { name: string; count: number; avg_rating: number }[];
    };
}

interface UserSettingsProps {
    userId: number;
    userName: string;
    onClose: () => void;
    onMovieClick: (tmdbId: number) => void;
}

const COLORS = ["#ef4444", "#f97316", "#eab308", "#22c55e", "#14b8a6", "#3b82f6", "#8b5cf6", "#ec4899"];
const RATING_COLORS = ["#ef4444", "#f97316", "#eab308", "#84cc16", "#22c55e"];

export default function UserSettings({ userId, userName, onClose, onMovieClick }: UserSettingsProps) {
    const [statistics, setStatistics] = useState<Statistics | null>(null);
    const [ratings, setRatings] = useState<Rating[]>([]);
    const [loading, setLoading] = useState(true);
    const [loadingRatings, setLoadingRatings] = useState(false);
    const [activeTab, setActiveTab] = useState<"overview" | "insights" | "charts" | "ratings">("overview");
    const [ratingsPage, setRatingsPage] = useState(1);
    const [hasMoreRatings, setHasMoreRatings] = useState(true);

    const backdropRef = useRef<HTMLDivElement>(null);
    const contentRef = useRef<HTMLDivElement>(null);

    // GSAP entrance animation
    useEffect(() => {
        const ctx = gsap.context(() => {
            gsap.fromTo(backdropRef.current, { opacity: 0 }, { opacity: 1, duration: 0.3, ease: "power2.out" });

            gsap.fromTo(
                contentRef.current,
                { opacity: 0, y: 60, scale: 0.95 },
                { opacity: 1, y: 0, scale: 1, duration: 0.5, ease: "power3.out", delay: 0.1 }
            );
        });

        return () => ctx.revert();
    }, []);

    // Animate content when stats load
    useEffect(() => {
        if (!loading && statistics && contentRef.current) {
            const ctx = gsap.context(() => {
                gsap.fromTo(
                    ".stat-card",
                    { opacity: 0, y: 20, scale: 0.95 },
                    { opacity: 1, y: 0, scale: 1, duration: 0.4, stagger: 0.08, ease: "power2.out" }
                );

                gsap.fromTo(
                    ".insight-card",
                    { opacity: 0, x: -20 },
                    { opacity: 1, x: 0, duration: 0.4, stagger: 0.1, ease: "power2.out", delay: 0.3 }
                );

                gsap.fromTo(
                    ".chart-section",
                    { opacity: 0, y: 30 },
                    { opacity: 1, y: 0, duration: 0.5, stagger: 0.15, ease: "power2.out", delay: 0.2 }
                );
            }, contentRef);

            return () => ctx.revert();
        }
    }, [loading, statistics, activeTab]);

    const loadStatistics = useCallback(async () => {
        setLoading(true);
        try {
            const data = await api.getUserStatistics(userId);
            setStatistics(data);
        } catch (error) {
            console.error("Failed to load statistics:", error);
        } finally {
            setLoading(false);
        }
    }, [userId]);

    const loadRatings = useCallback(
        async (page: number, append = false) => {
            setLoadingRatings(true);
            try {
                const data = await api.getUserRatings(userId, page, 30, true);
                if (append) {
                    setRatings((prev) => [...prev, ...(data.ratings || [])]);
                } else {
                    setRatings(data.ratings || []);
                }
                setHasMoreRatings(data.has_more || false);
            } catch (error) {
                console.error("Failed to load ratings:", error);
            } finally {
                setLoadingRatings(false);
            }
        },
        [userId]
    );

    useEffect(() => {
        loadStatistics();
    }, [loadStatistics]);

    useEffect(() => {
        if (activeTab === "ratings" && ratings.length === 0) {
            loadRatings(1);
        }
    }, [activeTab, ratings.length, loadRatings]);

    const handleLoadMoreRatings = () => {
        const nextPage = ratingsPage + 1;
        setRatingsPage(nextPage);
        loadRatings(nextPage, true);
    };

    // Prepare radar chart data for genre preferences
    const radarData =
        statistics?.genre_distribution.slice(0, 6).map((g) => ({
            genre: g.name,
            count: g.count,
            rating: g.avg_rating,
            fullMark: 5,
        })) || [];

    return (
        <div className="fixed inset-0 z-50 overflow-y-auto">
            <div ref={backdropRef} className="fixed inset-0 bg-black/85 modal-backdrop opacity-0" onClick={onClose} />
            <div className="relative min-h-screen flex items-start justify-center p-4 pt-8">
                <div
                    ref={contentRef}
                    className="relative w-full max-w-6xl bg-gradient-to-b from-background to-background/95 border border-border rounded-2xl overflow-hidden shadow-2xl opacity-0"
                >
                    {/* Header with gradient */}
                    <div className="relative overflow-hidden">
                        <div className="absolute inset-0 bg-gradient-to-r from-primary/20 via-primary/10 to-transparent" />
                        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-primary/20 via-transparent to-transparent" />

                        <div className="relative flex items-center justify-between p-6">
                            <div className="flex items-center gap-4">
                                <div className="relative">
                                    <div className="w-16 h-16 rounded-full bg-gradient-to-br from-primary to-primary/50 flex items-center justify-center ring-4 ring-primary/20">
                                        <span className="text-2xl font-bold text-primary-foreground">
                                            {userName.charAt(0).toUpperCase()}
                                        </span>
                                    </div>
                                    <div className="absolute -bottom-1 -right-1 w-6 h-6 rounded-full bg-background flex items-center justify-center">
                                        <Award className="w-4 h-4 text-yellow-500" />
                                    </div>
                                </div>
                                <div>
                                    <h2 className="text-2xl font-bold">{userName}</h2>
                                    <p className="text-muted-foreground flex items-center gap-2">
                                        <Film className="w-4 h-4" />
                                        {statistics?.total_ratings.toLocaleString() || 0} movies rated
                                        {statistics?.insights.rating_style && (
                                            <span className="px-2 py-0.5 rounded-full bg-primary/20 text-primary text-xs font-medium ml-2">
                                                {statistics.insights.rating_style}
                                            </span>
                                        )}
                                    </p>
                                </div>
                            </div>
                            <button onClick={onClose} className="p-2 hover:bg-secondary rounded-lg transition-colors">
                                <X className="w-5 h-5" />
                            </button>
                        </div>
                    </div>

                    {/* Tabs */}
                    <div className="flex border-b border-border bg-secondary/20">
                        {(["overview", "insights", "charts", "ratings"] as const).map((tab) => (
                            <button
                                key={tab}
                                onClick={() => setActiveTab(tab)}
                                className={cn(
                                    "flex-1 px-6 py-3.5 text-sm font-medium transition-all relative",
                                    activeTab === tab ? "text-primary" : "text-muted-foreground hover:text-foreground"
                                )}
                            >
                                {tab.charAt(0).toUpperCase() + tab.slice(1)}
                                {activeTab === tab && (
                                    <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r from-transparent via-primary to-transparent" />
                                )}
                            </button>
                        ))}
                    </div>

                    {/* Content */}
                    <div className="p-6 max-h-[70vh] overflow-y-auto">
                        {loading ? (
                            <div className="flex flex-col items-center justify-center py-20 gap-4">
                                <div className="relative">
                                    <div className="animate-spin w-12 h-12 border-4 border-primary/20 border-t-primary rounded-full" />
                                    <Sparkles className="absolute inset-0 m-auto w-5 h-5 text-primary animate-pulse" />
                                </div>
                                <p className="text-muted-foreground">Analyzing your movie taste...</p>
                            </div>
                        ) : statistics ? (
                            <>
                                {/* Overview Tab */}
                                {activeTab === "overview" && (
                                    <div className="space-y-6">
                                        {/* Hero Stats */}
                                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                            <div className="stat-card group relative overflow-hidden bg-gradient-to-br from-primary/20 to-primary/5 rounded-2xl p-5 border border-primary/20 opacity-0">
                                                <div className="absolute top-0 right-0 w-20 h-20 bg-primary/10 rounded-full -translate-y-1/2 translate-x-1/2" />
                                                <Film className="w-5 h-5 text-primary mb-2" />
                                                <p className="text-3xl font-bold">
                                                    {statistics.total_ratings.toLocaleString()}
                                                </p>
                                                <p className="text-sm text-muted-foreground">Movies Rated</p>
                                            </div>

                                            <div className="stat-card group relative overflow-hidden bg-gradient-to-br from-yellow-500/20 to-yellow-500/5 rounded-2xl p-5 border border-yellow-500/20 opacity-0">
                                                <div className="absolute top-0 right-0 w-20 h-20 bg-yellow-500/10 rounded-full -translate-y-1/2 translate-x-1/2" />
                                                <Star className="w-5 h-5 text-yellow-500 mb-2" />
                                                <p className="text-3xl font-bold">{statistics.average_rating}</p>
                                                <p className="text-sm text-muted-foreground">Avg Rating</p>
                                            </div>

                                            <div className="stat-card group relative overflow-hidden bg-gradient-to-br from-green-500/20 to-green-500/5 rounded-2xl p-5 border border-green-500/20 opacity-0">
                                                <div className="absolute top-0 right-0 w-20 h-20 bg-green-500/10 rounded-full -translate-y-1/2 translate-x-1/2" />
                                                <Heart className="w-5 h-5 text-green-500 mb-2" />
                                                <p className="text-3xl font-bold">
                                                    {statistics.insights.liked_count.toLocaleString()}
                                                </p>
                                                <p className="text-sm text-muted-foreground">Loved (4-5★)</p>
                                            </div>

                                            <div className="stat-card group relative overflow-hidden bg-gradient-to-br from-red-500/20 to-red-500/5 rounded-2xl p-5 border border-red-500/20 opacity-0">
                                                <div className="absolute top-0 right-0 w-20 h-20 bg-red-500/10 rounded-full -translate-y-1/2 translate-x-1/2" />
                                                <ThumbsDown className="w-5 h-5 text-red-500 mb-2" />
                                                <p className="text-3xl font-bold">
                                                    {statistics.insights.disliked_count.toLocaleString()}
                                                </p>
                                                <p className="text-sm text-muted-foreground">Disliked (1-2★)</p>
                                            </div>
                                        </div>

                                        {/* Rating Distribution & Favorite Genre */}
                                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                                            {/* Rating Distribution */}
                                            <div className="chart-section bg-secondary/30 rounded-2xl p-6 border border-border opacity-0">
                                                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                                                    <BarChart3 className="w-5 h-5 text-primary" />
                                                    Rating Distribution
                                                </h3>
                                                <div className="h-52">
                                                    <ResponsiveContainer width="100%" height="100%">
                                                        <BarChart data={statistics.rating_distribution} barSize={24}>
                                                            <CartesianGrid
                                                                strokeDasharray="3 3"
                                                                stroke="#333"
                                                                vertical={false}
                                                            />
                                                            <XAxis
                                                                dataKey="rating"
                                                                stroke="#888"
                                                                tick={{ fontSize: 11 }}
                                                                tickFormatter={(v) => `${v}`}
                                                            />
                                                            <YAxis stroke="#888" />
                                                            <Tooltip
                                                                contentStyle={{
                                                                    backgroundColor: "#1a1a1a",
                                                                    border: "1px solid #333",
                                                                    borderRadius: "12px",
                                                                }}
                                                                labelFormatter={(v) => `${v} Stars`}
                                                                formatter={(value: number) => [
                                                                    value.toLocaleString(),
                                                                    "Movies",
                                                                ]}
                                                            />
                                                            <Bar
                                                                dataKey="count"
                                                                radius={[8, 8, 0, 0]}
                                                                fill="url(#ratingGradient)"
                                                            />
                                                            <defs>
                                                                <linearGradient
                                                                    id="ratingGradient"
                                                                    x1="0"
                                                                    y1="0"
                                                                    x2="0"
                                                                    y2="1"
                                                                >
                                                                    <stop
                                                                        offset="0%"
                                                                        stopColor="#ef4444"
                                                                        stopOpacity={1}
                                                                    />
                                                                    <stop
                                                                        offset="100%"
                                                                        stopColor="#ef4444"
                                                                        stopOpacity={0.5}
                                                                    />
                                                                </linearGradient>
                                                            </defs>
                                                        </BarChart>
                                                    </ResponsiveContainer>
                                                </div>
                                            </div>

                                            {/* Quick Insights */}
                                            <div className="chart-section bg-secondary/30 rounded-2xl p-6 border border-border opacity-0">
                                                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                                                    <Sparkles className="w-5 h-5 text-primary" />
                                                    Your Movie Profile
                                                </h3>
                                                <div className="space-y-4">
                                                    {statistics.insights.favorite_genre && (
                                                        <div className="flex items-center justify-between p-3 bg-primary/10 rounded-xl">
                                                            <div className="flex items-center gap-3">
                                                                <div className="w-10 h-10 rounded-full bg-primary/20 flex items-center justify-center">
                                                                    <Heart className="w-5 h-5 text-primary" />
                                                                </div>
                                                                <div>
                                                                    <p className="text-sm text-muted-foreground">
                                                                        Favorite Genre
                                                                    </p>
                                                                    <p className="font-semibold">
                                                                        {statistics.insights.favorite_genre}
                                                                    </p>
                                                                </div>
                                                            </div>
                                                        </div>
                                                    )}

                                                    {statistics.insights.avg_movie_year && (
                                                        <div className="flex items-center justify-between p-3 bg-secondary/50 rounded-xl">
                                                            <div className="flex items-center gap-3">
                                                                <div className="w-10 h-10 rounded-full bg-secondary flex items-center justify-center">
                                                                    <Calendar className="w-5 h-5 text-muted-foreground" />
                                                                </div>
                                                                <div>
                                                                    <p className="text-sm text-muted-foreground">
                                                                        Average Movie Era
                                                                    </p>
                                                                    <p className="font-semibold">
                                                                        {statistics.insights.avg_movie_year}
                                                                    </p>
                                                                </div>
                                                            </div>
                                                        </div>
                                                    )}

                                                    <div className="flex items-center justify-between p-3 bg-secondary/50 rounded-xl">
                                                        <div className="flex items-center gap-3">
                                                            <div className="w-10 h-10 rounded-full bg-secondary flex items-center justify-center">
                                                                <Target className="w-5 h-5 text-muted-foreground" />
                                                            </div>
                                                            <div>
                                                                <p className="text-sm text-muted-foreground">
                                                                    Approval Rate
                                                                </p>
                                                                <p className="font-semibold">
                                                                    {statistics.insights.liked_percentage}% liked
                                                                </p>
                                                            </div>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>

                                        {/* Genre Distribution */}
                                        <div className="chart-section bg-secondary/30 rounded-2xl p-6 border border-border opacity-0">
                                            <h3 className="text-lg font-semibold mb-4">Top Genres</h3>
                                            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                                                {statistics.genre_distribution.slice(0, 8).map((genre, i) => (
                                                    <div
                                                        key={genre.name}
                                                        className="relative overflow-hidden rounded-xl p-4 border border-border bg-gradient-to-br from-secondary/50 to-secondary/20"
                                                    >
                                                        <div
                                                            className="absolute inset-0 opacity-20"
                                                            style={{ backgroundColor: COLORS[i % COLORS.length] }}
                                                        />
                                                        <p className="font-medium relative z-10">{genre.name}</p>
                                                        <p className="text-2xl font-bold relative z-10">
                                                            {genre.count}
                                                        </p>
                                                        <div className="flex items-center gap-1 text-sm text-muted-foreground relative z-10">
                                                            <Star className="w-3 h-3 text-yellow-400 fill-yellow-400" />
                                                            {genre.avg_rating}
                                                        </div>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    </div>
                                )}

                                {/* Insights Tab */}
                                {activeTab === "insights" && (
                                    <div className="space-y-6">
                                        {/* Rating Style Card */}
                                        <div className="insight-card relative overflow-hidden bg-gradient-to-r from-primary/20 via-primary/10 to-transparent rounded-2xl p-6 border border-primary/20 opacity-0">
                                            <div className="flex items-center gap-4">
                                                <div className="w-16 h-16 rounded-2xl bg-primary/20 flex items-center justify-center">
                                                    <Award className="w-8 h-8 text-primary" />
                                                </div>
                                                <div>
                                                    <p className="text-sm text-muted-foreground">Your Rating Style</p>
                                                    <p className="text-2xl font-bold">
                                                        {statistics.insights.rating_style}
                                                    </p>
                                                    <p className="text-sm text-muted-foreground mt-1">
                                                        Based on your average rating of {statistics.average_rating}{" "}
                                                        stars
                                                    </p>
                                                </div>
                                            </div>
                                        </div>

                                        {/* Sentiment Breakdown */}
                                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                            <div className="insight-card bg-green-500/10 rounded-2xl p-5 border border-green-500/20 opacity-0">
                                                <div className="flex items-center gap-3 mb-3">
                                                    <Heart className="w-6 h-6 text-green-500" />
                                                    <span className="font-medium">Loved It!</span>
                                                </div>
                                                <p className="text-3xl font-bold text-green-500">
                                                    {statistics.insights.liked_count.toLocaleString()}
                                                </p>
                                                <p className="text-sm text-muted-foreground">Movies rated 4-5 stars</p>
                                                <div className="mt-3 h-2 bg-secondary rounded-full overflow-hidden">
                                                    <div
                                                        className="h-full bg-gradient-to-r from-green-500 to-green-400 rounded-full"
                                                        style={{
                                                            width: `${(statistics.insights.liked_count / statistics.total_ratings) * 100}%`,
                                                        }}
                                                    />
                                                </div>
                                            </div>

                                            <div className="insight-card bg-yellow-500/10 rounded-2xl p-5 border border-yellow-500/20 opacity-0">
                                                <div className="flex items-center gap-3 mb-3">
                                                    <Meh className="w-6 h-6 text-yellow-500" />
                                                    <span className="font-medium">It was OK</span>
                                                </div>
                                                <p className="text-3xl font-bold text-yellow-500">
                                                    {statistics.insights.neutral_count.toLocaleString()}
                                                </p>
                                                <p className="text-sm text-muted-foreground">
                                                    Movies rated 2.5-3.5 stars
                                                </p>
                                                <div className="mt-3 h-2 bg-secondary rounded-full overflow-hidden">
                                                    <div
                                                        className="h-full bg-gradient-to-r from-yellow-500 to-yellow-400 rounded-full"
                                                        style={{
                                                            width: `${(statistics.insights.neutral_count / statistics.total_ratings) * 100}%`,
                                                        }}
                                                    />
                                                </div>
                                            </div>

                                            <div className="insight-card bg-red-500/10 rounded-2xl p-5 border border-red-500/20 opacity-0">
                                                <div className="flex items-center gap-3 mb-3">
                                                    <ThumbsDown className="w-6 h-6 text-red-500" />
                                                    <span className="font-medium">Not a Fan</span>
                                                </div>
                                                <p className="text-3xl font-bold text-red-500">
                                                    {statistics.insights.disliked_count.toLocaleString()}
                                                </p>
                                                <p className="text-sm text-muted-foreground">Movies rated 1-2 stars</p>
                                                <div className="mt-3 h-2 bg-secondary rounded-full overflow-hidden">
                                                    <div
                                                        className="h-full bg-gradient-to-r from-red-500 to-red-400 rounded-full"
                                                        style={{
                                                            width: `${(statistics.insights.disliked_count / statistics.total_ratings) * 100}%`,
                                                        }}
                                                    />
                                                </div>
                                            </div>
                                        </div>

                                        {/* Genre Preferences */}
                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                            {/* Top Rated Genres */}
                                            <div className="insight-card bg-secondary/30 rounded-2xl p-6 border border-border opacity-0">
                                                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                                                    <TrendingUp className="w-5 h-5 text-green-500" />
                                                    Your Top-Rated Genres
                                                </h3>
                                                <div className="space-y-3">
                                                    {statistics.insights.top_rated_genres.map((genre, i) => (
                                                        <div key={genre.name} className="flex items-center gap-3">
                                                            <span className="w-6 h-6 rounded-full bg-green-500/20 flex items-center justify-center text-sm font-medium text-green-500">
                                                                {i + 1}
                                                            </span>
                                                            <div className="flex-1">
                                                                <div className="flex items-center justify-between">
                                                                    <span className="font-medium">{genre.name}</span>
                                                                    <span className="flex items-center gap-1 text-sm">
                                                                        <Star className="w-3 h-3 text-yellow-400 fill-yellow-400" />
                                                                        {genre.avg_rating}
                                                                    </span>
                                                                </div>
                                                                <p className="text-xs text-muted-foreground">
                                                                    {genre.count} movies
                                                                </p>
                                                            </div>
                                                        </div>
                                                    ))}
                                                </div>
                                            </div>

                                            {/* Lowest Rated Genres */}
                                            <div className="insight-card bg-secondary/30 rounded-2xl p-6 border border-border opacity-0">
                                                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                                                    <TrendingUp className="w-5 h-5 text-red-500 rotate-180" />
                                                    Genres You're Tough On
                                                </h3>
                                                <div className="space-y-3">
                                                    {statistics.insights.lowest_rated_genres.map((genre, i) => (
                                                        <div key={genre.name} className="flex items-center gap-3">
                                                            <span className="w-6 h-6 rounded-full bg-red-500/20 flex items-center justify-center text-sm font-medium text-red-500">
                                                                {i + 1}
                                                            </span>
                                                            <div className="flex-1">
                                                                <div className="flex items-center justify-between">
                                                                    <span className="font-medium">{genre.name}</span>
                                                                    <span className="flex items-center gap-1 text-sm">
                                                                        <Star className="w-3 h-3 text-yellow-400 fill-yellow-400" />
                                                                        {genre.avg_rating}
                                                                    </span>
                                                                </div>
                                                                <p className="text-xs text-muted-foreground">
                                                                    {genre.count} movies
                                                                </p>
                                                            </div>
                                                        </div>
                                                    ))}
                                                </div>
                                            </div>
                                        </div>

                                        {/* Genre Radar */}
                                        {radarData.length >= 3 && (
                                            <div className="insight-card bg-secondary/30 rounded-2xl p-6 border border-border opacity-0">
                                                <h3 className="text-lg font-semibold mb-4">Genre Taste Profile</h3>
                                                <div className="h-72">
                                                    <ResponsiveContainer width="100%" height="100%">
                                                        <RadarChart data={radarData}>
                                                            <PolarGrid stroke="#333" />
                                                            <PolarAngleAxis
                                                                dataKey="genre"
                                                                tick={{ fill: "#888", fontSize: 12 }}
                                                            />
                                                            <PolarRadiusAxis
                                                                angle={30}
                                                                domain={[0, 5]}
                                                                tick={{ fill: "#666" }}
                                                            />
                                                            <Radar
                                                                name="Avg Rating"
                                                                dataKey="rating"
                                                                stroke="#ef4444"
                                                                fill="#ef4444"
                                                                fillOpacity={0.3}
                                                            />
                                                            <Tooltip
                                                                contentStyle={{
                                                                    backgroundColor: "#1a1a1a",
                                                                    border: "1px solid #333",
                                                                    borderRadius: "12px",
                                                                }}
                                                            />
                                                        </RadarChart>
                                                    </ResponsiveContainer>
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                )}

                                {/* Charts Tab */}
                                {activeTab === "charts" && (
                                    <div className="space-y-6">
                                        {/* Activity Timeline */}
                                        {statistics.monthly_activity.length > 0 && (
                                            <div className="chart-section bg-secondary/30 rounded-2xl p-6 border border-border opacity-0">
                                                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                                                    <Clock className="w-5 h-5 text-primary" />
                                                    Rating Activity (Last 12 Months)
                                                </h3>
                                                <div className="h-64">
                                                    <ResponsiveContainer width="100%" height="100%">
                                                        <AreaChart data={statistics.monthly_activity}>
                                                            <defs>
                                                                <linearGradient
                                                                    id="activityGradient"
                                                                    x1="0"
                                                                    y1="0"
                                                                    x2="0"
                                                                    y2="1"
                                                                >
                                                                    <stop
                                                                        offset="0%"
                                                                        stopColor="#ef4444"
                                                                        stopOpacity={0.5}
                                                                    />
                                                                    <stop
                                                                        offset="100%"
                                                                        stopColor="#ef4444"
                                                                        stopOpacity={0}
                                                                    />
                                                                </linearGradient>
                                                            </defs>
                                                            <CartesianGrid
                                                                strokeDasharray="3 3"
                                                                stroke="#333"
                                                                vertical={false}
                                                            />
                                                            <XAxis dataKey="month" stroke="#888" />
                                                            <YAxis stroke="#888" />
                                                            <Tooltip
                                                                contentStyle={{
                                                                    backgroundColor: "#1a1a1a",
                                                                    border: "1px solid #333",
                                                                    borderRadius: "12px",
                                                                }}
                                                                formatter={(value: number, name: string) => [
                                                                    name === "count" ? value.toLocaleString() : value,
                                                                    name === "count" ? "Movies Rated" : "Avg Rating",
                                                                ]}
                                                            />
                                                            <Area
                                                                type="monotone"
                                                                dataKey="count"
                                                                stroke="#ef4444"
                                                                strokeWidth={2}
                                                                fill="url(#activityGradient)"
                                                            />
                                                        </AreaChart>
                                                    </ResponsiveContainer>
                                                </div>
                                            </div>
                                        )}

                                        {/* Decade Distribution */}
                                        {statistics.decade_distribution.length > 0 && (
                                            <div className="chart-section bg-secondary/30 rounded-2xl p-6 border border-border opacity-0">
                                                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                                                    <Calendar className="w-5 h-5 text-primary" />
                                                    Movies by Decade
                                                </h3>
                                                <div className="h-64">
                                                    <ResponsiveContainer width="100%" height="100%">
                                                        <BarChart data={statistics.decade_distribution} barSize={50}>
                                                            <CartesianGrid
                                                                strokeDasharray="3 3"
                                                                stroke="#333"
                                                                vertical={false}
                                                            />
                                                            <XAxis dataKey="decade" stroke="#888" />
                                                            <YAxis stroke="#888" />
                                                            <Tooltip
                                                                contentStyle={{
                                                                    backgroundColor: "#1a1a1a",
                                                                    border: "1px solid #333",
                                                                    borderRadius: "12px",
                                                                }}
                                                                formatter={(value: number) => [
                                                                    value.toLocaleString(),
                                                                    "Movies",
                                                                ]}
                                                            />
                                                            <Bar dataKey="count" radius={[8, 8, 0, 0]}>
                                                                {statistics.decade_distribution.map((_, index) => (
                                                                    <Cell
                                                                        key={`cell-${index}`}
                                                                        fill={COLORS[index % COLORS.length]}
                                                                    />
                                                                ))}
                                                            </Bar>
                                                        </BarChart>
                                                    </ResponsiveContainer>
                                                </div>
                                            </div>
                                        )}

                                        {/* Genre Pie Chart */}
                                        {statistics.genre_distribution.length > 0 && (
                                            <div className="chart-section bg-secondary/30 rounded-2xl p-6 border border-border opacity-0">
                                                <h3 className="text-lg font-semibold mb-4">Genre Distribution</h3>
                                                <div className="h-80">
                                                    <ResponsiveContainer width="100%" height="100%">
                                                        <PieChart>
                                                            <Pie
                                                                data={statistics.genre_distribution.slice(0, 8)}
                                                                cx="50%"
                                                                cy="50%"
                                                                outerRadius={100}
                                                                innerRadius={60}
                                                                paddingAngle={2}
                                                                dataKey="count"
                                                                label={({ name, percent }) =>
                                                                    `${name} ${(percent * 100).toFixed(0)}%`
                                                                }
                                                            >
                                                                {statistics.genre_distribution
                                                                    .slice(0, 8)
                                                                    .map((_, index) => (
                                                                        <Cell
                                                                            key={`cell-${index}`}
                                                                            fill={COLORS[index % COLORS.length]}
                                                                        />
                                                                    ))}
                                                            </Pie>
                                                            <Tooltip
                                                                contentStyle={{
                                                                    backgroundColor: "#1a1a1a",
                                                                    border: "1px solid #333",
                                                                    borderRadius: "12px",
                                                                }}
                                                                formatter={(
                                                                    value: number,
                                                                    name: string,
                                                                    props: any
                                                                ) => [
                                                                    `${value.toLocaleString()} movies (★${props.payload.avg_rating})`,
                                                                    props.payload.name,
                                                                ]}
                                                            />
                                                        </PieChart>
                                                    </ResponsiveContainer>
                                                </div>
                                            </div>
                                        )}

                                        {/* Yearly Activity */}
                                        {statistics.yearly_activity.length > 0 && (
                                            <div className="chart-section bg-secondary/30 rounded-2xl p-6 border border-border opacity-0">
                                                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                                                    <TrendingUp className="w-5 h-5 text-primary" />
                                                    Yearly Rating Activity
                                                </h3>
                                                <div className="h-64">
                                                    <ResponsiveContainer width="100%" height="100%">
                                                        <LineChart data={statistics.yearly_activity}>
                                                            <CartesianGrid
                                                                strokeDasharray="3 3"
                                                                stroke="#333"
                                                                vertical={false}
                                                            />
                                                            <XAxis dataKey="year" stroke="#888" />
                                                            <YAxis stroke="#888" />
                                                            <Tooltip
                                                                contentStyle={{
                                                                    backgroundColor: "#1a1a1a",
                                                                    border: "1px solid #333",
                                                                    borderRadius: "12px",
                                                                }}
                                                                formatter={(value: number, name: string) => [
                                                                    name === "count" ? value.toLocaleString() : value,
                                                                    name === "count" ? "Movies Rated" : "Avg Rating",
                                                                ]}
                                                            />
                                                            <Line
                                                                type="monotone"
                                                                dataKey="count"
                                                                stroke="#ef4444"
                                                                strokeWidth={3}
                                                                dot={{ fill: "#ef4444", strokeWidth: 2 }}
                                                            />
                                                        </LineChart>
                                                    </ResponsiveContainer>
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                )}

                                {/* Ratings List Tab */}
                                {activeTab === "ratings" && (
                                    <div className="space-y-4">
                                        <p className="text-sm text-muted-foreground">
                                            Showing {ratings.length} of {statistics.total_ratings.toLocaleString()}{" "}
                                            ratings
                                        </p>

                                        {loadingRatings && ratings.length === 0 ? (
                                            <div className="flex items-center justify-center py-12">
                                                <div className="animate-spin w-8 h-8 border-2 border-primary border-t-transparent rounded-full" />
                                            </div>
                                        ) : ratings.length === 0 ? (
                                            <p className="text-center text-muted-foreground py-10">No ratings yet</p>
                                        ) : (
                                            <>
                                                <div className="grid gap-2">
                                                    {ratings.map((rating) => (
                                                        <button
                                                            key={rating.tmdb_id}
                                                            onClick={() => {
                                                                onMovieClick(rating.tmdb_id);
                                                                onClose();
                                                            }}
                                                            className="flex items-center gap-4 p-3 rounded-xl hover:bg-secondary/50 transition-colors text-left group"
                                                        >
                                                            <div className="w-12 h-18 flex-shrink-0 rounded-lg overflow-hidden bg-secondary">
                                                                {rating.movie?.poster_url ? (
                                                                    <Image
                                                                        src={rating.movie.poster_url}
                                                                        alt={rating.movie?.title || ""}
                                                                        width={48}
                                                                        height={72}
                                                                        className="object-cover w-full h-full group-hover:scale-105 transition-transform"
                                                                    />
                                                                ) : (
                                                                    <div className="w-full h-full flex items-center justify-center text-2xl">
                                                                        🎬
                                                                    </div>
                                                                )}
                                                            </div>
                                                            <div className="flex-1 min-w-0">
                                                                <p className="font-medium truncate group-hover:text-primary transition-colors">
                                                                    {rating.movie?.title || `Movie #${rating.tmdb_id}`}
                                                                </p>
                                                                <div className="flex items-center gap-3 text-sm text-muted-foreground">
                                                                    {rating.movie?.year && (
                                                                        <span>{rating.movie.year}</span>
                                                                    )}
                                                                    <span>
                                                                        {new Date(
                                                                            rating.created_at
                                                                        ).toLocaleDateString()}
                                                                    </span>
                                                                </div>
                                                            </div>
                                                            <div className="flex items-center gap-1.5 px-3 py-1.5 bg-secondary rounded-full">
                                                                <Star className="w-4 h-4 text-yellow-400 fill-yellow-400" />
                                                                <span className="font-medium text-sm">
                                                                    {rating.rating}
                                                                </span>
                                                            </div>
                                                        </button>
                                                    ))}
                                                </div>

                                                {hasMoreRatings && (
                                                    <Button
                                                        variant="outline"
                                                        onClick={handleLoadMoreRatings}
                                                        disabled={loadingRatings}
                                                        className="w-full mt-4"
                                                    >
                                                        {loadingRatings ? (
                                                            <>
                                                                <div className="animate-spin w-4 h-4 border-2 border-primary border-t-transparent rounded-full mr-2" />
                                                                Loading...
                                                            </>
                                                        ) : (
                                                            <>
                                                                <ChevronDown className="w-4 h-4 mr-2" />
                                                                Load More Ratings
                                                            </>
                                                        )}
                                                    </Button>
                                                )}
                                            </>
                                        )}
                                    </div>
                                )}
                            </>
                        ) : (
                            <div className="text-center py-20 text-muted-foreground">Failed to load statistics</div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
