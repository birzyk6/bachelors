"use client";

import { useState, useEffect, useRef } from "react";
import {
    UserPlus,
    Search,
    ChevronLeft,
    ChevronRight,
    Users,
    SlidersHorizontal,
    ArrowUpDown,
    ArrowUp,
    ArrowDown,
} from "lucide-react";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent } from "@/components/ui/card";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuLabel,
    DropdownMenuSeparator,
    DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { ScrollArea } from "@/components/ui/scroll-area";
import gsap from "gsap";

interface UserData {
    id: number;
    movielens_user_id: number | null;
    display_name: string;
    ratings_count: number;
}

interface UserSelectorProps {
    onUserSelect: (user: UserData) => void;
    onStartOnboarding: () => void;
}

const RATING_FILTERS = [
    { label: "All Users", min: 0 },
    { label: "20+ ratings", min: 20 },
    { label: "50+ ratings", min: 50 },
    { label: "100+ ratings", min: 100 },
    { label: "500+ ratings", min: 500 },
    { label: "1000+ ratings", min: 1000 },
];

const SORT_OPTIONS = [
    { label: "Most Ratings", field: "ratings_count", direction: "desc" as const },
    { label: "Fewest Ratings", field: "ratings_count", direction: "asc" as const },
    { label: "Name (A-Z)", field: "display_name", direction: "asc" as const },
    { label: "Name (Z-A)", field: "display_name", direction: "desc" as const },
    { label: "User ID (Low)", field: "id", direction: "asc" as const },
    { label: "User ID (High)", field: "id", direction: "desc" as const },
];

const ITEMS_PER_PAGE = 20;

export default function UserSelector({ onUserSelect, onStartOnboarding }: UserSelectorProps) {
    const [users, setUsers] = useState<UserData[]>([]);
    const [loading, setLoading] = useState(true);
    const [searchQuery, setSearchQuery] = useState("");
    const [page, setPage] = useState(1);
    const [totalUsers, setTotalUsers] = useState(0);
    const [minRatings, setMinRatings] = useState(20);
    const [sortField, setSortField] = useState<string>("ratings_count");
    const [sortDirection, setSortDirection] = useState<"asc" | "desc">("desc");

    const containerRef = useRef<HTMLDivElement>(null);
    const listRef = useRef<HTMLDivElement>(null);

    // GSAP entrance animation
    useEffect(() => {
        if (!containerRef.current) return;

        const ctx = gsap.context(() => {
            gsap.fromTo(
                containerRef.current,
                { opacity: 0, y: 20 },
                { opacity: 1, y: 0, duration: 0.5, ease: "power2.out" }
            );
        });

        return () => ctx.revert();
    }, []);

    // Animate user items when they load
    useEffect(() => {
        if (!loading && users.length > 0 && listRef.current) {
            gsap.set(".user-item", { opacity: 1, x: 0 });
        }
    }, [loading, users]);

    // Load users function - direct API call
    const loadUsers = async (pageNum: number, search: string, minRat: number, sortF: string, sortD: "asc" | "desc") => {
        setLoading(true);
        try {
            const data = await api.getUsers(pageNum, ITEMS_PER_PAGE, search, minRat, sortF, sortD);
            setUsers(data.users || []);
            setTotalUsers(data.total || 0);
        } catch (error) {
            console.error("Failed to load users:", error);
        } finally {
            setLoading(false);
        }
    };

    // Load users when filters change (reset to page 1)
    useEffect(() => {
        setPage(1);
        loadUsers(1, searchQuery, minRatings, sortField, sortDirection);
    }, [searchQuery, minRatings, sortField, sortDirection]);

    const handlePageChange = (newPage: number) => {
        setPage(newPage);
        loadUsers(newPage, searchQuery, minRatings, sortField, sortDirection);
    };

    const totalPages = Math.ceil(totalUsers / ITEMS_PER_PAGE);

    const handleSortChange = (field: string, direction: "asc" | "desc") => {
        setSortField(field);
        setSortDirection(direction);
    };

    const currentSort =
        SORT_OPTIONS.find((s) => s.field === sortField && s.direction === sortDirection) || SORT_OPTIONS[0];

    const handleUserIdSearch = () => {
        const userId = parseInt(searchQuery);
        if (!isNaN(userId) && userId > 0) {
            onUserSelect({
                id: userId,
                movielens_user_id: userId,
                display_name: `User ${userId}`,
                ratings_count: 0,
            });
        }
    };

    const selectedFilter = RATING_FILTERS.find((f) => f.min === minRatings) || RATING_FILTERS[1];

    return (
        <div ref={containerRef} className="max-w-2xl mx-auto opacity-0">
            {/* New user button */}
            <Button
                onClick={onStartOnboarding}
                className="w-full mb-6 h-auto py-5 text-lg font-semibold shadow-lg shadow-primary/20"
            >
                <UserPlus className="w-6 h-6 mr-3" />
                Create New Profile
            </Button>

            {/* Stats */}
            <div className="flex items-center justify-center gap-2 mb-4 text-muted-foreground">
                <Users className="w-4 h-4" />
                <span className="text-sm">{totalUsers.toLocaleString()} users available</span>
            </div>

            {/* Divider */}
            <div className="relative my-6">
                <div className="absolute inset-0 flex items-center">
                    <div className="w-full border-t border-border" />
                </div>
                <div className="relative flex justify-center">
                    <span className="px-4 bg-background text-muted-foreground text-sm">
                        or impersonate an existing user
                    </span>
                </div>
            </div>

            {/* Search and Filter Row */}
            <div className="flex gap-2 mb-4">
                {/* Search */}
                <div className="relative flex-1">
                    <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground pointer-events-none" />
                    <Input
                        type="text"
                        placeholder="Search by name or enter user ID..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        onKeyDown={(e) => {
                            if (e.key === "Enter") handleUserIdSearch();
                        }}
                        className="pl-12 bg-secondary/50"
                    />
                </div>

                {/* Sort dropdown */}
                <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                        <Button variant="outline" className="gap-2">
                            <ArrowUpDown className="w-4 h-4" />
                            <span className="hidden sm:inline">{currentSort.label}</span>
                        </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                        <DropdownMenuLabel>Sort By</DropdownMenuLabel>
                        <DropdownMenuSeparator />
                        {SORT_OPTIONS.map((option) => (
                            <DropdownMenuItem
                                key={`${option.field}-${option.direction}`}
                                onClick={() => handleSortChange(option.field, option.direction)}
                                className={cn(
                                    sortField === option.field &&
                                        sortDirection === option.direction &&
                                        "bg-primary/20 text-primary"
                                )}
                            >
                                {option.direction === "asc" ? (
                                    <ArrowUp className="w-3 h-3 mr-2" />
                                ) : (
                                    <ArrowDown className="w-3 h-3 mr-2" />
                                )}
                                {option.label}
                            </DropdownMenuItem>
                        ))}
                    </DropdownMenuContent>
                </DropdownMenu>

                {/* Filter dropdown */}
                <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                        <Button
                            variant={minRatings > 0 ? "default" : "outline"}
                            className={cn("gap-2", minRatings > 0 && "bg-primary/20 text-primary hover:bg-primary/30")}
                        >
                            <SlidersHorizontal className="w-4 h-4" />
                            <span className="hidden sm:inline">{selectedFilter.label}</span>
                        </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                        <DropdownMenuLabel>Minimum Ratings</DropdownMenuLabel>
                        <DropdownMenuSeparator />
                        {RATING_FILTERS.map((filter) => (
                            <DropdownMenuItem
                                key={filter.min}
                                onClick={() => setMinRatings(filter.min)}
                                className={cn(minRatings === filter.min && "bg-primary/20 text-primary")}
                            >
                                {filter.label}
                            </DropdownMenuItem>
                        ))}
                    </DropdownMenuContent>
                </DropdownMenu>
            </div>

            {/* Quick user ID jump */}
            {searchQuery && !isNaN(parseInt(searchQuery)) && (
                <Button variant="outline" onClick={handleUserIdSearch} className="w-full mb-4 justify-start">
                    <span className="text-primary">Jump to User ID: </span>
                    <span className="font-medium ml-1">{searchQuery}</span>
                </Button>
            )}

            {/* User list */}
            <Card className="bg-card/50">
                <CardContent className="p-0">
                    <div ref={listRef}>
                        <ScrollArea className="h-80">
                            {loading ? (
                                <div className="p-8 text-center">
                                    <div className="animate-spin w-8 h-8 border-2 border-primary border-t-transparent rounded-full mx-auto mb-3" />
                                    <p className="text-muted-foreground">Loading users...</p>
                                </div>
                            ) : users.length === 0 ? (
                                <div className="p-8 text-center text-muted-foreground">
                                    {searchQuery
                                        ? "No users found matching your search"
                                        : "No users match the current filters"}
                                </div>
                            ) : (
                                <div className="divide-y divide-border">
                                    {users.map((user) => (
                                        <button
                                            key={user.id}
                                            onClick={() => onUserSelect(user)}
                                            className="user-item w-full p-4 hover:bg-accent transition-colors flex items-center gap-4 text-left opacity-0"
                                        >
                                            <Avatar className="w-12 h-12">
                                                <AvatarFallback className="bg-primary text-primary-foreground text-lg font-bold">
                                                    {user.display_name.charAt(0).toUpperCase()}
                                                </AvatarFallback>
                                            </Avatar>
                                            <div className="flex-1 min-w-0">
                                                <p className="font-medium truncate">{user.display_name}</p>
                                                <p className="text-sm text-muted-foreground">
                                                    {user.ratings_count.toLocaleString()} ratings
                                                    {user.movielens_user_id && (
                                                        <span className="ml-2 opacity-60">
                                                            ID: {user.movielens_user_id}
                                                        </span>
                                                    )}
                                                </p>
                                            </div>
                                        </button>
                                    ))}
                                </div>
                            )}
                        </ScrollArea>
                    </div>

                    {/* Pagination controls */}
                    {totalPages > 1 && !loading && (
                        <div className="flex items-center justify-between p-3 border-t border-border">
                            <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => handlePageChange(page - 1)}
                                disabled={page <= 1}
                                className="gap-1"
                            >
                                <ChevronLeft className="w-4 h-4" />
                                Previous
                            </Button>

                            <div className="flex items-center gap-1">
                                {/* First page */}
                                {page > 3 && (
                                    <>
                                        <Button
                                            variant="ghost"
                                            size="sm"
                                            onClick={() => handlePageChange(1)}
                                            className="w-8 h-8 p-0"
                                        >
                                            1
                                        </Button>
                                        {page > 4 && <span className="px-1 text-muted-foreground">...</span>}
                                    </>
                                )}

                                {/* Page numbers */}
                                {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                                    let pageNum: number;
                                    if (totalPages <= 5) {
                                        pageNum = i + 1;
                                    } else if (page <= 3) {
                                        pageNum = i + 1;
                                    } else if (page >= totalPages - 2) {
                                        pageNum = totalPages - 4 + i;
                                    } else {
                                        pageNum = page - 2 + i;
                                    }

                                    if (pageNum < 1 || pageNum > totalPages) return null;

                                    return (
                                        <Button
                                            key={pageNum}
                                            variant={page === pageNum ? "default" : "ghost"}
                                            size="sm"
                                            onClick={() => handlePageChange(pageNum)}
                                            className={cn(
                                                "w-8 h-8 p-0",
                                                page === pageNum && "bg-primary text-primary-foreground"
                                            )}
                                        >
                                            {pageNum}
                                        </Button>
                                    );
                                })}

                                {/* Last page */}
                                {page < totalPages - 2 && totalPages > 5 && (
                                    <>
                                        {page < totalPages - 3 && (
                                            <span className="px-1 text-muted-foreground">...</span>
                                        )}
                                        <Button
                                            variant="ghost"
                                            size="sm"
                                            onClick={() => handlePageChange(totalPages)}
                                            className="w-8 h-8 p-0"
                                        >
                                            {totalPages}
                                        </Button>
                                    </>
                                )}
                            </div>

                            <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => handlePageChange(page + 1)}
                                disabled={page >= totalPages}
                                className="gap-1"
                            >
                                Next
                                <ChevronRight className="w-4 h-4" />
                            </Button>
                        </div>
                    )}

                    {/* Page info */}
                    {totalPages > 0 && !loading && (
                        <div className="text-center text-xs text-muted-foreground pb-3">
                            Page {page} of {totalPages} • Showing {users.length} of {totalUsers.toLocaleString()} users
                        </div>
                    )}
                </CardContent>
            </Card>
        </div>
    );
}
