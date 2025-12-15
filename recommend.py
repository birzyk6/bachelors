#!/usr/bin/env python3
"""
CLI entry point for Two-Tower movie recommendations.

Usage:
    python recommend.py                      # Interactive mode
    python recommend.py -c "genre action"    # Single command
    python recommend.py -c "similar 260"     # Find similar movies
    python recommend.py -c "search matrix"   # Search by title
"""

from model.src.cli import main

if __name__ == "__main__":
    main()
