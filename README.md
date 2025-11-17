# Bachelor's Thesis - Movie Recommendation System

👉🤓👈

## Overview

Movie recommendation system using collaborative filtering, content-based filtering, and two-tower models.

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management.

### Install uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Install Dependencies

```bash
# Install all dependencies
uv sync

# Install with dev dependencies
uv sync --group dev
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
uv run pre-commit install

# Run hooks manually on all files
uv run pre-commit run --all-files
```

## Project Structure

```
bachelors/
├── model/
│   ├── preprocessing.ipynb    # Data preprocessing pipeline
│   ├── requirements.txt       # Model-specific dependencies (legacy)
│   └── data/                  # Dataset storage
├── .github/
│   ├── workflows/             # CI/CD pipelines
│   └── PULL_REQUEST_TEMPLATE/ # PR templates
├── pyproject.toml             # Project dependencies and config
└── README.md
```
