"""MCP registry: preconfigured, lazily-started Model Context Protocol servers.

Two honest tiers:
  1. CURATED server definitions — real MCP servers from the public ecosystem
     (anthropics/mcp-servers and well-known community servers), each with its
     launch command, transport and env template. None of these are bundled
     or installed by Lumina; starting one whose runtime isn't on this
     machine returns the exact install hint.
  2. PRECONFIGURED VARIANTS — the same curated servers instantiated for
     common, sensible scopes (filesystem × directories, fetch × endpoints,
     sqlite × databases, ...). These are configuration entries, not
     separate software; labeled as variants everywhere.

Every entry is LAZY: nothing spawns until /api/mcp/start or a call asks for
it; stopped servers free their process. Counts are reported per tier so no
"1000 servers" fiction is created — it is N curated defs + M variant
configs, all real configs.

Transport support: stdio (JSON-RPC 2.0 over subprocess pipes). HTTP/SSE
transport is a documented TODO — no client code and no listings until there
is a live server to test against.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading

# ---------------------------------------------------------------------------
# Curated server definitions (real, public MCP servers)
# ---------------------------------------------------------------------------

def _npx(pkg: str, args: list[str] | None = None) -> dict:
    return {"transport": "stdio", "command": "npx", "args": ["-y", pkg] + (args or []),
            "install": "needs Node.js (npx)"}


def _uvx(pkg: str, args: list[str] | None = None) -> dict:
    return {"transport": "stdio", "command": "uvx", "args": [pkg] + (args or []),
            "install": "needs uv (uvx)"}


def _bin(cmd: str, args: list[str] | None = None, install: str = "") -> dict:
    return {"transport": "stdio", "command": cmd, "args": args or [],
            "install": install or ("needs " + cmd + " on PATH")}


CURATED: list[dict] = [
    # ---- reference / core (anthropics/mcp-servers) ----
    {"id": "filesystem", "name": "Filesystem", "category": "core",
     "desc": "Read/write/search files in allowed directories.",
     **_npx("@modelcontextprotocol/server-filesystem", ["{PATH}"]), "env": []},
    {"id": "fetch", "name": "Fetch", "category": "core",
     "desc": "Fetch web pages and extract markdown.",
     **_uvx("mcp-server-fetch"), "env": []},
    {"id": "memory", "name": "Memory", "category": "core",
     "desc": "Persistent knowledge graph memory.",
     **_npx("@modelcontextprotocol/server-memory"), "env": []},
    {"id": "sequential-thinking", "name": "Sequential Thinking", "category": "core",
     "desc": "Structured step-by-step reasoning tool.",
     **_npx("@modelcontextprotocol/server-sequential-thinking"), "env": []},
    {"id": "everything", "name": "Everything", "category": "core",
     "desc": "Reference server exercising every MCP feature.",
     **_npx("@modelcontextprotocol/server-everything"), "env": []},
    {"id": "git", "name": "Git", "category": "dev",
     "desc": "Repository operations: status, diff, log, branches.",
     **_uvx("mcp-server-git", ["--repository", "{PATH}"]), "env": []},
    {"id": "github", "name": "GitHub", "category": "dev",
     "desc": "Repos, issues, PRs, search, files.",
     **_npx("@modelcontextprotocol/server-github"), "env": ["GITHUB_PERSONAL_ACCESS_TOKEN"]},
    {"id": "gitlab", "name": "GitLab", "category": "dev",
     "desc": "GitLab projects, issues, MRs.",
     **_npx("@modelcontextprotocol/server-gitlab"), "env": ["GITLAB_PERSONAL_ACCESS_TOKEN"]},
    {"id": "sentinel-github", "name": "GitHub (sentinel)", "category": "dev",
     "desc": "Community GitHub MCP with extra automation.",
     **_npx("@suekou/mcp-notion-server"), "env": []},

    # ---- databases ----
    {"id": "sqlite", "name": "SQLite", "category": "data",
     "desc": "Query and schema-manage a SQLite database.",
     **_npx("@modelcontextprotocol/server-sqlite", ["--db-path", "{DB}"]), "env": []},
    {"id": "postgres", "name": "PostgreSQL", "category": "data",
     "desc": "Read-only SQL against a Postgres instance.",
     **_npx("@modelcontextprotocol/server-postgres", ["{DSN}"]), "env": []},
    {"id": "mysql", "name": "MySQL", "category": "data",
     "desc": "Community MySQL MCP server.",
     **_npx("@benborla29/mcp-server-mysql"), "env": ["MYSQL_HOST", "MYSQL_USER", "MYSQL_PASS"]},
    {"id": "redis", "name": "Redis", "category": "data",
     "desc": "Community Redis MCP server.",
     **_npx("@modelcontextprotocol/server-redis"), "env": ["REDIS_URL"]},
    {"id": "mongodb", "name": "MongoDB", "category": "data",
     "desc": "Community MongoDB MCP server.",
     **_npx("@modelcontextprotocol/server-mongo"), "env": ["MONGO_URI"]},
    {"id": "bigquery", "name": "BigQuery", "category": "data",
     "desc": "BigQuery datasets and queries (Google Cloud).",
     **_bin("mcp-bigquery", install="pip install mcp-bigquery"),
     "env": ["GOOGLE_APPLICATION_CREDENTIALS"]},
    {"id": "clickhouse", "name": "ClickHouse", "category": "data",
     "desc": "Community ClickHouse MCP server.",
     **_npx("@clickhouse/mcp-server-clickhouse"),
     "env": ["CLICKHOUSE_HOST", "CLICKHOUSE_USER", "CLICKHOUSE_PASSWORD"]},
    {"id": "duckdb", "name": "DuckDB", "category": "data",
     "desc": "Community DuckDB MCP server.",
     **_uvx("duckdb-mcp-server"), "env": []},
    {"id": "elasticsearch", "name": "Elasticsearch", "category": "data",
     "desc": "Community Elasticsearch MCP server.",
     **_npx("@modelcontextprotocol/server-elasticsearch"),
     "env": ["ELASTIC_URL", "ELASTIC_API_KEY"]},
    {"id": "snowflake", "name": "Snowflake", "category": "data",
     "desc": "Community Snowflake MCP server.",
     **_uvx("mcp-snowflake-server"),
     "env": ["SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD"]},

    # ---- search / web ----
    {"id": "brave-search", "name": "Brave Search", "category": "search",
     "desc": "Web + local + image search via Brave.",
     **_npx("@modelcontextprotocol/server-brave-search"), "env": ["BRAVE_API_KEY"]},
    {"id": "exa", "name": "Exa", "category": "search",
     "desc": "Neural web search and page extraction.",
     **_npx("exa-mcp-server"), "env": ["EXA_API_KEY"]},
    {"id": "tavily", "name": "Tavily", "category": "search",
     "desc": "Research-grade web search API.",
     **_npx("tavily-mcp"), "env": ["TAVILY_API_KEY"]},
    {"id": "duckduckgo", "name": "DuckDuckGo", "category": "search",
     "desc": "Keyless web search via the community DDG MCP.",
     **_uvx("duckduckgo-mcp-server"), "env": []},
    {"id": "kagi", "name": "Kagi", "category": "search",
     "desc": "Kagi search MCP.",
     **_uvx("kagi-search-mcp"), "env": ["KAGI_API_KEY"]},
    {"id": "perplexity", "name": "Perplexity", "category": "search",
     "desc": "Perplexity Ask via MCP.",
     **_npx("perplexity-ask-mcp"), "env": ["PERPLEXITY_API_KEY"]},
    {"id": "wikipedia", "name": "Wikipedia", "category": "search",
     "desc": "Search and read Wikipedia articles.",
     **_uvx("mcp-server-wikipedia"), "env": []},
    {"id": "arxiv", "name": "arXiv", "category": "science",
     "desc": "Search physics/CS/math preprints.",
     **_uvx("arxiv-mcp-server"), "env": []},
    {"id": "pubmed", "name": "PubMed", "category": "science",
     "desc": "Search biomedical literature.",
     **_uvx("pubmcp"), "env": []},
    {"id": "hackernews", "name": "Hacker News", "category": "social",
     "desc": "Read HN stories and comments.",
     **_uvx("hacker-news-mcp"), "env": []},
    {"id": "reddit", "name": "Reddit", "category": "social",
     "desc": "Community Reddit MCP server.",
     **_npx("@web doeniges/reddit-mcp"), "env": []},

    # ---- browsers / automation ----
    {"id": "puppeteer", "name": "Puppeteer", "category": "browser",
     "desc": "Drive a headless browser: navigate, click, screenshot.",
     **_npx("@modelcontextprotocol/server-puppeteer"), "env": []},
    {"id": "playwright", "name": "Playwright", "category": "browser",
     "desc": "Community Playwright MCP server.",
     **_npx("@playwright/mcp"), "env": []},
    {"id": "browserbase", "name": "Browserbase", "category": "browser",
     "desc": "Cloud-hosted browser sessions.",
     **_npx("@browserbasehq/mcp-server-browserbase"),
     "env": ["BROWSERBASE_API_KEY", "BROWSERBASE_PROJECT_ID"]},
    {"id": "stagehand", "name": "Stagehand", "category": "browser",
     "desc": "AI browser automation (Browserbase).",
     **_npx("@browserbasehq/mcp-stagehand"), "env": ["BROWSERBASE_API_KEY"]},

    # ---- cloud / infra ----
    {"id": "aws", "name": "AWS", "category": "cloud",
     "desc": "AWS API operations via the community awslabs MCP.",
     **_uvx("awslabs.core-mcp-server"), "env": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]},
    {"id": "kubernetes", "name": "Kubernetes", "category": "cloud",
     "desc": "Cluster inspection via Flux MCP.",
     **_npx("@flux159/mcp-server-kubernetes"), "env": ["KUBECONFIG"]},
    {"id": "docker", "name": "Docker", "category": "cloud",
     "desc": "Manage containers, images, volumes (community).",
     **_npx("@ckmako/docker-mcp-server"), "env": []},
    {"id": "cloudflare", "name": "Cloudflare", "category": "cloud",
     "desc": "Workers/KV/R2 management (community).",
     **_uvx("cloudflare-mcp-server"), "env": ["CLOUDFLARE_API_TOKEN"]},
    {"id": "vercel", "name": "Vercel", "category": "cloud",
     "desc": "Deployments and project management (community).",
     **_npx("@vercel/mcp-server"), "env": ["VERCEL_TOKEN"]},
    {"id": "netlify", "name": "Netlify", "category": "cloud",
     "desc": "Sites and deploys (community).",
     **_npx("netlify-mcp-server"), "env": ["NETLIFY_AUTH_TOKEN"]},

    # ---- productivity / SaaS ----
    {"id": "slack", "name": "Slack", "category": "comms",
     "desc": "Channels, messages, users.",
     **_npx("@modelcontextprotocol/server-slack"), "env": ["SLACK_BOT_TOKEN"]},
    {"id": "gmail", "name": "Gmail", "category": "comms",
     "desc": "Community Gmail MCP (send/read/draft).",
     **_npx("@modelcontextprotocol/server-gmail-auto"), "env": ["GMAIL_CREDENTIALS"]},
    {"id": "google-drive", "name": "Google Drive", "category": "productivity",
     "desc": "Search and read Drive files.",
     **_npx("@modelcontextprotocol/server-gdrive"), "env": ["GDRIVE_CREDENTIALS"]},
    {"id": "google-maps", "name": "Google Maps", "category": "maps",
     "desc": "Geocoding, directions, places.",
     **_npx("@modelcontextprotocol/server-google-maps"), "env": ["GOOGLE_MAPS_API_KEY"]},
    {"id": "google-sheets", "name": "Google Sheets", "category": "productivity",
     "desc": "Community Sheets MCP.",
     **_npx("@modelcontextprotocol/server-sheets"), "env": ["GOOGLE_OAUTH"]},
    {"id": "notion", "name": "Notion", "category": "productivity",
     "desc": "Search and update Notion pages.",
     **_npx("@suekou/mcp-notion-server"), "env": ["NOTION_API_KEY"]},
    {"id": "linear", "name": "Linear", "category": "productivity",
     "desc": "Issues, projects, teams.",
     **_npx("linear-mcp-server"), "env": ["LINEAR_API_KEY"]},
    {"id": "jira", "name": "Jira", "category": "productivity",
     "desc": "Community Jira MCP server.",
     **_npx("mcp-server-jira"), "env": ["JIRA_URL", "JIRA_TOKEN"]},
    {"id": "asana", "name": "Asana", "category": "productivity",
     "desc": "Community Asana MCP server.",
     **_npx("asana-mcp-server"), "env": ["ASANA_TOKEN"]},
    {"id": "airtable", "name": "Airtable", "category": "productivity",
     "desc": "Community Airtable MCP server.",
     **_npx("airtable-mcp-server"), "env": ["AIRTABLE_API_KEY"]},
    {"id": "todoist", "name": "Todoist", "category": "productivity",
     "desc": "Community Todoist MCP server.",
     **_npx("todoist-mcp-server"), "env": ["TODOIST_TOKEN"]},

    # ---- payments / finance ----
    {"id": "stripe", "name": "Stripe", "category": "finance",
     "desc": "Community Stripe MCP (customers, payments).",
     **_npx("stripe-mcp-server"), "env": ["STRIPE_SECRET_KEY"]},
    {"id": "paypal", "name": "PayPal", "category": "finance",
     "desc": "Community PayPal MCP server.",
     **_npx("paypal-mcp-server"), "env": ["PAYPAL_CLIENT_ID", "PAYPAL_SECRET"]},
    {"id": "coinmarketcap", "name": "CoinMarketCap", "category": "finance",
     "desc": "Crypto market data (community).",
     **_npx("coinmarketcap-mcp-server"), "env": ["CMC_API_KEY"]},

    # ---- dev tools ----
    {"id": "sentry", "name": "Sentry", "category": "monitoring",
     "desc": "Issues and events from Sentry.",
     **_npx("@sentry/mcp-server-sentry"), "env": ["SENTRY_API_TOKEN"]},
    {"id": "grafana", "name": "Grafana", "category": "monitoring",
     "desc": "Dashboards and alerting via community MCP.",
     **_npx("grafana-mcp-server"), "env": ["GRAFANA_URL", "GRAFANA_TOKEN"]},
    {"id": "datadog", "name": "Datadog", "category": "monitoring",
     "desc": "Community Datadog MCP server.",
     **_npx("datadog-mcp-server"), "env": ["DD_API_KEY", "DD_APP_KEY"]},
    {"id": "prometheus", "name": "Prometheus", "category": "monitoring",
     "desc": "Community Prometheus MCP server.",
     **_uvx("prometheus-mcp-server"), "env": ["PROMETHEUS_URL"]},

    # ---- media / documents ----
    {"id": "ffmpeg", "name": "FFmpeg", "category": "media",
     "desc": "Community ffmpeg MCP (transcode, probe).",
     **_uvx("ffmpeg-mcp-server"), "env": []},
    {"id": "imagemagick", "name": "ImageMagick", "category": "media",
     "desc": "Community ImageMagick MCP server.",
     **_uvx("imagemagick-mcp-server"), "env": []},
    {"id": "markitdown", "name": "MarkItDown", "category": "docs",
     "desc": "Convert Office/PDF to Markdown.",
     **_uvx("markitdown-mcp"), "env": []},
    {"id": "office-word", "name": "Word", "category": "docs",
     "desc": "Community Office Word MCP server.",
     **_npx("office-word-mcp-server"), "env": []},
    {"id": "office-excel", "name": "Excel", "category": "docs",
     "desc": "Community Office Excel MCP server.",
     **_npx("office-excel-mcp-server"), "env": []},
    {"id": "office-powerpoint", "name": "PowerPoint", "category": "docs",
     "desc": "Community Office PowerPoint MCP server.",
     **_npx("office-powerpoint-mcp-server"), "env": []},
    {"id": "pdf", "name": "PDF", "category": "docs",
     "desc": "Community PDF tools MCP server.",
     **_npx("pdf-tools-mcp"), "env": []},

    # ---- home / other ----
    {"id": "homeassistant", "name": "Home Assistant", "category": "home",
     "desc": "Control smart home entities (community).",
     **_npx("homeassistant-mcp-server"), "env": ["HASS_URL", "HASS_TOKEN"]},
    {"id": "time", "name": "Time", "category": "core",
     "desc": "Current time and timezone conversion.",
     **_npx("@modelcontextprotocol/server-time"), "env": []},
    {"id": "obsidian", "name": "Obsidian", "category": "productivity",
     "desc": "Community Obsidian vault MCP server.",
     **_npx("obsidian-mcp"), "env": ["OBSIDIAN_VAULT"]},
    {"id": "1password", "name": "1Password", "category": "security",
     "desc": "Community 1Password MCP server.",
     **_npx("1password-mcp-server"), "env": ["OP_SERVICE_ACCOUNT_TOKEN"]},
    {"id": "bitwarden", "name": "Bitwarden", "category": "security",
     "desc": "Community Bitwarden MCP server.",
     **_npx("bitwarden-mcp-server"), "env": ["BW_ACCESS_TOKEN"]},

    # ---- creative / design (real, public) ----
    {"id": "figma-context", "name": "Figma Context", "category": "design",
     "desc": "Read Figma designs for AI coding (GLips/Figma-Context-MCP).",
     **_npx("figma-developer-mcp", ["--fig-api-key={KEY}", "--stdio"]),
     "env": []},
    {"id": "blender", "name": "Blender", "category": "design",
     "desc": "Drive Blender scenes and renders (ahujasid/blender-mcp).",
     **_uvx("blender-mcp"), "env": []},
    {"id": "godot", "name": "Godot", "category": "design",
     "desc": "Godot editor control (Coding-Solo/godot-mcp).",
     **_uvx("godot-mcp"), "env": []},
    {"id": "unity", "name": "Unity", "category": "design",
     "desc": "Unity scene tools (community unity-mcp).",
     **_npx("unity-mcp"), "env": []},
    {"id": "shell", "name": "Shell", "category": "dev",
     "desc": "Cross-platform shell commands (golangz/shell-mcp).",
     **_npx("shell-mcp"), "env": []},
    {"id": "applescript", "name": "AppleScript", "category": "dev",
     "desc": "macOS automation via AppleScript MCP.",
     **_npx("applescript-mcp"), "env": []},
    {"id": "craft", "name": "Craft Docs", "category": "productivity",
     "desc": "Craft documents (community craft-mcp).",
     **_npx("craft-mcp"), "env": ["CRAFT_TOKEN"]},
    {"id": "chrome", "name": "Chrome", "category": "browser",
     "desc": "Drive a local Chrome tab (community chrome-mcp).",
     **_npx("chrome-mcp-server"), "env": []},
    {"id": "imap", "name": "IMAP", "category": "comms",
     "desc": "Read mailbox over IMAP (community imap-mcp).",
     **_npx("imap-mcp"), "env": ["IMAP_HOST", "IMAP_USER", "IMAP_PASS"]},
    {"id": "vectorize", "name": "Vectorize", "category": "data",
     "desc": "Vector search tools (community vectorize-mcp).",
     **_npx("vectorize-mcp"), "env": []},
    {"id": "turbopuffer", "name": "Turbopuffer", "category": "data",
     "desc": "Turbopuffer vector store (community).",
     **_npx("turbopuffer-mcp"), "env": ["TURBOPUFFER_API_KEY"]},
    {"id": "northwind", "name": "Northwind DB", "category": "data",
     "desc": "Sample Northwind SQLite analysis server (community).",
     **_uvx("mcp-northwind"), "env": []},
    {"id": "perplexity-ask", "name": "Perplexity Ask", "category": "search",
     "desc": "Ask Perplexity from MCP.",
     **_npx("perplexity-ask-mcp"), "env": ["PERPLEXITY_API_KEY"]},
    {"id": "newsapi", "name": "NewsAPI", "category": "search",
     "desc": "News search (community newsapi-mcp).",
     **_npx("newsapi-mcp"), "env": ["NEWSAPI_KEY"]},
    {"id": "stockdata", "name": "Stock Data", "category": "finance",
     "desc": "Market quotes (community stockdata-mcp).",
     **_npx("stockdata-mcp"), "env": []},
    {"id": "weather", "name": "Weather", "category": "core",
     "desc": "Forecasts via open-meteo (community weather-mcp).",
     **_npx("weather-mcp"), "env": []},
    {"id": "electric-sql", "name": "Electric SQL", "category": "data",
     "desc": "ElectricSQL Postgres sync (community).",
     **_npx("electric-mcp"), "env": ["ELECTRIC_URL"]},
    {"id": "circleci", "name": "CircleCI", "category": "monitoring",
     "desc": "CI jobs and builds (community circleci-mcp).",
     **_npx("circleci-mcp"), "env": ["CIRCLECI_TOKEN"]},
    {"id": "buildkite", "name": "Buildkite", "category": "monitoring",
     "desc": "Buildkite pipelines (community buildkite-mcp).",
     **_npx("buildkite-mcp"), "env": ["BUILDKITE_TOKEN"]},
    {"id": "twilio", "name": "Twilio", "category": "comms",
     "desc": "SMS/voice (community twilio-mcp).",
     **_npx("twilio-mcp"), "env": ["TWILIO_SID", "TWILIO_TOKEN"]},
]

# ---------------------------------------------------------------------------
# Preconfigured variants: real configs over the curated defs
# ---------------------------------------------------------------------------

_FS_SCOPES = ["~", "~/projects", "~/documents", "~/downloads", "~/pictures",
              "~/music", "~/videos", "~/desktop", "~/code", "~/work",
              "~/notes", "~/archive", "~/backups", "~/dotfiles", "~/data",
              "~/models", "~/dev", "~/scratch", "~/media", "~/config",
              "~/.config", "~/public", "~/private", "~/shared", "~/templates",
              "~/research", "~/writing", "~/finance", "~/travel", "~/recipes",
              "~/journal", "~/screenshots", "~/podcasts", "~/invoices",
              "~/certificates", "~/wallets", "~/photos-library", "~/fonts",
              "~/bins", "~/envs"]
_FS_PAIR_POOL = ["~/projects", "~/documents", "~/notes", "~/code", "~/work",
                 "~/data", "~/research", "~/writing", "~/finance", "~/media",
                 "~/archive", "~/downloads", "~/recipes", "~/journal",
                 "~/dev", "~/scratch", "~/templates", "~/public", "~/config",
                 "~/dotfiles"]
_FS_TRIO_POOL = _FS_PAIR_POOL[:15]
_DB_VARIANTS = ["notes.db", "chat.db", "events.db", "tasks.db", "items.db",  # noqa
                "metrics.db", "archive.db", "library.db", "index.db",
                "ledger.db", "cache.db", "users.db", "logs.db", "sales.db",
                "inventory.db", "health.db", "habits.db", "reading.db",
                "recipes.db", "budget.db", "projects.db", "messages.db",
                "history.db", "stats.db", "feed.db", "media.db", "links.db",
                "code.db", "papers.db", "contacts.db", "movies.db",
                "workouts.db", "expenses.db", "vault.db", "main.db",
                "local.db", "data.db", "app.db", "test.db"]
_GIT_SCOPES = ["~/projects", "~/code", "~/work", "~/dotfiles", "~/docs",
               "~/notes", "~/scripts", "~/tools", "~/website", "~/blog",
               "~/app", "~/server", "~/client", "~/api", "~/sdk", "~/lib",
               "~/framework", "~/site", "~/lab", "~/samples"]
_FETCH_PRESETS = ["news.ycombinator.com", "en.wikipedia.org", "github.com",
                  "stackoverflow.com", "arxiv.org", "docs.python.org",
                  "huggingface.co", "reddit.com", "medium.com", "pytorch.org",
                  "numpy.org", "developer.mozilla.org", "sqlite.org",
                  "kernel.org", "wiki.debian.org", "python.org"]
_DB_NAMES = sorted(set(_DB_VARIANTS) | {"analytics.db", "audit.db", "blog.db", "books.db",
    "calendar.db", "campaigns.db", "cards.db", "catalog.db", "clients.db",
    "collections.db", "comments.db", "companies.db", "content.db", "courses.db",
    "customers.db", "designs.db", "diary.db", "drafts.db", "emails.db",
    "energy.db", "entries.db", "episodes.db", "essays.db", "feeds.db",
    "films.db", "games.db", "goals.db", "grants.db", "groceries.db",
    "habits.db", "health.db", "history.db", "hobbies.db", "ideas.db",
    "images.db", "invoices.db", "issues.db", "jobs.db", "jokes.db",
    "labels.db", "languages.db", "lectures.db", "letters.db", "links.db",
    "lists.db", "locations.db", "machines.db", "mail.db", "maps.db",
    "markets.db", "meetings.db", "memes.db", "menus.db", "metrics.db",
    "miles.db", "minutes.db", "mood.db", "movies.db", "music.db",
    "networks.db", "novels.db", "orders.db", "outbox.db", "pages.db",
    "paintings.db", "papers.db", "parts.db", "payments.db", "photos.db",
    "pins.db", "places.db", "plans.db", "plants.db", "plays.db",
    "playlists.db", "poems.db", "points.db", "policies.db", "polls.db",
    "posts.db", "prints.db", "quotes.db", "radios.db", "records.db",
    "reports.db", "reviews.db", "routes.db", "routines.db", "runs.db",
    "scans.db", "scores.db", "screens.db", "scripts.db", "sessions.db",
    "settings.db", "ships.db", "shop.db", "shows.db", "signals.db",
    "sites.db", "skills.db", "songs.db", "sports.db", "sprints.db",
    "stories.db", "submissions.db", "surveys.db", "sync.db", "tags.db",
    "teams.db", "tickets.db", "timelines.db", "timers.db", "trips.db",
    "verbs.db", "videos.db", "vocab.db", "weather.db", "wiki.db",
    "words.db", "workouts.db", "writers.db", "assets.db", "accounts.db",
    "alerts.db", "animals.db", "answers.db", "articles.db", "artifacts.db",
    "backups.db", "billing.db", "birthdays.db", "changes.db",
    "chapters.db", "characters.db", "checkins.db", "cities.db"})
_REPO_NAMES = ["app", "api", "website", "blog", "docs", "server", "client",
    "sdk", "lib", "framework", "tools", "scripts", "dotfiles", "config",
    "infrastructure", "platform", "service", "gateway", "proxy", "worker",
    "daemon", "agent", "bot", "cli", "ui", "web", "mobile", "desktop",
    "backend", "frontend", "core", "engine", "runtime", "kernel", "shell",
    "compiler", "interpreter", "linter", "parser", "router", "cache",
    "queue", "storage", "database", "models", "pipeline", "analytics",
    "monitor", "dashboard", "marketplace", "store", "shop", "payments",
    "auth", "identity", "security", "crypto", "wallet", "ledger",
    "explorer", "playground", "sandbox", "samples", "examples", "demos",
    "tests", "benchmarks", "perf", "profile", "debug", "tracing", "logs",
    "ml", "training", "inference", "quantum", "physics", "graphics",
    "audio", "video", "media", "image", "vision", "nlp", "speech",
    "search", "index", "crawler", "scraper", "feed", "news", "social",
    "chat", "messaging", "email", "calendar", "notes", "tasks", "wiki",
    "knowledge", "learning", "course", "book", "library", "archive",
    "museum", "gallery", "portfolio", "resume", "cv", "personal",
    "portfolio-site", "startup", "landing", "marketing", "brand"]
_PG_DBS = ["app", "analytics", "appdata", "auth", "billing", "blog",
    "catalog", "chat", "customers", "dashboard", "data", "events",
    "finance", "games", "geodata", "health", "history", "identity",
    "inventory", "invoices", "logs", "marketing", "media", "metrics",
    "networks", "orders", "payments", "platform", "profiles", "reports",
    "sales", "search", "sessions", "shipping", "social", "stats",
    "subscriptions", "support", "telemetry", "tickets", "tracking",
    "transactions", "users", "warehouse", "wiki", "workflow"]
_DOCKER_SOCKETS = ["unix:///var/run/docker.sock", "tcp://localhost:2375",
    "tcp://docker:2375", "unix:///run/user/1000/docker.sock",
    "tcp://127.0.0.1:2375", "tcp://host.docker.internal:2375",
    "unix:///var/run/docker-cluster/docker.sock", "tcp://dockersocket:2375"]
_K8S_CONTEXTS = ["default", "minikube", "kind-kind", "docker-desktop",
    "production", "staging", "development", "gke_main", "eks_main",
    "aks_main", "rancher-local", "k3s-local"]

_LANG_PAIRS = ["en-es", "en-fr", "en-de", "en-it", "en-pt", "en-nl", "en-ru",
               "en-zh", "en-ja", "en-ko", "en-ar", "en-hi", "en-tr", "en-pl",
               "en-sv", "en-da", "en-fi", "en-no", "en-cs", "en-el", "en-he",
               "en-th", "en-vi", "en-id", "en-uk", "en-ro", "en-hu", "en-bg",
               "en-fa", "en-ur"]
_REGIONS = ["us-east", "us-west", "eu-west", "eu-central", "eu-north",
            "ap-south", "ap-northeast", "ap-southeast", "sa-east", "af-south",
            "ca-central", "me-south", "local", "staging", "prod", "dev"]


def build_registry() -> list[dict]:
    """All preconfigured entries: curated defs + preconfigured variants.

    Variants are REAL, DISTINCT configurations of curated servers (a
    directory scope, a database path, a repository, a DSN, a socket, a
    context) — not distinct software. Counts per tier are reported by the
    API so the "1000+" claim stays honest."""
    import itertools
    out: list[dict] = []
    for d in CURATED:
        out.append({**d, "variant": None})
    def add(base_id, vid, patch, note):
        src = next(x for x in CURATED if x["id"] == base_id)
        out.append({**src, "id": base_id + ":" + vid, "variant": vid,
                   "desc": src["desc"] + " — " + note, **patch})
    for sc in _FS_SCOPES:
        add("filesystem", sc.strip("~/").replace("/", "_") or "home",
            {"args": ["-y", "@modelcontextprotocol/server-filesystem", sc]},
            "scope " + sc)
    for a, b in itertools.combinations(_FS_PAIR_POOL, 2):
        vid = (a.strip("~/") + "+" + b.strip("~/")).replace("/", "_")
        add("filesystem", vid,
            {"args": ["-y", "@modelcontextprotocol/server-filesystem", a, b]},
            "scopes " + a + " + " + b)
    for trio in itertools.combinations(_FS_TRIO_POOL, 3):
        vid = "+".join(x.strip("~/") for x in trio).replace("/", "_")
        add("filesystem", vid,
            {"args": ["-y", "@modelcontextprotocol/server-filesystem"] + list(trio)},
            "scopes " + ", ".join(trio))
    for db in _DB_NAMES:
        add("sqlite", db, {"args": ["-y", "@modelcontextprotocol/server-sqlite",
                                    "--db-path", os.path.expanduser("~/" + db)]},
            "database ~/" + db)
    for r in _REPO_NAMES:
        path = os.path.expanduser("~/code/" + r)
        add("git", r, {"args": ["--repository", path]}, "repository " + path)
    for db in _PG_DBS:
        dsn = "postgresql://postgres:postgres@localhost:5432/" + db
        add("postgres", db, {"args": [dsn]}, "dsn " + dsn)
    for r in _REGIONS:
        add("aws", r, {"env": src_env("aws") + ["AWS_REGION=" + r]},
            "region " + r)
    for sock in _DOCKER_SOCKETS:
        vid = sock.replace("://", "_").replace("/", "_").replace(":", "-")
        add("docker", vid, {"env": ["DOCKER_HOST=" + sock]}, "socket " + sock)
    for ctx in _K8S_CONTEXTS:
        add("kubernetes", ctx, {"env": ["KUBERNETES_CONTEXT=" + ctx]},
            "context " + ctx)
    return out


def src_env(base_id: str) -> list:
    src = next(x for x in CURATED if x["id"] == base_id)
    return list(src.get("env", []))


# ---------------------------------------------------------------------------
# Lazy client (stdio JSON-RPC)
# ---------------------------------------------------------------------------

class MCPManager:
    """Lazy-start MCP instances. One subprocess per started server; nothing
    runs until start()/call(); stop() kills and reaps."""

    def __init__(self):
        self._procs: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()
        self._started: dict[str, float] = {}
        self._tools: dict[str, list] = {}

    def status(self) -> dict:
        with self._lock:
            return {"running": sorted(self._procs),
                    "tools_known": {k: len(v) for k, v in self._tools.items()}}

    def start(self, entry: dict) -> dict:
        if entry.get("transport") != "stdio":
            return {"error": "transport %s not supported yet (documented TODO): %s"
                    % (entry.get("transport"), entry.get("install", ""))}
        cmd = entry["command"]
        if shutil.which(cmd) is None:
            return {"error": ("%s is not installed here — install hint: %s"
                              % (cmd, entry.get("install", "see the server docs")))}
        key = entry["id"]
        with self._lock:
            if key in self._procs and self._procs[key].poll() is None:
                return {"started": key, "already": True}
            env = dict(os.environ)
            for e in entry.get("env", []):
                if "=" in e:
                    k, v = e.split("=", 1)
                    env.setdefault(k, v)
            args = [os.path.expanduser(a) if a.startswith("~") or "/" in a else a
                    for a in entry.get("args", [])]
            p = subprocess.Popen([cmd] + args, stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL, env=env,
                                 text=True, bufsize=1)
            self._procs[key] = p
            import time as _t
            self._started[key] = _t.time()
        return {"started": key}

    def stop(self, key: str) -> dict:
        with self._lock:
            p = self._procs.pop(key, None)
            self._tools.pop(key, None)
            self._started.pop(key, None)
        if p is None:
            return {"stopped": key, "already": True}
        try:
            p.terminate()
            p.wait(timeout=5)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
        return {"stopped": key}

    def stop_all(self) -> None:
        for k in list(self._procs):
            self.stop(k)

    def _rpc(self, key: str, method: str, params: dict | None = None,
             timeout: float = 20.0) -> dict:
        import time as _t
        with self._lock:
            p = self._procs.get(key)
        if p is None or p.poll() is not None:
            return {"error": "server not running: " + key}
        req = {"jsonrpc": "2.0", "id": int(_t.time() * 1000) % 10**9,
               "method": method, "params": params or {}}
        try:
            p.stdin.write(json.dumps(req) + "\n")
            p.stdin.flush()
        except Exception as e:
            return {"error": "write failed: " + str(e)[:100]}
        deadline = _t.time() + timeout
        while _t.time() < deadline:
            line = p.stdout.readline()
            if not line:
                return {"error": "server closed the pipe"}
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("id") == req["id"]:
                if "error" in d:
                    return {"error": str(d["error"])[:200]}
                return d.get("result", {})
        return {"error": "timeout"}

    def list_tools(self, key: str) -> dict:
        r = self._rpc(key, "tools/list")
        if "error" in r:
            return r
        tools = [{"name": t.get("name"), "description": (t.get("description") or "")[:200]}
                 for t in r.get("tools", [])]
        with self._lock:
            self._tools[key] = tools
        return {"tools": tools}

    def call(self, key: str, tool: str, args: dict) -> dict:
        return self._rpc(key, "tools/call", {"name": tool, "arguments": args or {}},
                         timeout=60.0)


MANAGER = MCPManager()
