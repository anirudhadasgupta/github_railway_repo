import os
from dataclasses import dataclass


def _detect_qdrant_url() -> str:
    """Resolve Qdrant URL from common Railway/env patterns to reduce manual setup."""
    candidates = [
        os.environ.get("QDRANT_URL"),
        os.environ.get("QDRANT_HTTP_URL"),
        os.environ.get("QDRANT_ENDPOINT"),
        os.environ.get("QDRANT_INTERNAL_URL"),
        os.environ.get("QDRANT_PUBLIC_URL"),
    ]

    for url in candidates:
        if url:
            return url

    host = os.environ.get("QDRANT_HOST")
    port = os.environ.get("QDRANT_PORT") or os.environ.get("QDRANT_HTTP_PORT")
    if host and port:
        return f"http://{host}:{port}"

    return ""


@dataclass(frozen=True)
class Settings:
    github_owner: str = os.getenv("GITHUB_OWNER", "anirudhadasgupta")
    github_pat: str = os.environ.get("GITHUB_PAT", "")
    openai_api_key: str = os.environ.get("OPENAI_API_KEY", "")

    database_url: str = os.environ.get("DATABASE_URL", "")

    qdrant_url: str = _detect_qdrant_url()
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY")

    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    embedding_dims: int = int(os.getenv("EMBEDDING_DIMS", "1024"))

    # indexing defaults
    max_file_bytes: int = int(os.getenv("MAX_FILE_BYTES", "250000"))  # 250KB
    include_globs: str = os.getenv("INCLUDE_GLOBS", "")  # optional
    exclude_globs: str = os.getenv(
        "EXCLUDE_GLOBS",
        "node_modules/**,dist/**,build/**,.git/**,vendor/**,**/*.min.js,**/*.lock",
    )



settings = Settings()


def require_core_secrets() -> None:
    """Ensure mandatory secrets are configured before starting services."""
    missing = []
    if not settings.github_pat:
        missing.append("GITHUB_PAT")
    if not settings.openai_api_key:
        missing.append("OPENAI_API_KEY")
    if not settings.database_url:
        missing.append("DATABASE_URL")
    if not settings.qdrant_url:
        missing.append("QDRANT_URL")

    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"Missing required environment variable(s): {joined}. Set them as Railway service variables."
        )
