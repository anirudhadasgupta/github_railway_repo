import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    github_owner: str = os.getenv("GITHUB_OWNER", "anirudhadasgupta")
    github_pat: str = os.environ.get("GITHUB_PAT", "")
    openai_api_key: str = os.environ.get("OPENAI_API_KEY", "")

    database_url: str = os.environ.get("DATABASE_URL", "")

    qdrant_url: str = os.environ.get("QDRANT_URL", "")
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

    allow_token: str | None = os.getenv("ALLOW_TOKEN")


settings = Settings()
