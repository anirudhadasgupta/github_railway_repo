import time

from .config import settings
from .db import Database
from .github_client import GitHubClient
from .indexer import Indexer
from .qdrant_store import QdrantStore


def main() -> None:
    db = Database(settings.database_url)
    gh = GitHubClient(settings.github_pat)
    qdrant = QdrantStore(settings.qdrant_url, settings.qdrant_api_key, settings.embedding_dims)
    qdrant.ensure_collection()

    indexer = Indexer(
        github=gh,
        qdrant=qdrant,
        db=db,
        openai_api_key=settings.openai_api_key,
        embedding_model=settings.embedding_model,
        embedding_dims=settings.embedding_dims,
    )

    while True:
        job = db.newest_pending_job()
        if not job:
            time.sleep(5)
            continue

        job_id = job["job_id"]
        owner = job["owner"]
        repo = job["repo"]
        ref = job["ref"]

        try:
            db.update_job(job_id, "RUNNING", "Indexing started", 0.05)
            indexer.index_repo_ref(
                owner=owner,
                repo=repo,
                ref=ref,
                max_bytes=settings.max_file_bytes,
                include_globs=settings.include_globs.split(",") if settings.include_globs else [],
                exclude_globs=settings.exclude_globs.split(",") if settings.exclude_globs else [],
            )
            db.update_job(job_id, "DONE", "Indexing complete", 1.0)
        except Exception as exc:  # noqa: BLE001
            db.update_job(job_id, "ERROR", str(exc)[:4000], 1.0)


if __name__ == "__main__":
    main()
