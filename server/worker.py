import logging
import time
import uuid

from .config import require_core_secrets, settings
from .db import Database
from .github_client import GitHubClient
from .indexer import Indexer
from .qdrant_store import QdrantStore

# Structured logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}',
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("mcp.worker")

# Worker instance diagnostics
WORKER_INSTANCE_ID = str(uuid.uuid4())
WORKER_START_TIME = time.time()


def main() -> None:
    logger.info("worker_startup instance_id=%s", WORKER_INSTANCE_ID)
    require_core_secrets()

    logger.info("initializing_database")
    db = Database(settings.database_url)

    logger.info("initializing_github_client")
    gh = GitHubClient(settings.github_pat)

    logger.info("initializing_qdrant url=%s", settings.qdrant_url)
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

    logger.info("worker_ready instance_id=%s", WORKER_INSTANCE_ID)

    while True:
        job = db.newest_pending_job()
        if not job:
            time.sleep(5)
            continue

        job_id = job["job_id"]
        owner = job["owner"]
        repo = job["repo"]
        ref = job["ref"]

        logger.info("job_started job_id=%s repo=%s/%s ref=%s", job_id, owner, repo, ref)
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
            logger.info("job_completed job_id=%s repo=%s/%s", job_id, owner, repo)
        except Exception as exc:  # noqa: BLE001
            db.update_job(job_id, "ERROR", str(exc)[:4000], 1.0)
            logger.error("job_failed job_id=%s repo=%s/%s error=%s", job_id, owner, repo, str(exc))


if __name__ == "__main__":
    main()
