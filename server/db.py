import logging
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

from psycopg import sql
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

logger = logging.getLogger("mcp.db")


class Database:
    def __init__(self, url: str):
        self.pool = ConnectionPool(conninfo=url, kwargs={"autocommit": True}, min_size=1, max_size=5)
        self._run_migrations()

    def _run_migrations(self) -> None:
        """Run SQL migrations from the migrations directory with retry/backoff."""
        migrations_dir = Path(__file__).parent.parent / "migrations"
        if not migrations_dir.exists():
            return
        migration_files = sorted(migrations_dir.glob("*.sql"))
        for migration_file in migration_files:
            sql_content = migration_file.read_text()
            self._with_retry(
                lambda sql=sql_content: self._execute_migration(sql),
                max_attempts=5,
                context=f"migration:{migration_file.name}",
            )
            logger.info("db_migration_applied file=%s", migration_file.name)

    def _execute_migration(self, sql_content: str) -> None:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql_content)

    def _with_retry(self, fn, max_attempts: int = 5, context: str = "db_op"):
        """Execute function with exponential backoff retry."""
        attempt = 0
        while True:
            try:
                return fn()
            except Exception as e:
                attempt += 1
                if attempt >= max_attempts:
                    logger.error("db_retry_exhausted context=%s attempts=%d error=%s", context, attempt, str(e))
                    raise
                jitter = random.uniform(0.5, 1.5)
                sleep_time = min(2**attempt * jitter, 10)
                logger.warning("db_retry context=%s attempt=%d sleep=%.2f error=%s", context, attempt, sleep_time, str(e))
                time.sleep(sleep_time)

    @contextmanager
    def connection(self):
        with self.pool.connection() as conn:
            yield conn

    def execute(self, query: str, params: Optional[Sequence[Any]] = None) -> None:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params or [])

    def fetchone(self, query: str, params: Optional[Sequence[Any]] = None) -> Any:
        with self.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(query, params or [])
                return cur.fetchone()

    def fetchall(self, query: str, params: Optional[Sequence[Any]] = None) -> List[Any]:
        with self.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(query, params or [])
                return cur.fetchall()

    def upsert_repo(self, owner: str, name: str, default_branch: Optional[str]) -> int:
        query = sql.SQL(
            """
            INSERT INTO repos(owner, name, default_branch) VALUES (%s, %s, %s)
            ON CONFLICT (owner, name) DO UPDATE SET default_branch = excluded.default_branch
            RETURNING id
            """
        )
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (owner, name, default_branch))
                row = cur.fetchone()
                return int(row[0])

    def upsert_file(self, repo_id: int, path: str, ref: str, blob_sha: str, size_bytes: int) -> int:
        query = sql.SQL(
            """
            INSERT INTO files(repo_id, path, ref, blob_sha, size_bytes)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (repo_id, path, ref) DO UPDATE SET blob_sha=excluded.blob_sha, size_bytes=excluded.size_bytes, updated_at=now()
            RETURNING id
            """
        )
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (repo_id, path, ref, blob_sha, size_bytes))
                row = cur.fetchone()
                return int(row[0])

    def delete_chunks(self, repo_id: int, path: str, ref: str) -> None:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM chunks WHERE repo_id=%s AND path=%s AND ref=%s", (repo_id, path, ref))

    def insert_chunks(self, repo_id: int, repo_name: str, ref: str, chunks: Iterable[Any]) -> None:
        rows = [
            (
                ch.chunk_id,
                repo_id,
                repo_name,
                ref,
                ch.path,
                ch.language,
                ch.symbol,
                ch.start_line,
                ch.end_line,
                ch.content_hash,
                ch.text,
            )
            for ch in chunks
        ]
        if not rows:
            return
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO chunks(chunk_id, repo_id, repo_name, ref, path, language, symbol, start_line, end_line, content_hash, text)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (chunk_id) DO UPDATE SET text=excluded.text, content_hash=excluded.content_hash, start_line=excluded.start_line, end_line=excluded.end_line, language=excluded.language, symbol=excluded.symbol
                    """,
                    rows,
                )

    def search_keyword(self, repo: str, query: str, limit: int = 20) -> List[Any]:
        return self.fetchall(
            """
            SELECT c.chunk_id, r.owner, r.name, c.ref, c.path, c.start_line, c.end_line, c.text
            FROM chunks c
            JOIN repos r ON c.repo_id = r.id
            WHERE r.name = %s AND c.text ILIKE %s
            ORDER BY length(c.text) ASC
            LIMIT %s
            """,
            (repo, f"%{query}%", limit),
        )

    def get_chunk(self, chunk_id: str) -> Optional[Any]:
        return self.fetchone(
            """
            SELECT c.chunk_id, r.owner, c.repo_name, c.ref, c.path, c.start_line, c.end_line, c.text
            FROM chunks c
            JOIN repos r ON c.repo_id = r.id
            WHERE c.chunk_id = %s
            """,
            (chunk_id,),
        )

    def newest_pending_job(self) -> Optional[Any]:
        return self.fetchone(
            """
            SELECT job_id, owner, repo, ref FROM index_jobs
            WHERE status='PENDING'
            ORDER BY created_at ASC
            LIMIT 1
            """,
        )

    def update_job(self, job_id: str, status: str, message: str, progress: float) -> None:
        self.execute(
            "UPDATE index_jobs SET status=%s, message=%s, progress=%s, updated_at=now() WHERE job_id=%s",
            (status, message, progress, job_id),
        )
