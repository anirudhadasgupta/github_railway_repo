import logging
import random
import time
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

logger = logging.getLogger("mcp.qdrant")


class QdrantStore:
    def __init__(self, url: str, api_key: Optional[str], dims: int):
        self.client = QdrantClient(url=url, api_key=api_key)
        self.dims = dims

    def ensure_collection(self, name: str = "gh_code") -> None:
        """Ensure collection exists with retry/backoff for transient failures."""
        def _do_ensure():
            collections = self.client.get_collections().collections
            if any(c.name == name for c in collections):
                logger.info("qdrant_collection_exists name=%s", name)
                return
            self.client.create_collection(
                collection_name=name,
                vectors_config=qm.VectorParams(size=self.dims, distance=qm.Distance.COSINE),
            )
            logger.info("qdrant_collection_created name=%s dims=%d", name, self.dims)

        self._with_retry(_do_ensure, max_attempts=5)

    def upsert_points(self, collection: str, points: List[qm.PointStruct]) -> None:
        if not points:
            return
        self._with_retry(lambda: self.client.upsert(collection_name=collection, points=points))

    def delete_path(self, collection: str, repo: str, ref: str, path: str) -> None:
        self._with_retry(
            lambda: self.client.delete(
                collection_name=collection,
                points_selector=qm.FilterSelector(
                    qm.Filter(
                        must=[
                            qm.FieldCondition(key="repo", match=qm.MatchValue(value=repo)),
                            qm.FieldCondition(key="ref", match=qm.MatchValue(value=ref)),
                            qm.FieldCondition(key="path", match=qm.MatchValue(value=path)),
                        ]
                    )
                ),
            )
        )

    def search(self, collection: str, vector: List[float], repo: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        filters = None
        if repo:
            filters = qm.Filter(must=[qm.FieldCondition(key="repo", match=qm.MatchValue(value=repo))])
        res = self._with_retry(
            lambda: self.client.query_points(
                collection_name=collection,
                query=vector,
                query_filter=filters,
                limit=limit,
            )
        )
        out: List[Dict[str, Any]] = []
        for item in res.points:
            payload = item.payload or {}
            out.append(
                {
                    "chunk_id": payload.get("chunk_id"),
                    "score": item.score,
                    "repo": payload.get("repo"),
                    "ref": payload.get("ref"),
                    "path": payload.get("path"),
                    "start_line": payload.get("start_line"),
                    "end_line": payload.get("end_line"),
                    "symbol": payload.get("symbol"),
                }
            )
        return out

    def _with_retry(self, fn, max_attempts: int = 5):
        attempt = 0
        while True:
            try:
                return fn()
            except Exception:  # noqa: BLE001
                attempt += 1
                if attempt >= max_attempts:
                    raise
                jitter = random.uniform(0.5, 1.5)
                time.sleep(min(2**attempt * jitter, 10))
