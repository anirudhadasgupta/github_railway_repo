import fnmatch
import random
import time
import uuid
from typing import Iterable, List, Sequence

from openai import OpenAI
from qdrant_client.http import models as qm

from .chunking import CodeChunk, chunk_file
from .db import Database
from .github_client import GitHubClient
from .qdrant_store import QdrantStore


class Indexer:
    def __init__(
        self,
        github: GitHubClient,
        qdrant: QdrantStore,
        db: Database,
        openai_api_key: str,
        embedding_model: str,
        embedding_dims: int,
        collection: str = "gh_code",
    ):
        self.github = github
        self.qdrant = qdrant
        self.db = db
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.embedding_model = embedding_model
        self.embedding_dims = embedding_dims
        self.collection = collection

    def _filter_paths(self, entries: Sequence[dict], include_globs: List[str], exclude_globs: List[str]) -> List[dict]:
        files = [e for e in entries if e.get("type") == "blob"]

        if include_globs:
            files = [f for f in files if any(fnmatch.fnmatch(f.get("path", ""), pat) for pat in include_globs)]

        if exclude_globs:
            files = [f for f in files if not any(fnmatch.fnmatch(f.get("path", ""), pat) for pat in exclude_globs)]
        return files

    def _embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        attempt = 0
        while True:
            try:
                resp = self.openai_client.embeddings.create(
                    model=self.embedding_model,
                    input=list(texts),
                    dimensions=self.embedding_dims,
                )
                return [d.embedding for d in resp.data]
            except Exception:  # noqa: BLE001
                attempt += 1
                if attempt >= 5:
                    raise
                jitter = random.uniform(0.5, 1.5)
                time.sleep(min(2**attempt * jitter, 10))

    def _batch(self, items: Sequence[CodeChunk], size: int = 32) -> Iterable[List[CodeChunk]]:
        for i in range(0, len(items), size):
            yield list(items[i : i + size])

    def index_repo_ref(
        self,
        owner: str,
        repo: str,
        ref: str,
        max_bytes: int,
        include_globs: List[str],
        exclude_globs: List[str],
    ) -> None:
        meta = self.github.get_repo(owner, repo)
        default_branch = meta.get("default_branch")
        tree = self.github.get_tree_recursive(owner, repo, ref)
        repo_id = self.db.upsert_repo(owner, repo, default_branch)

        entries = tree.get("tree", [])
        files = self._filter_paths(entries, include_globs, exclude_globs)

        for entry in files:
            path = entry.get("path")
            size = int(entry.get("size") or 0)
            sha = entry.get("sha", "")

            if size > max_bytes:
                continue

            text = self.github.read_text_file(owner, repo, path, ref)
            chunks = chunk_file(repo=repo, ref=ref, path=path, text=text)

            self.db.upsert_file(repo_id=repo_id, path=path, ref=ref, blob_sha=sha, size_bytes=size)
            self.db.delete_chunks(repo_id=repo_id, path=path, ref=ref)
            self.qdrant.delete_path(self.collection, repo, ref, path)

            for batch in self._batch(chunks):
                embeddings = self._embed_texts([c.text for c in batch])
                points = []
                for emb, chunk in zip(embeddings, batch):
                    payload = {
                        "chunk_id": chunk.chunk_id,
                        "repo": repo,
                        "ref": ref,
                        "path": path,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "symbol": chunk.symbol,
                    }
                    points.append(
                        qm.PointStruct(
                            id=str(uuid.uuid4()),
                            vector=emb,
                            payload=payload,
                        )
                    )
                self.qdrant.upsert_points(self.collection, points)
                self.db.insert_chunks(repo_id=repo_id, repo_name=repo, ref=ref, chunks=batch)

    def semantic_search(self, repo: str, query: str, limit: int = 5) -> List[dict]:
        vector = self._embed_texts([query])[0]
        return self.qdrant.search(self.collection, vector=vector, repo=repo, limit=limit)
