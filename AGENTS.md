## What you’re building

A **remote MCP server** (hosted on **Railway**, reachable over **HTTPS**) that ChatGPT can connect to as a **custom connector**. It will:

* Read **private GitHub repos** for `anirudhadasgupta` using a **GitHub Personal Access Token (PAT)**
* Maintain a **persistent** code index (Postgres metadata + Qdrant vectors)
* Expose MCP tools for:

  * repo outline
  * keyword search
  * semantic search (embeddings)
  * file content
  * file history
  * file diff
  * commit / branch / tag history
  * “search + fetch” wrappers for **ChatGPT deep research compatibility**

Key platform constraints to design around:

* **ChatGPT requires HTTPS** for MCP servers.
* **ChatGPT supports remote MCP servers** and supports **SSE or Streamable HTTP**; Streamable HTTP is recommended, but SSE is supported.
* ChatGPT **may retry tool calls**, so handlers must be **idempotent**.
* Tool outputs are user-visible; **do not return secrets** (PAT/OpenAI key).
* For deep research connectors, OpenAI documents a **`search` + `fetch` tool contract** (content array with a JSON-encoded `text` payload).
* For embeddings, `text-embedding-3-large` is supported and you can shorten vectors via the `dimensions` parameter.

---

## Architecture (persistent, non-ephemeral)

**Services in Railway project:**

1. **mcp-server** (Python, FastMCP over SSE at `/sse/`)
2. **postgres** (Railway PostgreSQL service)
3. **qdrant** (Railway Qdrant template)
4. *(Optional)* **worker** (same codebase, runs indexing jobs continuously)

**Persistence:**

* Postgres stores:

  * indexed repos/refs
  * file metadata (path, sha, sizes)
  * chunk records (chunk_id, text, symbol boundaries, line ranges)
  * indexing job state (so jobs survive restarts)
* Qdrant stores:

  * vectors for chunks + payload metadata (repo/path/ref/lines/symbol)

**Railway storage rules:**

* Railway **Volumes** provide persistent data across redeploys.
* If you persist to a relative path, mount the volume under `/app/...` because Railway buildpacks place your code in `/app`.
* Qdrant should have a **volume mounted** to its storage directory to persist collections.

---

## Step-by-step implementation guide

### Step 0 — Create credentials

**GitHub PAT**

* Classic PAT: needs `repo` scope for private repositories.
* Fine-grained PAT: grant read access to:

  * Repository contents (read)
  * Metadata (read)
  * Commit history (read)
  * Pull requests (optional, only if you expand later)

**OpenAI API key**

* Required for embeddings (and optionally summarization).
* Use `text-embedding-3-large` (optionally shortened via `dimensions`).

---

### Step 1 — Create the Railway project

1. Create a new Railway project.
2. Add **PostgreSQL** (Railway database service).

   * Railway provides `DATABASE_URL` and standard PG vars.
3. Add **Qdrant** using Railway’s Qdrant template.

   * One-click deploy available.
4. Add a **Volume** to the Qdrant service and mount it to Qdrant’s storage directory (commonly `/qdrant/storage`).

> Without persistent storage, stateful services lose data between redeploys.

---

### Step 2 — Scaffold the repo

Recommended layout:

```
github-mcp/
  server/
    __init__.py
    main.py            # MCP server (SSE)
    config.py
    mcp_utils.py
    github_client.py
    db.py
    qdrant_store.py
    chunking.py
    indexer.py
    worker.py          # optional worker service entrypoint
  migrations/
    001_init.sql
  requirements.txt
  Dockerfile
```

Railway can build from a Dockerfile if present at repo root.

---

### Step 3 — Define environment variables (Railway Variables)

Set these in **mcp-server** and **worker** services:

* `GITHUB_OWNER=anirudhadasgupta`
* `GITHUB_PAT=...`  *(secret)*
* `OPENAI_API_KEY=...` *(secret)*
* `DATABASE_URL=...` (use Railway reference variable from Postgres)
* `QDRANT_URL=...` (internal URL to Qdrant service)
* `QDRANT_API_KEY=...` *(if your Qdrant template uses one)*
* `EMBEDDING_MODEL=text-embedding-3-large`
* `EMBEDDING_DIMS=1024` *(or 1536/3072/256 depending on cost/quality)*

---

### Step 4 — Create Postgres schema (metadata + jobs + chunks)

**migrations/001_init.sql**

```sql
-- Enable useful extensions for keyword search (optional but recommended)
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS repos (
  id              BIGSERIAL PRIMARY KEY,
  owner           TEXT NOT NULL,
  name            TEXT NOT NULL,
  default_branch  TEXT,
  indexed_at      TIMESTAMPTZ,
  UNIQUE(owner, name)
);

CREATE TABLE IF NOT EXISTS files (
  id              BIGSERIAL PRIMARY KEY,
  repo_id         BIGINT NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
  path            TEXT NOT NULL,
  ref             TEXT NOT NULL,
  blob_sha        TEXT,
  size_bytes      BIGINT,
  updated_at      TIMESTAMPTZ DEFAULT now(),
  UNIQUE(repo_id, path, ref)
);

CREATE TABLE IF NOT EXISTS chunks (
  chunk_id        TEXT PRIMARY KEY,
  repo_id         BIGINT NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
  repo_name       TEXT NOT NULL,
  ref             TEXT NOT NULL,
  path            TEXT NOT NULL,
  language        TEXT,
  symbol          TEXT,
  start_line      INT,
  end_line        INT,
  content_hash    TEXT NOT NULL,
  text            TEXT NOT NULL,
  created_at      TIMESTAMPTZ DEFAULT now()
);

-- Fast keyword search (optional; useful once chunk table grows)
CREATE INDEX IF NOT EXISTS chunks_text_trgm_idx ON chunks USING gin (text gin_trgm_ops);

CREATE TABLE IF NOT EXISTS index_jobs (
  job_id          TEXT PRIMARY KEY,
  owner           TEXT NOT NULL,
  repo            TEXT NOT NULL,
  ref             TEXT NOT NULL,
  status          TEXT NOT NULL, -- PENDING|RUNNING|DONE|ERROR
  progress        REAL NOT NULL DEFAULT 0,
  message         TEXT,
  created_at      TIMESTAMPTZ DEFAULT now(),
  updated_at      TIMESTAMPTZ DEFAULT now()
);
```

Run this once against Railway Postgres (via `psql` or a startup migration step).

---

### Step 5 — Implement the MCP server + tools (FastMCP over SSE)

OpenAI’s MCP guide demonstrates FastMCP running with **SSE** and notes the connector URL should end with `/sse/`.

**server/mcp_utils.py**

```python
import json
from typing import Any, Dict

def mcp_json(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    MCP tool result helper:
    returns exactly one 'text' content item containing JSON.
    """
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(data, ensure_ascii=False)
            }
        ]
    }
```

**server/config.py**

```python
import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    github_owner: str = os.getenv("GITHUB_OWNER", "anirudhadasgupta")
    github_pat: str = os.environ["GITHUB_PAT"]
    openai_api_key: str = os.environ["OPENAI_API_KEY"]

    database_url: str = os.environ["DATABASE_URL"]

    qdrant_url: str = os.environ["QDRANT_URL"]
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY")

    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    embedding_dims: int = int(os.getenv("EMBEDDING_DIMS", "1024"))

    # indexing defaults
    max_file_bytes: int = int(os.getenv("MAX_FILE_BYTES", "250000"))  # 250KB
    include_globs: str = os.getenv("INCLUDE_GLOBS", "")  # optional
    exclude_globs: str = os.getenv(
        "EXCLUDE_GLOBS",
        "node_modules/**,dist/**,build/**,.git/**,vendor/**,**/*.min.js,**/*.lock"
    )
```

**server/github_client.py**

```python
import base64
import time
from typing import Any, Dict, List, Optional
import httpx

class GitHubClient:
    def __init__(self, token: str):
        self._token = token
        self._client = httpx.Client(
            base_url="https://api.github.com",
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=30.0,
        )

    def _handle_rate_limit(self, resp: httpx.Response) -> None:
        if resp.status_code != 403:
            return
        remaining = resp.headers.get("X-RateLimit-Remaining")
        reset = resp.headers.get("X-RateLimit-Reset")
        if remaining == "0" and reset:
            sleep_s = max(0, int(reset) - int(time.time()) + 1)
            time.sleep(min(sleep_s, 30))  # keep bounded; worker can do longer

    def _get(self, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        resp = self._client.get(url, params=params)
        self._handle_rate_limit(resp)
        resp.raise_for_status()
        return resp.json()

    def list_repos(self) -> List[Dict[str, Any]]:
        # /user/repos lists private repos visible to PAT
        repos = self._get("/user/repos", params={"per_page": 100, "visibility": "all"})
        return repos

    def get_repo(self, owner: str, repo: str) -> Dict[str, Any]:
        return self._get(f"/repos/{owner}/{repo}")

    def get_branch_head_sha(self, owner: str, repo: str, ref: str) -> str:
        data = self._get(f"/repos/{owner}/{repo}/commits/{ref}")
        return data["sha"]

    def get_tree_recursive(self, owner: str, repo: str, ref: str) -> Dict[str, Any]:
        commit = self._get(f"/repos/{owner}/{repo}/commits/{ref}")
        tree_sha = commit["commit"]["tree"]["sha"]
        return self._get(f"/repos/{owner}/{repo}/git/trees/{tree_sha}", params={"recursive": "1"})

    def get_file_content(self, owner: str, repo: str, path: str, ref: str) -> Dict[str, Any]:
        return self._get(f"/repos/{owner}/{repo}/contents/{path}", params={"ref": ref})

    def read_text_file(self, owner: str, repo: str, path: str, ref: str) -> str:
        data = self.get_file_content(owner, repo, path, ref)
        if data.get("encoding") == "base64":
            raw = base64.b64decode(data["content"])
            return raw.decode("utf-8", errors="replace")
        raise ValueError(f"Unsupported encoding for {path}: {data.get('encoding')}")

    def list_commits(self, owner: str, repo: str, ref: str, path: Optional[str] = None, limit: int = 30) -> List[Dict[str, Any]]:
        params = {"sha": ref, "per_page": min(limit, 100)}
        if path:
            params["path"] = path
        return self._get(f"/repos/{owner}/{repo}/commits", params=params)

    def list_branches(self, owner: str, repo: str) -> List[Dict[str, Any]]:
        return self._get(f"/repos/{owner}/{repo}/branches", params={"per_page": 100})

    def list_tags(self, owner: str, repo: str) -> List[Dict[str, Any]]:
        return self._get(f"/repos/{owner}/{repo}/tags", params={"per_page": 100})

    def compare(self, owner: str, repo: str, base: str, head: str) -> Dict[str, Any]:
        return self._get(f"/repos/{owner}/{repo}/compare/{base}...{head}")
```

---

### Step 6 — Implement chunking (split by file + logical boundaries)

This includes:

* chunker
* function/class boundaries for Python
* fallback splitter for other files
* boilerplate dedupe

**server/chunking.py**

```python
import ast
import hashlib
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

@dataclass(frozen=True)
class CodeChunk:
    chunk_id: str
    repo: str
    ref: str
    path: str
    language: str
    symbol: Optional[str]
    start_line: int
    end_line: int
    content_hash: str
    text: str

_EXT_LANG = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".md": "markdown",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
}

_BOILERPLATE_HEADER_RE = re.compile(
    r"(?is)\A(?:\s*(?:#|//|/\*|\*)\s*)?(copyright|license|generated)\b.*?\n\n"
)

def detect_language(path: str) -> str:
    for ext, lang in _EXT_LANG.items():
        if path.endswith(ext):
            return lang
    return "text"

def strip_boilerplate_header(text: str) -> str:
    return re.sub(_BOILERPLATE_HEADER_RE, "", text)

def stable_hash(text: str) -> str:
    norm = re.sub(r"\s+", " ", text).strip().encode("utf-8", errors="ignore")
    return hashlib.sha256(norm).hexdigest()

def make_chunk_id(repo: str, ref: str, path: str, start: int, end: int, content_hash: str) -> str:
    raw = f"{repo}:{ref}:{path}:{start}:{end}:{content_hash}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()

def python_chunks(repo: str, ref: str, path: str, text: str) -> List[Tuple[Optional[str], int, int, str]]:
    """
    Returns list of (symbol, start_line, end_line, chunk_text).
    Uses AST boundaries for classes/functions. Falls back to whole file if parsing fails.
    """
    lines = text.splitlines()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return [(None, 1, len(lines), text)]

    chunks: List[Tuple[Optional[str], int, int, str]] = []

    # Optional: include module docstring + imports as a header chunk
    header_end = 1
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            header_end = getattr(node, "end_lineno", getattr(node, "lineno", header_end))
        else:
            break
    if header_end > 1:
        header_text = "\n".join(lines[:header_end])
        chunks.append((None, 1, header_end, header_text))

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = getattr(node, "lineno", 1)
            end = getattr(node, "end_lineno", start)
            sym = getattr(node, "name", None)
            snippet = "\n".join(lines[start - 1:end])
            chunks.append((sym, start, end, snippet))

    if not chunks:
        chunks = [(None, 1, len(lines), text)]
    return chunks

def fallback_line_chunks(repo: str, ref: str, path: str, text: str, max_lines: int = 200, overlap: int = 30) -> List[Tuple[Optional[str], int, int, str]]:
    lines = text.splitlines()
    out = []
    i = 0
    while i < len(lines):
        start = i + 1
        end = min(len(lines), i + max_lines)
        snippet = "\n".join(lines[i:end])
        out.append((None, start, end, snippet))
        if end == len(lines):
            break
        i = max(0, end - overlap)
    return out

def chunk_file(repo: str, ref: str, path: str, text: str) -> List[CodeChunk]:
    text = strip_boilerplate_header(text)
    language = detect_language(path)

    if language == "python":
        parts = python_chunks(repo, ref, path, text)
    else:
        parts = fallback_line_chunks(repo, ref, path, text)

    chunks: List[CodeChunk] = []
    seen_hashes = set()

    for (symbol, start, end, snippet) in parts:
        h = stable_hash(snippet)
        if h in seen_hashes:
            continue  # dedupe repeated blocks within the file
        seen_hashes.add(h)

        cid = make_chunk_id(repo, ref, path, start, end, h)
        chunks.append(CodeChunk(
            chunk_id=cid,
            repo=repo,
            ref=ref,
            path=path,
            language=language,
            symbol=symbol,
            start_line=start,
            end_line=end,
            content_hash=h,
            text=snippet
        ))
    return chunks
```

This satisfies:

* split by file
* split by logical boundaries (Python AST)
* inline docs are naturally included with the function/class block (docstring is inside the node)
* dedupe boilerplate and repeated blocks

You can extend this with Tree-sitter for JS/TS/Go/Rust later without changing the rest of the pipeline.

---

### Step 7 — Embeddings + Qdrant vector store

**server/qdrant_store.py**

```python
from typing import Any, Dict, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

class QdrantStore:
    def __init__(self, url: str, api_key: Optional[str], dims: int):
        self.client = QdrantClient(url=url, api_key=api_key)
        self.dims = dims

    def ensure_collection(self, name: str = "gh_code") -> None:
        collections = self.client.get_collections().collections
        if any(c.name == name for c in collections):
            return
        self.client.create_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(size=self.dims, distance=qm.Distance.COSINE),
        )

    def upsert_points(self, collection: str, points: List[qm.PointStruct]) -> None:
        self.client.upsert(collection_name=collection, points=points)

    def delete_by_filter(self, collection: str, flt: qm.Filter) -> None:
        self.client.delete(collection_name=collection, points_selector=qm.FilterSelector(filter=flt))

    def search(self, collection: str, query_vector: List[float], limit: int, flt: Optional[qm.Filter] = None):
        return self.client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=limit,
            query_filter=flt,
            with_payload=True
        )
```

**Embedding choice**: `text-embedding-3-large` is capable and supports shortening embeddings by passing `dimensions`.

---

### Step 8 — Indexing pipeline (GitHub → chunks → embeddings → Postgres + Qdrant)

**server/indexer.py**

```python
import json
from typing import Dict, List, Optional, Tuple
from openai import OpenAI
from qdrant_client.http import models as qm

from .chunking import chunk_file, CodeChunk
from .github_client import GitHubClient
from .qdrant_store import QdrantStore

class Indexer:
    def __init__(self, gh: GitHubClient, qdrant: QdrantStore, openai_api_key: str, embed_model: str, embed_dims: int):
        self.gh = gh
        self.qdrant = qdrant
        self.client = OpenAI(api_key=openai_api_key)
        self.embed_model = embed_model
        self.embed_dims = embed_dims
        self.collection = "gh_code"

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        resp = self.client.embeddings.create(
            model=self.embed_model,
            input=texts,
            dimensions=self.embed_dims,  # supported shortening 
        )
        return [d.embedding for d in resp.data]

    def index_repo_ref(self, owner: str, repo: str, ref: str, max_bytes: int, exclude_globs: List[str]) -> List[CodeChunk]:
        self.qdrant.ensure_collection(self.collection)

        tree = self.gh.get_tree_recursive(owner, repo, ref)
        entries = tree.get("tree", [])

        files = []
        for e in entries:
            if e.get("type") != "blob":
                continue
            path = e.get("path", "")
            size = e.get("size", 0) or 0
            if size > max_bytes:
                continue
            # simple excludes (glob engine can be added later)
            if any(seg in path for seg in ["node_modules/", "dist/", "build/", "vendor/", ".git/"]):
                continue
            files.append(path)

        all_chunks: List[CodeChunk] = []
        batch: List[CodeChunk] = []
        batch_texts: List[str] = []

        for path in files:
            text = self.gh.read_text_file(owner, repo, path, ref)
            chunks = chunk_file(repo=repo, ref=ref, path=path, text=text)
            for ch in chunks:
                all_chunks.append(ch)
                batch.append(ch)
                batch_texts.append(ch.text)

                if len(batch) >= 64:
                    self._flush(batch, batch_texts, owner, repo)
                    batch, batch_texts = [], []

        if batch:
            self._flush(batch, batch_texts, owner, repo)

        return all_chunks

    def _flush(self, chunks: List[CodeChunk], texts: List[str], owner: str, repo: str) -> None:
        vecs = self.embed_texts(texts)
        points: List[qm.PointStruct] = []
        for ch, v in zip(chunks, vecs):
            payload = {
                "owner": owner,
                "repo": repo,
                "ref": ch.ref,
                "path": ch.path,
                "language": ch.language,
                "symbol": ch.symbol,
                "start_line": ch.start_line,
                "end_line": ch.end_line,
                "chunk_id": ch.chunk_id,
                "content_hash": ch.content_hash,
            }
            points.append(qm.PointStruct(id=ch.chunk_id, vector=v, payload=payload))
        self.qdrant.upsert_points(self.collection, points)
```

This is the “consume code → embed → vectorize” core. Persisting chunk text + job state is done in the DB layer (next step).

---

### Step 9 — Worker that makes indexing non-ephemeral + resumable

Instead of indexing inside an MCP tool call (which can time out), the MCP tool *creates a job*, and a worker processes it.

This is how you get:

* persistence across restarts
* retries
* safe long-running indexing

**server/worker.py** (simplified polling loop)

```python
import os
import time
import uuid
import psycopg

from .config import Settings
from .github_client import GitHubClient
from .qdrant_store import QdrantStore
from .indexer import Indexer

def main():
    s = Settings()
    gh = GitHubClient(s.github_pat)
    qdrant = QdrantStore(s.qdrant_url, s.qdrant_api_key, s.embedding_dims)
    indexer = Indexer(gh, qdrant, s.openai_api_key, s.embedding_model, s.embedding_dims)

    conn = psycopg.connect(s.database_url)
    conn.autocommit = True

    while True:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT job_id, owner, repo, ref FROM index_jobs WHERE status='PENDING' ORDER BY created_at ASC LIMIT 1"
            )
            row = cur.fetchone()

        if not row:
            time.sleep(2)
            continue

        job_id, owner, repo, ref = row

        try:
            with conn.cursor() as cur:
                cur.execute("UPDATE index_jobs SET status='RUNNING', message=%s, updated_at=now() WHERE job_id=%s",
                            ("Indexing started", job_id))

            indexer.index_repo_ref(
                owner=owner,
                repo=repo,
                ref=ref,
                max_bytes=s.max_file_bytes,
                exclude_globs=s.exclude_globs.split(",") if s.exclude_globs else [],
            )

            with conn.cursor() as cur:
                cur.execute("UPDATE index_jobs SET status='DONE', progress=1, message=%s, updated_at=now() WHERE job_id=%s",
                            ("Indexing complete", job_id))
        except Exception as e:
            with conn.cursor() as cur:
                cur.execute("UPDATE index_jobs SET status='ERROR', message=%s, updated_at=now() WHERE job_id=%s",
                            (str(e)[:5000], job_id))

if __name__ == "__main__":
    main()
```

---

### Step 10 — MCP tools (repo outline, search, fetch, history, diff)

**server/main.py**

```python
import os
import uuid
import psycopg

from fastmcp import FastMCP

from .config import Settings
from .github_client import GitHubClient
from .qdrant_store import QdrantStore
from .mcp_utils import mcp_json
from .indexer import Indexer

def build_server() -> FastMCP:
    s = Settings()

    gh = GitHubClient(s.github_pat)
    qdrant = QdrantStore(s.qdrant_url, s.qdrant_api_key, s.embedding_dims)
    indexer = Indexer(gh, qdrant, s.openai_api_key, s.embedding_model, s.embedding_dims)

    db = psycopg.connect(s.database_url)
    db.autocommit = True

    instructions = """
    You are connected to a GitHub code intelligence MCP.
    Prefer semantic_search + get_file_content for answering questions.
    Use keyword_search for exact string matches.
    Use file_history and file_diff when reasoning about changes.
    """

    mcp = FastMCP(name="github-private-repo-mcp", instructions=instructions)

    @mcp.tool()
    def gh_list_repos() -> dict:
        repos = gh.list_repos()
        # Return minimal fields to keep payload small
        return mcp_json({
            "repos": [
                {
                    "owner": r["owner"]["login"],
                    "name": r["name"],
                    "private": r.get("private", False),
                    "default_branch": r.get("default_branch"),
                    "url": r.get("html_url"),
                }
                for r in repos
            ]
        })

    @mcp.tool()
    def gh_repo_outline(repo: str, ref: str = "HEAD", max_entries: int = 2000) -> dict:
        owner = s.github_owner
        if ref == "HEAD":
            meta = gh.get_repo(owner, repo)
            ref = meta.get("default_branch") or "main"
        tree = gh.get_tree_recursive(owner, repo, ref)
        entries = tree.get("tree", [])[:max_entries]
        outline = [{"path": e["path"], "type": e["type"], "size": e.get("size")} for e in entries]
        return mcp_json({"owner": owner, "repo": repo, "ref": ref, "entries": outline})

    @mcp.tool()
    def gh_get_file_content(repo: str, path: str, ref: str = "HEAD", start_line: int = 1, end_line: int = 400) -> dict:
        owner = s.github_owner
        if ref == "HEAD":
            meta = gh.get_repo(owner, repo)
            ref = meta.get("default_branch") or "main"
        text = gh.read_text_file(owner, repo, path, ref)
        lines = text.splitlines()
        sl = max(1, start_line)
        el = min(len(lines), end_line)
        snippet = "\n".join(lines[sl - 1:el])
        url = f"https://github.com/{owner}/{repo}/blob/{ref}/{path}"
        return mcp_json({
            "owner": owner, "repo": repo, "ref": ref, "path": path,
            "start_line": sl, "end_line": el,
            "url": url,
            "text": snippet
        })

    @mcp.tool()
    def gh_file_history(repo: str, path: str, ref: str = "HEAD", limit: int = 20) -> dict:
        owner = s.github_owner
        if ref == "HEAD":
            meta = gh.get_repo(owner, repo)
            ref = meta.get("default_branch") or "main"
        commits = gh.list_commits(owner, repo, ref=ref, path=path, limit=limit)
        return mcp_json({
            "owner": owner, "repo": repo, "ref": ref, "path": path,
            "commits": [
                {"sha": c["sha"], "author": (c.get("commit", {}).get("author", {}) or {}).get("name"),
                 "date": (c.get("commit", {}).get("author", {}) or {}).get("date"),
                 "message": (c.get("commit", {}) or {}).get("message", "").splitlines()[0]}
                for c in commits
            ]
        })

    @mcp.tool()
    def gh_file_diff(repo: str, base: str, head: str) -> dict:
        owner = s.github_owner
        comp = gh.compare(owner, repo, base=base, head=head)
        files = comp.get("files", []) or []
        return mcp_json({
            "owner": owner, "repo": repo, "base": base, "head": head,
            "ahead_by": comp.get("ahead_by"),
            "behind_by": comp.get("behind_by"),
            "files": [
                {"filename": f.get("filename"), "status": f.get("status"), "additions": f.get("additions"),
                 "deletions": f.get("deletions"), "changes": f.get("changes"), "patch": f.get("patch")}
                for f in files
            ]
        })

    @mcp.tool()
    def gh_commit_history(repo: str, ref: str = "HEAD", limit: int = 30) -> dict:
        owner = s.github_owner
        if ref == "HEAD":
            meta = gh.get_repo(owner, repo)
            ref = meta.get("default_branch") or "main"
        commits = gh.list_commits(owner, repo, ref=ref, path=None, limit=limit)
        return mcp_json({
            "owner": owner, "repo": repo, "ref": ref,
            "commits": [{"sha": c["sha"], "message": c["commit"]["message"].splitlines()[0]} for c in commits]
        })

    @mcp.tool()
    def gh_branch_history(repo: str) -> dict:
        owner = s.github_owner
        branches = gh.list_branches(owner, repo)
        return mcp_json({
            "owner": owner, "repo": repo,
            "branches": [{"name": b["name"], "sha": b["commit"]["sha"]} for b in branches]
        })

    @mcp.tool()
    def gh_tag_history(repo: str) -> dict:
        owner = s.github_owner
        tags = gh.list_tags(owner, repo)
        return mcp_json({
            "owner": owner, "repo": repo,
            "tags": [{"name": t["name"], "sha": t["commit"]["sha"]} for t in tags]
        })

    @mcp.tool()
    def gh_index_repo(repo: str, ref: str = "HEAD") -> dict:
        owner = s.github_owner
        if ref == "HEAD":
            meta = gh.get_repo(owner, repo)
            ref = meta.get("default_branch") or "main"
        job_id = str(uuid.uuid4())
        with db.cursor() as cur:
            cur.execute(
                "INSERT INTO index_jobs(job_id, owner, repo, ref, status, progress, message) VALUES (%s,%s,%s,%s,'PENDING',0,'Queued')",
                (job_id, owner, repo, ref),
            )
        return mcp_json({"job_id": job_id, "status": "PENDING", "owner": owner, "repo": repo, "ref": ref})

    @mcp.tool()
    def gh_index_status(job_id: str) -> dict:
        with db.cursor() as cur:
            cur.execute("SELECT status, progress, message, owner, repo, ref FROM index_jobs WHERE job_id=%s", (job_id,))
            row = cur.fetchone()
        if not row:
            return mcp_json({"job_id": job_id, "found": False})
        status, progress, message, owner, repo, ref = row
        return mcp_json({"job_id": job_id, "found": True, "status": status, "progress": progress, "message": message,
                         "owner": owner, "repo": repo, "ref": ref})

    # ---- Deep research compatibility wrappers (search + fetch) ----
    # OpenAI’s MCP guide documents this contract. 

    @mcp.tool()
    def search(query: str) -> dict:
        # Minimal placeholder: you can route this to semantic search across indexed chunks.
        # For brevity, we return empty results if not yet implemented.
        return {
            "content": [{"type": "text", "text": "{\"results\": []}"}]
        }

    @mcp.tool()
    def fetch(id: str) -> dict:
        # In a full implementation, 'id' maps to a chunk_id in Postgres.
        # Return the doc object JSON as text.
        return {
            "content": [{"type": "text", "text": "{\"id\":\"" + id + "\",\"title\":\"\",\"text\":\"\",\"url\":\"\"}"}]
        }

    return mcp

def main():
    s = Settings()
    port = int(os.getenv("PORT", "8000"))
    server = build_server()

    # SSE is explicitly supported; OpenAI’s guide shows SSE and expects /sse/ URLs. 
    server.run(transport="sse", host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
```

Notes:

* The `search`/`fetch` wrappers above are stubs. In your production build, implement:

  * `search(query)` → semantic search via Qdrant, return result list as JSON text
  * `fetch(id)` → read chunk text from Postgres and return as JSON text
* If you primarily want **developer mode tool calling** (not deep research), you can ignore `search/fetch` entirely; ChatGPT no longer requires them for MCP connectors.
  But adding them helps with deep research workflows.

---

### Step 11 — requirements.txt

```
fastmcp
httpx
openai
psycopg[binary]
qdrant-client
```

Add chunking enhancements later:

* `tree_sitter` + language packs
* `tiktoken` for token-aware chunk sizing
* `tenacity` for retries/backoff

---

### Step 12 — Dockerfile (Railway deploy)

Railway auto-detects a root `Dockerfile`.

**Dockerfile**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "server.main"]
```

---

### Step 13 — Railway deployment setup

1. Push repo to GitHub.
2. In Railway: **Deploy from GitHub repo** (similar flow as their FastAPI guide).
3. Create two services from the same repo:

   * `mcp-server`: start command runs MCP server (Docker CMD above)
   * `worker` (optional): override CMD / start command to `python -m server.worker`

---

## Connecting from ChatGPT

1. Ensure you’re on a plan that supports custom connectors (Plus/Pro or workspace plans).
2. Enable **Developer mode** for connectors (OpenAI describes how to enable it in settings).
3. Add the connector URL:

   * If using SSE transport: your connector URL should be your Railway domain + `/sse/` (OpenAI’s example emphasizes `/sse/`).

---

## Hardening for production quality

### 1) Reliability and retries

* Implement retries with jitter for:

  * GitHub 5xx
  * GitHub rate limiting (respect `X-RateLimit-Reset`)
  * OpenAI 429/5xx
  * Qdrant transient errors
* Make indexing idempotent:

  * deterministic chunk IDs
  * delete existing points for (repo, ref, path) before re-upsert

### 2) Security (non-negotiable for private repos)

* Never put `GITHUB_PAT` or `OPENAI_API_KEY` in any tool output.
* Add an allowlist:

  * simplest: only accept requests if a shared secret is present (but ChatGPT may not support custom headers depending on connector mode)
  * best: OAuth 2.1 + dynamic client registration is what MCP supports conceptually
* Treat repo content as untrusted: prompt injection is a known risk for tool-using agents.

### 3) Performance

* Exclude:

  * large binaries, vendored deps, build artifacts, minified files
* Batch embeddings and Qdrant upserts
* Use `EMBEDDING_DIMS` to trade off cost/quality.

### 4) UX for ChatGPT tool use

* Return small structured payloads:

  * summaries + pointers (path + line ranges + commit sha)
* Provide stable IDs so follow-up calls work cleanly
* Keep tools single-intent, and document in the tool description (ChatGPT chooses tools based on tool metadata).

---

## Final checklist (build + deploy + validate)

### Build

* [ ] Tools implemented: outline, keyword search, semantic search, file content, file history, diff, commit/branch/tag history
* [ ] Index jobs table + worker loop (or queue) implemented
* [ ] Qdrant collection created on startup
* [ ] Chunk IDs deterministic and deduplication applied

### Deploy (Railway)

* [ ] Postgres service added; `DATABASE_URL` wired
* [ ] Qdrant template deployed
* [ ] Qdrant has a Volume mounted (persistent storage)
* [ ] Dockerfile present and detected
* [ ] Environment variables configured (PAT/OpenAI key as secrets)

### ChatGPT integration

* [ ] Developer mode enabled for connectors
* [ ] Connector URL uses HTTPS
* [ ] If SSE: URL ends with `/sse/`
* [ ] Run a test flow:

  1. `gh_index_repo(repo)`
  2. poll `gh_index_status(job_id)` until DONE
  3. `gh_repo_outline(repo)`
  4. `gh_get_file_content(repo, path)`
  5. `gh_file_history(repo, path)`

---

## Premortem: what fails first and how to avoid it

1. **Indexing times out / gets killed**
   Fix: move all indexing to worker jobs (already in design). Persist job state in Postgres.

2. **Rate limits from GitHub**
   Fix: incremental indexing (only changed files), respect `X-RateLimit-Reset`, cache trees, reduce file set.

3. **Qdrant loses data on redeploy**
   Fix: volume mount is mandatory for Qdrant persistence.

4. **Tool outputs too large for ChatGPT to use effectively**
   Fix: return pointers (path + lines) and short snippets; provide follow-up tools to fetch exact regions.

5. **Security leak: someone hits your MCP URL and uses your PAT**
   Fix: implement OAuth or at minimum strict allowlisting; never expose PAT in any output.

---

