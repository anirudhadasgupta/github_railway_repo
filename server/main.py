import asyncio
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP
from starlette.responses import JSONResponse, PlainTextResponse, StreamingResponse

from .config import require_core_secrets, settings
from .db import Database
from .github_client import GitHubClient
from .indexer import Indexer
from .mcp_utils import mcp_json
from .qdrant_store import QdrantStore

# Structured logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}',
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("mcp.server")

# Instance diagnostics - proves restart vs re-register
INSTANCE_ID = str(uuid.uuid4())
INSTANCE_START_TIME = time.time()

# HTTP path discovery helpers (use env overrides when provided)
STREAMABLE_HTTP_PATH = os.getenv("FASTMCP_STREAMABLE_HTTP_PATH", "/mcp")
SSE_PATH = os.getenv("FASTMCP_SSE_PATH", "/sse")
MESSAGE_PATH = os.getenv("FASTMCP_MESSAGE_PATH", "/messages/")

# Tokenless design: no session management required
# Clients just call help(action="reconnect") on errors


def _default_ref(gh: GitHubClient, owner: str, repo: str, ref: str) -> str:
    if ref != "HEAD":
        return ref
    meta = gh.get_repo(owner, repo)
    return meta.get("default_branch") or "main"


def _build_discovery(base_url: Optional[str] = None, root_path: str = "") -> dict:
    """Return current transport entrypoints for recovery/discovery flows."""
    transport = os.getenv("MCP_TRANSPORT", "http")
    prefix = root_path.rstrip("/")

    def full_path(path: str) -> str:
        if base_url:
            return f"{base_url.rstrip('/')}{prefix}{path}"
        return f"{prefix}{path}" if prefix else path

    return {
        "transport": transport,
        "root_path": prefix or "/",
        "streamable_http": full_path(STREAMABLE_HTTP_PATH),
        "sse": full_path(SSE_PATH),
        "messages": full_path(MESSAGE_PATH),
    }


def _base_capabilities(discovery: dict) -> dict:
    """
    Minimal server manifest used by both HTTP routes and MCP tools.
    Tokenless design - no session management required.
    """
    return {
        "name": "github-private-repo-mcp",
        "instance_id": INSTANCE_ID,
        "uptime_seconds": round(time.time() - INSTANCE_START_TIME, 2),
        "discovery": discovery,
        "recovery": "On error: help(reconnect) → tools/list → retry",
    }


def build_server() -> FastMCP:
    logger.info("server_startup instance_id=%s", INSTANCE_ID)
    require_core_secrets()

    logger.info("initializing_github_client")
    gh = GitHubClient(settings.github_pat)

    logger.info("initializing_qdrant url=%s", settings.qdrant_url)
    qdrant = QdrantStore(settings.qdrant_url, settings.qdrant_api_key, settings.embedding_dims)
    qdrant.ensure_collection()

    logger.info("initializing_database")
    db = Database(settings.database_url)
    indexer = Indexer(
        github=gh,
        qdrant=qdrant,
        db=db,
        openai_api_key=settings.openai_api_key,
        embedding_model=settings.embedding_model,
        embedding_dims=settings.embedding_dims,
    )

    # Minimal recovery playbook - short, mechanical, unambiguous
    instructions = """
    GitHub code intelligence MCP. 5 tools: help, search, fetch, explore, index.

    ERROR RECOVERY (on any "Resource not found", tool missing, 404, 502, 424):
      1. help(action="reconnect")
      2. tools/list (rediscover tools)
      3. retry once
    """

    # Enable stateless_http to handle OpenAI's DELETE requests that terminate sessions
    # Session persistence is handled at application level via _session_cache
    mcp = FastMCP(name="github-private-repo-mcp", instructions=instructions, stateless_http=True)

    # Tool annotations for OpenAI/ChatGPT compatibility
    # readOnlyHint: true = skips "Continue" confirmation prompt in ChatGPT
    # openWorldHint: indicates interaction with external data sources
    # destructiveHint: false = not a destructive operation (default is true)
    # idempotentHint: true = repeated calls produce same result
    tool_opts = {"annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True}}

    # Search tool annotations per OpenAI specifications
    # openWorldHint: indicates the tool searches external/dynamic data
    search_tool_opts = {"annotations": {"readOnlyHint": True, "openWorldHint": True, "destructiveHint": False}}

    # Fetch tool annotations - retrieves specific content by ID
    fetch_tool_opts = {"annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True}}

    # Write tool annotations - for indexing operations
    write_tool_opts = {"annotations": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True}}

    @mcp.custom_route("/health", methods=["GET"])
    async def health_check(request):
        return JSONResponse({
            "status": "ok",
            "instance_id": INSTANCE_ID,
            "uptime_seconds": round(time.time() - INSTANCE_START_TIME, 2),
        })

    @mcp.custom_route("/", methods=["GET"])
    async def root_route(request):
        """Lightweight landing route with discovery pointers."""
        discovery = _build_discovery(
            base_url=str(request.base_url), root_path=request.scope.get("root_path", "")
        )
        caps = _base_capabilities(discovery)
        base = str(request.base_url).rstrip("/")
        root_path = request.scope.get("root_path", "").rstrip("/")

        def link(path: str) -> str:
            return f"{base}{root_path}{path}" if root_path else f"{base}{path}"

        return JSONResponse({
            "status": "ok",
            "message": "GitHub MCP server is ready.",
            "links": {
                "health": link("/health"),
                "help": link("/.well-known/mcp/help"),
            },
            "capabilities": caps,
        })

    help_payload = {
        "title": "MCP Recovery Playbook",
        "tools": ["help", "search", "fetch", "explore", "index"],
        "recovery": {
            "trigger": "Resource not found | tool missing | 404 | 502 | 424",
            "steps": [
                "1. help(action='reconnect')",
                "2. tools/list (rediscover)",
                "3. retry once",
            ],
        },
    }

    @mcp.custom_route("/help", methods=["GET"])
    async def help_route(request):
        """Lightweight troubleshooting guide for tool retries."""
        discovery = _build_discovery(
            base_url=str(request.base_url), root_path=request.scope.get("root_path", "")
        )
        caps = _base_capabilities(discovery)
        return JSONResponse({**help_payload, "capabilities": caps})

    # Stable, conventional alias that survives client-side wrapper rotation because it
    # is served directly by your Railway domain.
    @mcp.custom_route("/.well-known/mcp/help", methods=["GET"])
    async def help_well_known_route(request):
        discovery = _build_discovery(
            base_url=str(request.base_url), root_path=request.scope.get("root_path", "")
        )
        caps = _base_capabilities(discovery)
        return JSONResponse({**help_payload, "capabilities": caps})

    @mcp.custom_route("/capabilities", methods=["GET"])
    async def capabilities_route(request):
        discovery = _build_discovery(
            base_url=str(request.base_url), root_path=request.scope.get("root_path", "")
        )
        return JSONResponse(_base_capabilities(discovery))

    # SSE fallback endpoint - guides client back to HTTP transport
    @mcp.custom_route("/sse", methods=["GET"])
    async def sse_fallback_route(request):
        """SSE fallback - returns recovery guidance."""
        logger.info("sse_fallback_triggered")

        async def sse_reconnect_stream():
            import json
            yield f"event: message\ndata: {json.dumps({'recovery': 'help(reconnect) → tools/list → retry'})}\n\n"

        return StreamingResponse(
            sse_reconnect_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    # Handle /mcp GET requests (SSE fallback)
    @mcp.custom_route("/mcp", methods=["GET"])
    async def mcp_sse_fallback_route(request):
        """GET fallback for /mcp - returns recovery guidance."""
        accept_header = request.headers.get("accept", "")
        if "text/event-stream" in accept_header:
            logger.info("mcp_sse_fallback_triggered")

            async def sse_reconnect_stream():
                import json
                yield f"event: message\ndata: {json.dumps({'recovery': 'help(reconnect) → tools/list → retry'})}\n\n"

            return StreamingResponse(
                sse_reconnect_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        # Non-SSE GET: return recovery info
        return JSONResponse({"recovery": "help(reconnect) → tools/list → retry"})

    # ---- MCP Resources ----
    # Resources provide discoverable data sources that clients can browse

    @mcp.resource("github://repos")
    def resource_repos() -> str:
        """List of accessible GitHub repositories."""
        repos = gh.list_repos()
        repo_list = [
            {"owner": r["owner"]["login"], "name": r["name"], "private": r.get("private", False)}
            for r in repos
        ]
        import json
        return json.dumps({"repos": repo_list}, ensure_ascii=False)

    @mcp.resource("github://indexed")
    def resource_indexed_repos() -> str:
        """List of repositories that have been indexed for semantic search."""
        rows = db.fetchall(
            "SELECT DISTINCT owner, name FROM repos WHERE indexed_at IS NOT NULL ORDER BY name"
        )
        import json
        return json.dumps({"indexed_repos": [{"owner": r["owner"], "name": r["name"]} for r in rows]}, ensure_ascii=False)

    @mcp.resource("github://status")
    def resource_status() -> str:
        """Server status and recovery info."""
        import json
        return json.dumps({
            "instance_id": INSTANCE_ID,
            "uptime_seconds": round(time.time() - INSTANCE_START_TIME, 2),
            "recovery": "help(reconnect) → tools/list → retry",
        }, ensure_ascii=False)

    # ============================================================================
    # CONSOLIDATED MCP TOOLS (5 total)
    # Reduced from 19 tools to improve ChatGPT compatibility and reduce URI rotation issues
    # ============================================================================

    # TOOL 1: help - Connection management and recovery playbook (tokenless)
    @mcp.tool(**tool_opts)
    def help(action: str = "status") -> dict:
        """
        Connection management and recovery playbook.

        Actions:
        - "status": Server status (default)
        - "reconnect": Re-establish connection after errors
        - "playbook": Get recovery playbook

        After any error (404, 502, 424, "Resource not found"):
          1. help(action="reconnect")
          2. tools/list (rediscover)
          3. retry once
        """
        logger.info("tool_call: help action=%s", action)
        now = time.time()
        discovery = _build_discovery(root_path=os.getenv("ROOT_PATH", ""))

        if action == "playbook":
            return mcp_json({**help_payload, "discovery": discovery})

        if action == "reconnect":
            return mcp_json({
                "status": "reconnected",
                "next": "tools/list then retry",
                "discovery": discovery,
            })

        # Default: status
        return mcp_json({
            "status": "ok",
            "instance_id": INSTANCE_ID,
            "uptime_seconds": round(now - INSTANCE_START_TIME, 2),
            "discovery": discovery,
        })

    # TOOL 2: search - Unified search (semantic + keyword modes)
    @mcp.tool(**search_tool_opts)
    def search(
        query: str,
        repo: Optional[str] = None,
        mode: str = "semantic",
        limit: int = 10
    ) -> dict:
        """
        Unified search tool for code discovery. Supports semantic and keyword search modes.

        Args:
            query: Search query (natural language for semantic, exact string for keyword)
            repo: Repository name to search in (required for keyword mode)
            mode: "semantic" (default) for natural language, "keyword" for exact matches
            limit: Maximum results to return (default 10)

        Returns results with chunk IDs that can be fetched with the fetch tool.
        For deep research compatibility, results include id, title, text, and url fields.
        """
        logger.info("tool_call: search query=%s repo=%s mode=%s", query, repo, mode)

        if mode == "keyword":
            if not repo:
                return mcp_json({"error": "repo is required for keyword search", "results": []})
            results = db.search_keyword(repo, query, limit=limit)
            formatted = [
                {
                    "id": r["chunk_id"],
                    "title": f"{r['path']}:{r['start_line']}-{r['end_line']}",
                    "text": r["text"],
                    "url": f"https://github.com/{r['owner']}/{r['name']}/blob/{r['ref']}/{r['path']}#L{r['start_line']}-L{r['end_line']}",
                    "path": r["path"],
                    "start_line": r["start_line"],
                    "end_line": r["end_line"],
                }
                for r in results
            ]
            return mcp_json({"query": query, "mode": mode, "results": formatted})

        # Semantic search (default)
        if not repo:
            repo = ""
        matches = indexer.semantic_search(repo, query, limit=limit) if repo else []
        results = []
        for m in matches:
            chunk = db.get_chunk(m.get("chunk_id")) if m.get("chunk_id") else None
            if not chunk:
                continue
            repo_name = chunk["repo_name"]
            results.append({
                "id": chunk["chunk_id"],
                "title": f"{chunk['path']}:{chunk['start_line']}-{chunk['end_line']}",
                "text": chunk["text"],
                "url": f"https://github.com/{settings.github_owner}/{repo_name}/blob/{chunk['ref']}/{chunk['path']}#L{chunk['start_line']}-L{chunk['end_line']}",
                "score": m.get("score"),
                "symbol": m.get("symbol"),
                "path": chunk["path"],
                "start_line": chunk["start_line"],
                "end_line": chunk["end_line"],
            })
        return mcp_json({"query": query, "mode": mode, "results": results})

    # TOOL 3: fetch - Get file content, diff, or chunk by ID
    @mcp.tool(**fetch_tool_opts)
    def fetch(
        target: str,
        repo: Optional[str] = None,
        ref: str = "HEAD",
        start_line: int = 1,
        end_line: int = 400,
        base: Optional[str] = None
    ) -> dict:
        """
        Fetch content from GitHub. Can retrieve file content, diffs, or indexed chunks.

        Args:
            target: File path, chunk ID (from search), or "diff" for comparing refs
            repo: Repository name (required for file/diff, optional for chunk ID)
            ref: Git ref (branch/tag/commit) for file content (default "HEAD")
            start_line: Starting line for file content (default 1)
            end_line: Ending line for file content (default 400)
            base: Base ref for diff comparison (required when target="diff")

        Examples:
            - fetch(target="src/main.py", repo="myrepo") - get file content
            - fetch(target="abc123...", repo=None) - get chunk by ID from search results
            - fetch(target="diff", repo="myrepo", base="main", ref="feature") - compare refs
        """
        logger.info("tool_call: fetch target=%s repo=%s ref=%s", target, repo, ref)
        owner = settings.github_owner

        # Check if target is a chunk ID (40-char hex)
        if len(target) == 40 and all(c in "0123456789abcdef" for c in target.lower()):
            chunk = db.get_chunk(target)
            if chunk:
                return mcp_json({
                    "type": "chunk",
                    "id": chunk["chunk_id"],
                    "title": f"{chunk['path']}:{chunk['start_line']}-{chunk['end_line']}",
                    "text": chunk["text"],
                    "url": f"https://github.com/{owner}/{chunk['repo_name']}/blob/{chunk['ref']}/{chunk['path']}#L{chunk['start_line']}-L{chunk['end_line']}",
                    "path": chunk["path"],
                    "ref": chunk["ref"],
                    "start_line": chunk["start_line"],
                    "end_line": chunk["end_line"],
                })
            # Fall through to try as file path if chunk not found

        if not repo:
            return mcp_json({"error": "repo is required for file/diff fetch", "found": False})

        # Diff mode
        if target == "diff" and base:
            comp = gh.compare(owner, repo, base=base, head=ref)
            files = comp.get("files", []) or []
            return mcp_json({
                "type": "diff",
                "owner": owner,
                "repo": repo,
                "base": base,
                "head": ref,
                "ahead_by": comp.get("ahead_by"),
                "behind_by": comp.get("behind_by"),
                "files": [
                    {
                        "filename": f.get("filename"),
                        "status": f.get("status"),
                        "additions": f.get("additions"),
                        "deletions": f.get("deletions"),
                        "changes": f.get("changes"),
                        "patch": f.get("patch"),
                    }
                    for f in files
                ],
            })

        # File content mode (default)
        resolved_ref = _default_ref(gh, owner, repo, ref)
        text = gh.read_text_file(owner, repo, target, resolved_ref)
        lines = text.splitlines()
        sl = max(1, start_line)
        el = min(len(lines), end_line)
        snippet = "\n".join(lines[sl - 1 : el])
        url = f"https://github.com/{owner}/{repo}/blob/{resolved_ref}/{target}"
        return mcp_json({
            "type": "file",
            "owner": owner,
            "repo": repo,
            "ref": resolved_ref,
            "path": target,
            "start_line": sl,
            "end_line": el,
            "total_lines": len(lines),
            "url": url,
            "text": snippet,
        })

    # TOOL 4: explore - List repos, repo outline, history (commits/branches/tags/files)
    @mcp.tool(**tool_opts)
    def explore(
        action: str = "repos",
        repo: Optional[str] = None,
        ref: str = "HEAD",
        path: Optional[str] = None,
        limit: int = 30
    ) -> dict:
        """
        Explore GitHub repositories and their history.

        Args:
            action: What to explore:
                - "repos": List all accessible repositories (default)
                - "outline": Get repository file tree structure
                - "commits": Get commit history
                - "branches": List all branches
                - "tags": List all tags
                - "file_history": Get history of a specific file (requires path)
            repo: Repository name (required for all except "repos")
            ref: Git ref for outline/commits/file_history (default "HEAD")
            path: File path for "file_history" action
            limit: Maximum entries to return (default 30, max 2000 for outline)

        Examples:
            - explore() - list all repos
            - explore(action="outline", repo="myrepo") - get file tree
            - explore(action="commits", repo="myrepo", limit=50) - recent commits
            - explore(action="file_history", repo="myrepo", path="src/main.py") - file history
        """
        logger.info("tool_call: explore action=%s repo=%s", action, repo)
        owner = settings.github_owner

        if action == "repos":
            repos = gh.list_repos()
            return mcp_json({
                "action": "repos",
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

        if not repo:
            return mcp_json({"error": "repo is required for this action", "action": action})

        resolved_ref = _default_ref(gh, owner, repo, ref)

        if action == "outline":
            tree = gh.get_tree_recursive(owner, repo, resolved_ref)
            entries = tree.get("tree", [])[:min(limit, 2000)]
            return mcp_json({
                "action": "outline",
                "owner": owner,
                "repo": repo,
                "ref": resolved_ref,
                "entries": [{"path": e["path"], "type": e["type"], "size": e.get("size")} for e in entries],
            })

        if action == "commits":
            commits = gh.list_commits(owner, repo, ref=resolved_ref, path=None, limit=limit)
            return mcp_json({
                "action": "commits",
                "owner": owner,
                "repo": repo,
                "ref": resolved_ref,
                "commits": [
                    {
                        "sha": c["sha"],
                        "author": (c.get("commit", {}).get("author", {}) or {}).get("name"),
                        "date": (c.get("commit", {}).get("author", {}) or {}).get("date"),
                        "message": (c.get("commit", {}) or {}).get("message", "").splitlines()[0],
                    }
                    for c in commits
                ],
            })

        if action == "branches":
            branches = gh.list_branches(owner, repo)
            return mcp_json({
                "action": "branches",
                "owner": owner,
                "repo": repo,
                "branches": [{"name": b["name"], "sha": b["commit"]["sha"]} for b in branches],
            })

        if action == "tags":
            tags = gh.list_tags(owner, repo)
            return mcp_json({
                "action": "tags",
                "owner": owner,
                "repo": repo,
                "tags": [{"name": t["name"], "sha": t["commit"]["sha"]} for t in tags],
            })

        if action == "file_history":
            if not path:
                return mcp_json({"error": "path is required for file_history action", "action": action})
            commits = gh.list_commits(owner, repo, ref=resolved_ref, path=path, limit=limit)
            return mcp_json({
                "action": "file_history",
                "owner": owner,
                "repo": repo,
                "ref": resolved_ref,
                "path": path,
                "commits": [
                    {
                        "sha": c["sha"],
                        "author": (c.get("commit", {}).get("author", {}) or {}).get("name"),
                        "date": (c.get("commit", {}).get("author", {}) or {}).get("date"),
                        "message": (c.get("commit", {}) or {}).get("message", "").splitlines()[0],
                    }
                    for c in commits
                ],
            })

        return mcp_json({"error": f"Unknown action: {action}", "valid_actions": ["repos", "outline", "commits", "branches", "tags", "file_history"]})

    # TOOL 5: index - Index repository and check status
    @mcp.tool(**write_tool_opts)
    def index(
        action: str = "start",
        repo: Optional[str] = None,
        ref: str = "HEAD",
        job_id: Optional[str] = None
    ) -> dict:
        """
        Index a repository for semantic search, or check indexing status.

        Args:
            action: "start" to begin indexing, "status" to check job progress
            repo: Repository name (required for "start")
            ref: Git ref to index (default "HEAD", resolves to default branch)
            job_id: Job ID to check status (required for "status" action)

        The indexing process runs asynchronously. Use "status" to poll for completion.

        Examples:
            - index(action="start", repo="myrepo") - start indexing
            - index(action="status", job_id="abc-123-...") - check progress
        """
        logger.info("tool_call: index action=%s repo=%s job_id=%s", action, repo, job_id)
        owner = settings.github_owner

        if action == "status":
            if not job_id:
                return mcp_json({"error": "job_id is required for status action", "action": action})
            row = db.fetchone(
                "SELECT status, progress, message, owner, repo, ref FROM index_jobs WHERE job_id=%s",
                (job_id,),
            )
            if not row:
                return mcp_json({"job_id": job_id, "found": False})
            return mcp_json({
                "action": "status",
                "job_id": job_id,
                "found": True,
                "status": row["status"],
                "progress": float(row.get("progress", 0) or 0),
                "message": row.get("message"),
                "owner": row.get("owner"),
                "repo": row.get("repo"),
                "ref": row.get("ref"),
            })

        # Default: start indexing
        if not repo:
            return mcp_json({"error": "repo is required for start action", "action": action})

        resolved_ref = _default_ref(gh, owner, repo, ref)
        job_id = str(uuid.uuid4())
        db.execute(
            "INSERT INTO index_jobs(job_id, owner, repo, ref, status, progress, message) VALUES (%s,%s,%s,%s,'PENDING',0,'Queued')",
            (job_id, owner, repo, resolved_ref),
        )
        return mcp_json({
            "action": "start",
            "job_id": job_id,
            "status": "PENDING",
            "owner": owner,
            "repo": repo,
            "ref": resolved_ref,
            "hint": "Use index(action='status', job_id='...') to check progress",
        })

    return mcp


def main() -> None:
    server = build_server()
    port = int(os.getenv("PORT", "8000"))
    # Default to streamable-http (recommended by OpenAI for ChatGPT)
    # Set MCP_TRANSPORT=sse for legacy SSE transport if needed
    transport = os.getenv("MCP_TRANSPORT", "http")
    logger.info("server_listening instance_id=%s port=%d transport=%s", INSTANCE_ID, port, transport)
    server.run(transport=transport, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
