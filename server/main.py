"""
GitHub Code Intelligence MCP Server - ChatGPT Compatible

This server implements the OpenAI MCP connector contract with exactly 2 tools:
- search(query: string) -> { results: [{ id, title, url }] }
- fetch(id: string) -> { id, title, text, url, metadata? }

Both tools have readOnlyHint=true for ChatGPT compatibility.
Extra features (explore, index) are available via HTTP endpoints only.
"""

import asyncio
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, StreamingResponse

from .config import require_core_secrets, settings
from .db import Database
from .github_client import GitHubClient
from .indexer import Indexer
from .mcp_utils import decode_doc_id, encode_doc_id, mcp_json
from .qdrant_store import QdrantStore

# Structured logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}',
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("mcp.server")

# Instance diagnostics
INSTANCE_ID = str(uuid.uuid4())
INSTANCE_START_TIME = time.time()

# Heartbeat configuration for Railway load balancer (configurable via env)
HEARTBEAT_INTERVAL_SECONDS = int(os.getenv("HEARTBEAT_INTERVAL_SECONDS", "30"))


class ConnectionKeepAliveMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add headers that prevent Railway/load balancers from closing connections.

    - Connection: keep-alive - Tells Railway not to close the socket
    - X-Accel-Buffering: no - Disables nginx buffering for instant data transmission
    - Cache-Control: no-cache, no-transform - Prevents caching and compression
    """

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Add keep-alive headers to all responses
        response.headers["Connection"] = "keep-alive"
        response.headers["X-Accel-Buffering"] = "no"
        response.headers["Cache-Control"] = "no-cache, no-transform"

        return response


def _default_ref(gh: GitHubClient, owner: str, repo: str, ref: str) -> str:
    """Resolve HEAD to actual default branch."""
    if ref != "HEAD":
        return ref
    meta = gh.get_repo(owner, repo)
    return meta.get("default_branch") or "main"


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

    # Minimal instructions for ChatGPT - focus on the 2-tool contract
    instructions = """
    GitHub code search MCP. Two tools: search and fetch.
    - search(query): Find code across repositories
    - fetch(id): Get full content of a search result
    """

    # Stateless HTTP for ChatGPT compatibility (handles DELETE requests that terminate sessions)
    mcp = FastMCP(name="github-private-repo-mcp", instructions=instructions, stateless_http=True)

    # ========================================================================
    # MCP TOOLS - Exactly 2 tools per OpenAI ChatGPT connector contract
    # ========================================================================

    # OpenAI tool annotations for ChatGPT compatibility
    # readOnlyHint: true = skips "Continue" confirmation prompt
    # openWorldHint: true = searches external/dynamic data
    # destructiveHint: false = not a destructive operation
    search_annotations = {
        "annotations": {
            "readOnlyHint": True,
            "openWorldHint": True,
            "destructiveHint": False,
        }
    }

    fetch_annotations = {
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
        }
    }

    @mcp.tool(**search_annotations)
    def search(query: str) -> dict:
        """
        Search for code across GitHub repositories.

        Args:
            query: Search query. Supports natural language or GitHub qualifiers:
                   - "authentication logic" (natural language)
                   - "repo:myrepo auth" (scope to specific repo)
                   - "path:src/api endpoint" (scope to path)
                   - "language:python async" (filter by language)

        Returns:
            Results array with id, title, and url for each match.
            Use fetch(id) to get the full content of any result.
        """
        logger.info("tool_call: search query=%s", query)
        owner = settings.github_owner
        results = []

        try:
            # Primary: GitHub Code Search API (instant, no indexing required)
            search_response = gh.search_code(query, owner=owner, per_page=10)
            items = search_response.get("items", [])

            for item in items:
                repo_info = item.get("repository", {})
                repo_name = repo_info.get("name", "")
                repo_owner = repo_info.get("owner", {}).get("login", owner)
                path = item.get("path", "")
                sha = item.get("sha", "HEAD")

                # Get the default branch for the URL
                ref = repo_info.get("default_branch", "main")

                # Create self-contained ID for fetch
                doc_id = encode_doc_id(repo_owner, repo_name, ref, path)

                # Build GitHub URL
                url = f"https://github.com/{repo_owner}/{repo_name}/blob/{ref}/{path}"

                results.append({
                    "id": doc_id,
                    "title": f"{repo_name}/{path}",
                    "url": url,
                })

            logger.info("search_github_results count=%d", len(results))

        except Exception as e:
            logger.warning("github_search_failed error=%s, trying fallback", str(e))

            # Fallback: Try indexed semantic search if GitHub search fails
            try:
                # Extract repo hint from query if present (e.g., "repo:myrepo")
                repo_hint = ""
                if "repo:" in query:
                    parts = query.split()
                    for p in parts:
                        if p.startswith("repo:"):
                            repo_hint = p.replace("repo:", "").split("/")[-1]
                            break

                if repo_hint:
                    matches = indexer.semantic_search(repo_hint, query, limit=10)
                    for m in matches:
                        chunk = db.get_chunk(m.get("chunk_id")) if m.get("chunk_id") else None
                        if not chunk:
                            continue

                        chunk_owner = chunk["owner"]
                        repo_name = chunk["repo_name"]
                        path = chunk["path"]
                        ref = chunk["ref"]

                        doc_id = encode_doc_id(chunk_owner, repo_name, ref, path)
                        url = f"https://github.com/{chunk_owner}/{repo_name}/blob/{ref}/{path}#L{chunk['start_line']}-L{chunk['end_line']}"

                        results.append({
                            "id": doc_id,
                            "title": f"{repo_name}/{path}:{chunk['start_line']}-{chunk['end_line']}",
                            "url": url,
                        })

                    logger.info("search_fallback_results count=%d", len(results))

            except Exception as fallback_error:
                logger.warning("fallback_search_failed error=%s", str(fallback_error))

        # OpenAI contract: { results: [{ id, title, url }] }
        return mcp_json({"results": results})

    @mcp.tool(**fetch_annotations)
    def fetch(id: str) -> dict:
        """
        Fetch the full content of a document by its ID.

        Args:
            id: Document ID from search results. This is a self-contained
                identifier that encodes the repository, ref, and file path.

        Returns:
            Document with id, title, text content, url, and metadata.
        """
        logger.info("tool_call: fetch id=%s", id)

        # Decode the self-contained ID
        decoded = decode_doc_id(id)
        if not decoded:
            return mcp_json({
                "id": id,
                "title": "Error",
                "text": f"Invalid document ID: {id}",
                "url": "",
                "metadata": {"error": True},
            })

        owner, repo, ref, path = decoded
        logger.info("fetch_decoded owner=%s repo=%s ref=%s path=%s", owner, repo, ref, path)

        try:
            # Fetch file content from GitHub
            content = gh.read_text_file(owner, repo, path, ref)
            lines = content.splitlines()
            url = f"https://github.com/{owner}/{repo}/blob/{ref}/{path}"

            # Detect language from extension
            ext = path.rsplit(".", 1)[-1] if "." in path else ""
            lang_map = {
                "py": "python", "js": "javascript", "ts": "typescript",
                "jsx": "javascript", "tsx": "typescript", "java": "java",
                "go": "go", "rs": "rust", "rb": "ruby", "php": "php",
                "c": "c", "cpp": "cpp", "h": "c", "hpp": "cpp",
                "cs": "csharp", "swift": "swift", "kt": "kotlin",
                "md": "markdown", "json": "json", "yaml": "yaml", "yml": "yaml",
                "sql": "sql", "sh": "shell", "bash": "shell",
            }
            language = lang_map.get(ext.lower(), ext)

            # OpenAI contract: { id, title, text, url, metadata? }
            return mcp_json({
                "id": id,
                "title": f"{repo}/{path}",
                "text": content,
                "url": url,
                "metadata": {
                    "owner": owner,
                    "repo": repo,
                    "ref": ref,
                    "path": path,
                    "lines": len(lines),
                    "language": language,
                },
            })

        except Exception as e:
            logger.error("fetch_failed error=%s", str(e))
            return mcp_json({
                "id": id,
                "title": f"{repo}/{path}",
                "text": f"Error fetching file: {str(e)}",
                "url": f"https://github.com/{owner}/{repo}/blob/{ref}/{path}",
                "metadata": {"error": True, "message": str(e)},
            })

    # ========================================================================
    # HTTP ENDPOINTS - Extra features (NOT MCP tools, for management/debug)
    # ========================================================================

    @mcp.custom_route("/health", methods=["GET"])
    async def health_check(request):
        """Health check endpoint."""
        return JSONResponse({
            "status": "ok",
            "instance_id": INSTANCE_ID,
            "uptime_seconds": round(time.time() - INSTANCE_START_TIME, 2),
        })

    @mcp.custom_route("/heartbeat", methods=["GET"])
    async def heartbeat_stream(request):
        """
        SSE heartbeat endpoint that sends pings every 30s.

        Prevents Railway/load balancer idle timeout by keeping the connection active.
        Clients can connect to this endpoint to maintain a persistent connection.

        Response format: Server-Sent Events (SSE)
        - event: ping
        - data: {"time": <unix_timestamp>, "instance_id": "<id>"}
        """

        async def generate_heartbeat():
            """Generate SSE heartbeat events every HEARTBEAT_INTERVAL_SECONDS."""
            try:
                while True:
                    data = {
                        "time": int(time.time()),
                        "instance_id": INSTANCE_ID,
                        "uptime_seconds": round(time.time() - INSTANCE_START_TIME, 2),
                    }
                    # SSE format: event line, data line, blank line
                    yield f"event: ping\ndata: {data}\n\n"
                    await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)
            except asyncio.CancelledError:
                logger.info("heartbeat_stream_closed instance_id=%s", INSTANCE_ID)
                raise

        return StreamingResponse(
            generate_heartbeat(),
            media_type="text/event-stream",
            headers={
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache, no-transform",
            },
        )

    @mcp.custom_route("/", methods=["GET"])
    async def root_route(request):
        """Landing page with discovery info."""
        return JSONResponse({
            "status": "ok",
            "message": "GitHub MCP server (ChatGPT compatible)",
            "tools": ["search", "fetch"],
            "instance_id": INSTANCE_ID,
            "uptime_seconds": round(time.time() - INSTANCE_START_TIME, 2),
        })

    @mcp.custom_route("/.well-known/mcp/help", methods=["GET"])
    async def help_route(request):
        """Recovery playbook for clients."""
        return JSONResponse({
            "title": "MCP Recovery Playbook",
            "tools": ["search", "fetch"],
            "recovery": {
                "trigger": "Resource not found | tool missing | 404 | 502",
                "steps": [
                    "1. Call tools/list to rediscover tools",
                    "2. Retry the failed operation",
                ],
            },
            "instance_id": INSTANCE_ID,
        })

    # ---- API Endpoints for extra features (HTTP only, not MCP) ----

    @mcp.custom_route("/api/repos", methods=["GET"])
    async def api_repos(request):
        """List accessible repositories."""
        repos = gh.list_repos()
        return JSONResponse({
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

    @mcp.custom_route("/api/repos/{repo}/outline", methods=["GET"])
    async def api_repo_outline(request):
        """Get repository file tree."""
        repo = request.path_params.get("repo", "")
        ref = request.query_params.get("ref", "HEAD")
        limit = int(request.query_params.get("limit", "500"))

        owner = settings.github_owner
        resolved_ref = _default_ref(gh, owner, repo, ref)

        tree = gh.get_tree_recursive(owner, repo, resolved_ref)
        entries = tree.get("tree", [])[:min(limit, 2000)]

        return JSONResponse({
            "owner": owner,
            "repo": repo,
            "ref": resolved_ref,
            "entries": [
                {"path": e["path"], "type": e["type"], "size": e.get("size")}
                for e in entries
            ],
        })

    @mcp.custom_route("/api/repos/{repo}/commits", methods=["GET"])
    async def api_repo_commits(request):
        """Get recent commits."""
        repo = request.path_params.get("repo", "")
        ref = request.query_params.get("ref", "HEAD")
        limit = int(request.query_params.get("limit", "30"))

        owner = settings.github_owner
        resolved_ref = _default_ref(gh, owner, repo, ref)

        commits = gh.list_commits(owner, repo, ref=resolved_ref, path=None, limit=limit)
        return JSONResponse({
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

    @mcp.custom_route("/api/index", methods=["POST"])
    async def api_index_repo(request):
        """Queue a repository for indexing."""
        try:
            body = await request.json()
        except Exception:
            body = {}

        repo = body.get("repo") or request.query_params.get("repo", "")
        ref = body.get("ref") or request.query_params.get("ref", "HEAD")

        if not repo:
            return JSONResponse({"error": "repo is required"}, status_code=400)

        owner = settings.github_owner
        resolved_ref = _default_ref(gh, owner, repo, ref)

        job_id = str(uuid.uuid4())
        db.execute(
            "INSERT INTO index_jobs(job_id, owner, repo, ref, status, progress, message) VALUES (%s,%s,%s,%s,'PENDING',0,'Queued')",
            (job_id, owner, repo, resolved_ref),
        )

        return JSONResponse({
            "job_id": job_id,
            "status": "PENDING",
            "owner": owner,
            "repo": repo,
            "ref": resolved_ref,
        })

    @mcp.custom_route("/api/index/{job_id}", methods=["GET"])
    async def api_index_status(request):
        """Check indexing job status."""
        job_id = request.path_params.get("job_id", "")

        row = db.fetchone(
            "SELECT status, progress, message, owner, repo, ref FROM index_jobs WHERE job_id=%s",
            (job_id,),
        )

        if not row:
            return JSONResponse({"job_id": job_id, "found": False}, status_code=404)

        return JSONResponse({
            "job_id": job_id,
            "found": True,
            "status": row["status"],
            "progress": float(row.get("progress", 0) or 0),
            "message": row.get("message"),
            "owner": row.get("owner"),
            "repo": row.get("repo"),
            "ref": row.get("ref"),
        })

    @mcp.custom_route("/api/indexed", methods=["GET"])
    async def api_indexed_repos(request):
        """List indexed repositories."""
        rows = db.fetchall(
            "SELECT DISTINCT owner, name FROM repos WHERE indexed_at IS NOT NULL ORDER BY name"
        )
        return JSONResponse({
            "indexed_repos": [{"owner": r["owner"], "name": r["name"]} for r in rows]
        })

    # Register middleware for keep-alive headers and buffering control
    # This ensures all responses include headers that prevent Railway/load balancers
    # from closing connections prematurely
    mcp._app.add_middleware(ConnectionKeepAliveMiddleware)
    logger.info("middleware_registered type=ConnectionKeepAliveMiddleware")

    return mcp


def main() -> None:
    server = build_server()
    port = int(os.getenv("PORT", "8000"))
    # Default to streamable-http (recommended by OpenAI for ChatGPT)
    transport = os.getenv("MCP_TRANSPORT", "http")
    logger.info("server_listening instance_id=%s port=%d transport=%s tools=search,fetch", INSTANCE_ID, port, transport)
    server.run(transport=transport, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
