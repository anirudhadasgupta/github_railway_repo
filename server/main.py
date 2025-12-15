import asyncio
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP
from starlette.responses import JSONResponse, PlainTextResponse

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

# Session cache with 1-hour TTL for application-level context persistence
# This persists context across MCP reconnections
SESSION_TTL_SECONDS = 3600  # 1 hour
_session_cache: Dict[str, Dict[str, Any]] = {}

# Connection keepalive tracking
_last_heartbeat: Dict[str, float] = {}
HEARTBEAT_INTERVAL_SECONDS = 30


def _get_or_create_session(session_token: Optional[str]) -> tuple[str, Dict[str, Any]]:
    """Get existing session or create new one. Returns (token, session_data)."""
    now = time.time()

    # Clean expired sessions
    expired = [k for k, v in _session_cache.items() if now - v.get("_created", 0) > SESSION_TTL_SECONDS]
    for k in expired:
        del _session_cache[k]

    # Try to find existing session
    if session_token and session_token in _session_cache:
        session = _session_cache[session_token]
        session["_last_access"] = now
        return session_token, session

    # Create new session
    new_token = str(uuid.uuid4())
    _session_cache[new_token] = {"_created": now, "_last_access": now}
    return new_token, _session_cache[new_token]


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
    NOTE: This does not (and cannot) control OpenAI/ChatGPT's rotating wrapper
    path (e.g. /ghtool/link_*). Clients must re-discover tools when wrapper
    routes change.
    """
    return {
        "name": "github-private-repo-mcp",
        "instance_id": INSTANCE_ID,
        "uptime_seconds": round(time.time() - INSTANCE_START_TIME, 2),
        "session_ttl_seconds": SESSION_TTL_SECONDS,
        "heartbeat_interval_seconds": HEARTBEAT_INTERVAL_SECONDS,
        "discovery": discovery,
        "http_endpoints": {
            "health": "/health",
            "help": "/help",
            "help_well_known": "/.well-known/mcp/help",
            "capabilities": "/capabilities",
        },
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

    # Keep the recovery playbook embedded here so the client/model can recover
    # even if /help is temporarily unreachable during tool-wrapper rotation.
    instructions = """
    You are connected to a GitHub code intelligence MCP.

    SESSION PERSISTENCE: Call get_session() at the start of each conversation to get
    a session_token. Store and reuse this token across reconnections to maintain
    context for up to 1 hour.

    WRAPPER DISCOVERY (MCP metadata for ghtool/* calls):
    - Before calling any ghtool/* function, call api_tool.list_resources with
      refetch_tools=true to discover the current wrapper path; do not reuse or
      copy /link_* paths from prior turns.
    - If a ghtool call returns "Resource not found", immediately rerun
      list_resources(refetch_tools=true) and retry once with the newly discovered
      path.
    - If a ghtool call fails with 502 or 424, call heartbeat (or get_session if
      missing) and retry once.

    Prefer semantic_search + gh_get_file_content for answering questions.
    Use keyword_search for exact string matches.
    Use file_history and gh_file_diff when reasoning about changes.

    RESOURCES: Use list_resources to discover available data sources.

    ERROR RECOVERY PLAYBOOK (available without calling help):
    - If you get "Resource not found" (often caused by tool-wrapper/URI rotation):
      1) re-discover tools (tools/list or list_resources(refetch_tools=true))
      2) DO NOT reuse cached /link_* paths
      3) retry the same operation once
    - If you get 502/Bad Gateway/timeout/424 wrapper:
      1) heartbeat(session_token) (or get_session() if missing)
      2) retry once (optional tiny backoff)
    - If heartbeat says session_valid=false:
      1) get_session() to refresh
      2) retry prior operation

    Discovery hints: /help and /.well-known/mcp/help return this playbook plus
    capabilities metadata; /capabilities returns discovery info only.
    """

    # Enable stateless_http to handle OpenAI's DELETE requests that terminate sessions
    # Session persistence is handled at application level via _session_cache
    mcp = FastMCP(name="github-private-repo-mcp", instructions=instructions, stateless_http=True)

    # Tool annotations for OpenAI compatibility
    # readOnlyHint: indicates tools don't modify state
    tool_opts = {"annotations": {"readOnlyHint": True, "safe": True}}

    # Search tool annotations per OpenAI specifications
    # openWorldHint: indicates the tool searches external/dynamic data
    search_tool_opts = {"annotations": {"readOnlyHint": True, "openWorldHint": True, "safe": True}}

    # Fetch tool annotations - retrieves specific content by ID
    fetch_tool_opts = {"annotations": {"readOnlyHint": True, "safe": True}}

    @mcp.custom_route("/health", methods=["GET"])
    async def health_check(request):
        return JSONResponse({
            "status": "ok",
            "instance_id": INSTANCE_ID,
            "uptime_seconds": round(time.time() - INSTANCE_START_TIME, 2),
            "active_sessions": len(_session_cache),
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
        "title": "Error recovery playbook",
        "rules": [
            {
                "when": "before calling any ghtool/* tool",
                "do": [
                    "list_resources(refetch_tools=true) via api_tool before choosing the wrapper path",
                    "do not reuse cached /link_* wrapper paths from prior turns",
                ],
            },
            {
                "when": "error == 'Resource not found'",
                "do": [
                    "re-discover tools (tools/list or list_resources(refetch_tools=true))",
                    "DO NOT reuse cached /link_* wrapper paths",
                    "retry the same operation once",
                ],
            },
            {
                "when": "error in {502, gateway-ish, 424 wrapper}",
                "do": [
                    "heartbeat(session_token) or get_session if missing",
                    "retry once (optional tiny backoff)",
                ],
            },
            {
                "when": "heartbeat returns session_valid=false",
                "do": [
                    "get_session() to refresh session",
                    "retry prior operation",
                ],
            },
        ],
    }

    help_tool_opts = {"annotations": {"readOnlyHint": True, "safe": True}}

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

    @mcp.tool(**help_tool_opts)
    def help() -> dict:
        """Return the error recovery playbook for retrying MCP tool calls."""
        discovery = _build_discovery(root_path=os.getenv("ROOT_PATH", ""))
        caps = _base_capabilities(discovery)
        return mcp_json({**help_payload, "capabilities": caps})

    @mcp.tool(**tool_opts)
    def capabilities() -> dict:
        """
        Return server manifest + discovery hints. Useful for clients after tool-wrapper
        rotation or transport changes.
        """
        discovery = _build_discovery(root_path=os.getenv("ROOT_PATH", ""))
        return mcp_json(_base_capabilities(discovery))

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

    @mcp.resource("github://session")
    def resource_session_info() -> str:
        """Current session information and TTL settings."""
        import json
        return json.dumps({
            "instance_id": INSTANCE_ID,
            "uptime_seconds": round(time.time() - INSTANCE_START_TIME, 2),
            "session_ttl_seconds": SESSION_TTL_SECONDS,
            "heartbeat_interval_seconds": HEARTBEAT_INTERVAL_SECONDS,
            "active_sessions": len(_session_cache),
        }, ensure_ascii=False)

    # ---- Heartbeat/Keepalive Tool ----
    @mcp.tool(**tool_opts)
    def heartbeat(session_token: Optional[str] = None) -> dict:
        """
        Send a heartbeat to keep the connection alive. Call periodically (every 30s)
        to prevent connection timeouts. Returns session status.
        """
        now = time.time()
        if session_token:
            _last_heartbeat[session_token] = now
            if session_token in _session_cache:
                _session_cache[session_token]["_last_access"] = now
        return mcp_json({
            "status": "alive",
            "timestamp": now,
            "session_valid": session_token in _session_cache if session_token else False,
        })

    @mcp.tool(**tool_opts)
    def get_session(session_token: Optional[str] = None) -> dict:
        """
        Get or create a persistent session token. Pass this token to subsequent
        tool calls to maintain context across reconnections. Sessions persist for 1 hour.
        """
        token, session_data = _get_or_create_session(session_token)
        return mcp_json({
            "session_token": token,
            "ttl_seconds": SESSION_TTL_SECONDS,
            "created": session_data.get("_created"),
            "last_access": session_data.get("_last_access"),
            "hint": "Pass this session_token to other tools to persist context across reconnections"
        })

    @mcp.tool(**tool_opts)
    def gh_list_repos() -> dict:
        logger.info("tool_call: gh_list_repos")
        repos = gh.list_repos()
        return mcp_json(
            {
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
            }
        )

    @mcp.tool(**tool_opts)
    def gh_repo_outline(repo: str, ref: str = "HEAD", max_entries: int = 2000) -> dict:
        logger.info("tool_call: gh_repo_outline repo=%s ref=%s", repo, ref)
        owner = settings.github_owner
        ref = _default_ref(gh, owner, repo, ref)
        tree = gh.get_tree_recursive(owner, repo, ref)
        entries = tree.get("tree", [])[:max_entries]
        outline = [{"path": e["path"], "type": e["type"], "size": e.get("size")} for e in entries]
        return mcp_json({"owner": owner, "repo": repo, "ref": ref, "entries": outline})

    @mcp.tool(**tool_opts)
    def gh_get_file_content(
        repo: str, path: str, ref: str = "HEAD", start_line: int = 1, end_line: int = 400
    ) -> dict:
        logger.info("tool_call: gh_get_file_content repo=%s path=%s ref=%s", repo, path, ref)
        owner = settings.github_owner
        ref = _default_ref(gh, owner, repo, ref)
        text = gh.read_text_file(owner, repo, path, ref)
        lines = text.splitlines()
        sl = max(1, start_line)
        el = min(len(lines), end_line)
        snippet = "\n".join(lines[sl - 1 : el])
        url = f"https://github.com/{owner}/{repo}/blob/{ref}/{path}"
        return mcp_json(
            {
                "owner": owner,
                "repo": repo,
                "ref": ref,
                "path": path,
                "start_line": sl,
                "end_line": el,
                "url": url,
                "text": snippet,
            }
        )

    @mcp.tool(**tool_opts)
    def gh_file_history(repo: str, path: str, ref: str = "HEAD", limit: int = 20) -> dict:
        logger.info("tool_call: gh_file_history repo=%s path=%s ref=%s", repo, path, ref)
        owner = settings.github_owner
        ref = _default_ref(gh, owner, repo, ref)
        commits = gh.list_commits(owner, repo, ref=ref, path=path, limit=limit)
        return mcp_json(
            {
                "owner": owner,
                "repo": repo,
                "ref": ref,
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
            }
        )

    @mcp.tool(**tool_opts)
    def gh_file_diff(repo: str, base: str, head: str) -> dict:
        logger.info("tool_call: gh_file_diff repo=%s base=%s head=%s", repo, base, head)
        owner = settings.github_owner
        comp = gh.compare(owner, repo, base=base, head=head)
        files = comp.get("files", []) or []
        return mcp_json(
            {
                "owner": owner,
                "repo": repo,
                "base": base,
                "head": head,
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
            }
        )

    @mcp.tool(**tool_opts)
    def gh_commit_history(repo: str, ref: str = "HEAD", limit: int = 30) -> dict:
        logger.info("tool_call: gh_commit_history repo=%s ref=%s", repo, ref)
        owner = settings.github_owner
        ref = _default_ref(gh, owner, repo, ref)
        commits = gh.list_commits(owner, repo, ref=ref, path=None, limit=limit)
        return mcp_json(
            {
                "owner": owner,
                "repo": repo,
                "ref": ref,
                "commits": [{"sha": c["sha"], "message": c["commit"]["message"].splitlines()[0]} for c in commits],
            }
        )

    @mcp.tool(**tool_opts)
    def gh_branch_history(repo: str) -> dict:
        logger.info("tool_call: gh_branch_history repo=%s", repo)
        owner = settings.github_owner
        branches = gh.list_branches(owner, repo)
        return mcp_json(
            {
                "owner": owner,
                "repo": repo,
                "branches": [{"name": b["name"], "sha": b["commit"]["sha"]} for b in branches],
            }
        )

    @mcp.tool(**tool_opts)
    def gh_tag_history(repo: str) -> dict:
        logger.info("tool_call: gh_tag_history repo=%s", repo)
        owner = settings.github_owner
        tags = gh.list_tags(owner, repo)
        return mcp_json(
            {
                "owner": owner,
                "repo": repo,
                "tags": [{"name": t["name"], "sha": t["commit"]["sha"]} for t in tags],
            }
        )

    @mcp.tool(**tool_opts)
    def gh_index_repo(repo: str, ref: str = "HEAD") -> dict:
        logger.info("tool_call: gh_index_repo repo=%s ref=%s", repo, ref)
        owner = settings.github_owner
        ref = _default_ref(gh, owner, repo, ref)
        job_id = str(uuid.uuid4())
        db.execute(
            "INSERT INTO index_jobs(job_id, owner, repo, ref, status, progress, message) VALUES (%s,%s,%s,%s,'PENDING',0,'Queued')",
            (job_id, owner, repo, ref),
        )
        return mcp_json({"job_id": job_id, "status": "PENDING", "owner": owner, "repo": repo, "ref": ref})

    @mcp.tool(**tool_opts)
    def gh_index_status(job_id: str) -> dict:
        logger.info("tool_call: gh_index_status job_id=%s", job_id)
        row = db.fetchone(
            "SELECT status, progress, message, owner, repo, ref FROM index_jobs WHERE job_id=%s",
            (job_id,),
        )
        if not row:
            return mcp_json({"job_id": job_id, "found": False})
        return mcp_json(
            {
                "job_id": job_id,
                "found": True,
                "status": row["status"],
                "progress": float(row.get("progress", 0) or 0),
                "message": row.get("message"),
                "owner": row.get("owner"),
                "repo": row.get("repo"),
                "ref": row.get("ref"),
            }
        )

    @mcp.tool(**search_tool_opts)
    def keyword_search(repo: str, query: str, limit: int = 20) -> dict:
        """
        Search for exact keyword matches in indexed code chunks.
        Use this for finding specific strings, function names, or identifiers.
        """
        logger.info("tool_call: keyword_search repo=%s query=%s", repo, query)
        results = db.search_keyword(repo, query, limit=limit)
        formatted = [
            {
                "chunk_id": r["chunk_id"],
                "owner": r["owner"],
                "repo": r["name"],
                "ref": r["ref"],
                "path": r["path"],
                "start_line": r["start_line"],
                "end_line": r["end_line"],
                "text": r["text"],
            }
            for r in results
        ]
        return mcp_json({"query": query, "results": formatted})

    @mcp.tool(**search_tool_opts)
    def semantic_search(repo: str, query: str, limit: int = 10) -> dict:
        """
        Semantic search using embeddings to find conceptually similar code.
        Use this for natural language queries about code functionality.
        """
        logger.info("tool_call: semantic_search repo=%s query=%s", repo, query)
        matches = indexer.semantic_search(repo, query, limit=limit)
        hydrated: List[Dict[str, Any]] = []
        for m in matches:
            chunk = db.get_chunk(m.get("chunk_id")) if m.get("chunk_id") else None
            hydrated.append(
                {
                    "score": m.get("score"),
                    "chunk_id": m.get("chunk_id"),
                    "repo": m.get("repo"),
                    "ref": m.get("ref"),
                    "path": m.get("path"),
                    "start_line": m.get("start_line"),
                    "end_line": m.get("end_line"),
                    "symbol": m.get("symbol"),
                    "text": chunk.get("text") if chunk else None,
                }
            )
        return mcp_json({"query": query, "results": hydrated})

    @mcp.tool(**search_tool_opts)
    def search(query: str, repo: Optional[str] = None, limit: int = 8) -> dict:
        """
        OpenAI deep research compatible search tool.
        Searches indexed code repositories using semantic similarity.
        Returns results with IDs that can be fetched with the fetch tool.
        """
        logger.info("tool_call: search query=%s repo=%s", query, repo)
        if repo is None:
            repo = ""
        matches = indexer.semantic_search(repo, query, limit=limit) if repo else []
        results = []
        for m in matches:
            chunk = db.get_chunk(m.get("chunk_id")) if m.get("chunk_id") else None
            if not chunk:
                continue
            repo_name = chunk["repo_name"]
            results.append(
                {
                    "id": chunk["chunk_id"],
                    "title": f"{chunk['path']}:{chunk['start_line']}-{chunk['end_line']}",
                    "text": chunk["text"],
                    "url": f"https://github.com/{settings.github_owner}/{repo_name}/blob/{chunk['ref']}/{chunk['path']}#L{chunk['start_line']}-L{chunk['end_line']}",
                }
            )
        return {
            "content": [
                {
                    "type": "text",
                    "text": mcp_json({"results": results})["content"][0]["text"],
                }
            ]
        }

    @mcp.tool(**fetch_tool_opts)
    def fetch(id: str) -> dict:
        """
        OpenAI deep research compatible fetch tool.
        Retrieves full content of a code chunk by its ID.
        Use IDs returned from the search tool.
        """
        logger.info("tool_call: fetch id=%s", id)
        chunk = db.get_chunk(id)
        if not chunk:
            return {
                "content": [
                    {"type": "text", "text": mcp_json({"found": False, "id": id})["content"][0]["text"]}
                ]
            }
        payload = {
            "id": chunk["chunk_id"],
            "title": f"{chunk['path']}:{chunk['start_line']}-{chunk['end_line']}",
            "text": chunk["text"],
            "url": f"https://github.com/{settings.github_owner}/{chunk['repo_name']}/blob/{chunk['ref']}/{chunk['path']}#L{chunk['start_line']}-L{chunk['end_line']}",
        }
        return {"content": [{"type": "text", "text": mcp_json(payload)["content"][0]["text"]}]}

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
