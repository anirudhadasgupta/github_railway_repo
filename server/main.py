import os
import uuid
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

from .config import require_core_secrets, settings
from .db import Database
from .github_client import GitHubClient
from .indexer import Indexer
from .mcp_utils import mcp_json
from .qdrant_store import QdrantStore


def _check_allow(token: Optional[str]) -> None:
    if settings.allow_token and token != settings.allow_token:
        raise PermissionError("Invalid allow token")


def _default_ref(gh: GitHubClient, owner: str, repo: str, ref: str) -> str:
    if ref != "HEAD":
        return ref
    meta = gh.get_repo(owner, repo)
    return meta.get("default_branch") or "main"


def build_server() -> FastMCP:
    require_core_secrets()

    gh = GitHubClient(settings.github_pat)
    qdrant = QdrantStore(settings.qdrant_url, settings.qdrant_api_key, settings.embedding_dims)
    qdrant.ensure_collection()

    db = Database(settings.database_url)
    indexer = Indexer(
        github=gh,
        qdrant=qdrant,
        db=db,
        openai_api_key=settings.openai_api_key,
        embedding_model=settings.embedding_model,
        embedding_dims=settings.embedding_dims,
    )

    instructions = """
    You are connected to a GitHub code intelligence MCP.
    Prefer semantic_search + gh_get_file_content for answering questions.
    Use keyword_search for exact string matches.
    Use file_history and gh_file_diff when reasoning about changes.
    """

    mcp = FastMCP(name="github-private-repo-mcp", instructions=instructions)
    tool_opts = {"annotations": {"readOnlyHint": True}}

    @mcp.tool(**tool_opts)
    def gh_list_repos(allow: Optional[str] = None) -> dict:
        _check_allow(allow)
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
    def gh_repo_outline(repo: str, ref: str = "HEAD", max_entries: int = 2000, allow: Optional[str] = None) -> dict:
        _check_allow(allow)
        owner = settings.github_owner
        ref = _default_ref(gh, owner, repo, ref)
        tree = gh.get_tree_recursive(owner, repo, ref)
        entries = tree.get("tree", [])[:max_entries]
        outline = [{"path": e["path"], "type": e["type"], "size": e.get("size")} for e in entries]
        return mcp_json({"owner": owner, "repo": repo, "ref": ref, "entries": outline})

    @mcp.tool(**tool_opts)
    def gh_get_file_content(
        repo: str, path: str, ref: str = "HEAD", start_line: int = 1, end_line: int = 400, allow: Optional[str] = None
    ) -> dict:
        _check_allow(allow)
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
    def gh_file_history(repo: str, path: str, ref: str = "HEAD", limit: int = 20, allow: Optional[str] = None) -> dict:
        _check_allow(allow)
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
    def gh_file_diff(repo: str, base: str, head: str, allow: Optional[str] = None) -> dict:
        _check_allow(allow)
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
    def gh_commit_history(repo: str, ref: str = "HEAD", limit: int = 30, allow: Optional[str] = None) -> dict:
        _check_allow(allow)
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
    def gh_branch_history(repo: str, allow: Optional[str] = None) -> dict:
        _check_allow(allow)
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
    def gh_tag_history(repo: str, allow: Optional[str] = None) -> dict:
        _check_allow(allow)
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
    def gh_index_repo(repo: str, ref: str = "HEAD", allow: Optional[str] = None) -> dict:
        _check_allow(allow)
        owner = settings.github_owner
        ref = _default_ref(gh, owner, repo, ref)
        job_id = str(uuid.uuid4())
        db.execute(
            "INSERT INTO index_jobs(job_id, owner, repo, ref, status, progress, message) VALUES (%s,%s,%s,%s,'PENDING',0,'Queued')",
            (job_id, owner, repo, ref),
        )
        return mcp_json({"job_id": job_id, "status": "PENDING", "owner": owner, "repo": repo, "ref": ref})

    @mcp.tool(**tool_opts)
    def gh_index_status(job_id: str, allow: Optional[str] = None) -> dict:
        _check_allow(allow)
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

    @mcp.tool(**tool_opts)
    def keyword_search(repo: str, query: str, limit: int = 20, allow: Optional[str] = None) -> dict:
        _check_allow(allow)
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

    @mcp.tool(**tool_opts)
    def semantic_search(repo: str, query: str, limit: int = 10, allow: Optional[str] = None) -> dict:
        _check_allow(allow)
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

    @mcp.tool(**tool_opts)
    def search(query: str, repo: Optional[str] = None, limit: int = 8, allow: Optional[str] = None) -> dict:
        _check_allow(allow)
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

    @mcp.tool(**tool_opts)
    def fetch(id: str, allow: Optional[str] = None) -> dict:
        _check_allow(allow)
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
    server.run(transport="sse", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
