import base64
import random
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
        if resp.status_code not in {403, 429}:
            return
        remaining = resp.headers.get("X-RateLimit-Remaining")
        reset = resp.headers.get("X-RateLimit-Reset")
        if remaining == "0" and reset:
            sleep_s = max(0, int(reset) - int(time.time()) + 1)
            time.sleep(min(sleep_s, 30))

    def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        attempt = 0
        while True:
            resp = self._client.request(method, url, **kwargs)
            if resp.status_code in {500, 502, 503, 504}:
                attempt += 1
                if attempt > 5:
                    resp.raise_for_status()
                    return resp
                jitter = random.uniform(0.5, 1.5)
                time.sleep(min(2**attempt * jitter, 10))
                continue
            if resp.status_code in {403, 429}:
                self._handle_rate_limit(resp)
            return resp

    def _get(self, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        resp = self._request("GET", url, params=params)
        resp.raise_for_status()
        return resp.json()

    def list_repos(self) -> List[Dict[str, Any]]:
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
