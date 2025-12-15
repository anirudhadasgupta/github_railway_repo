import base64
import json
from typing import Any, Dict, Optional, Tuple


def mcp_json(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    MCP tool result helper: return a single text content item that contains JSON.

    OpenAI MCP formatting requirement: tool results must be returned as a content array
    with exactly one item: { type: "text", text: "<JSON-encoded string>" }
    """
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(data, ensure_ascii=False),
            }
        ]
    }


def encode_doc_id(owner: str, repo: str, ref: str, path: str) -> str:
    """
    Encode document reference into a URL-safe base64 ID.

    Format: base64url("owner/repo/ref/path")

    This creates a self-contained ID that fetch() can decode without
    needing additional parameters - critical for OpenAI's fetch(id: string) contract.
    """
    combined = f"{owner}/{repo}/{ref}/{path}"
    # Use URL-safe base64 encoding (replaces + with -, / with _, no padding)
    encoded = base64.urlsafe_b64encode(combined.encode("utf-8")).decode("ascii")
    # Remove padding for cleaner IDs
    return encoded.rstrip("=")


def decode_doc_id(doc_id: str) -> Optional[Tuple[str, str, str, str]]:
    """
    Decode a document ID back to (owner, repo, ref, path).

    Returns None if the ID is invalid or cannot be decoded.
    """
    try:
        # Add back padding if needed
        padding = 4 - (len(doc_id) % 4)
        if padding != 4:
            doc_id += "=" * padding

        decoded = base64.urlsafe_b64decode(doc_id.encode("ascii")).decode("utf-8")
        parts = decoded.split("/", 3)  # Split into at most 4 parts
        if len(parts) != 4:
            return None
        owner, repo, ref, path = parts
        return (owner, repo, ref, path)
    except Exception:
        return None
