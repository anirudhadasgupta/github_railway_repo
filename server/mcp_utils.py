import json
from typing import Any, Dict


def mcp_json(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    MCP tool result helper: return a single text content item that contains JSON.
    """
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(data, ensure_ascii=False),
            }
        ]
    }
