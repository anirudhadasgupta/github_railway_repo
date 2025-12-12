import ast
import hashlib
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


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
    lines = text.splitlines()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return [(None, 1, len(lines), text)]

    chunks: List[Tuple[Optional[str], int, int, str]] = []

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
            snippet = "\n".join(lines[start - 1 : end])
            chunks.append((sym, start, end, snippet))

    if not chunks:
        chunks = [(None, 1, len(lines), text)]
    return chunks


def fallback_line_chunks(
    repo: str, ref: str, path: str, text: str, max_lines: int = 200, overlap: int = 30
) -> List[Tuple[Optional[str], int, int, str]]:
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
            continue
        seen_hashes.add(h)

        cid = make_chunk_id(repo, ref, path, start, end, h)
        chunks.append(
            CodeChunk(
                chunk_id=cid,
                repo=repo,
                ref=ref,
                path=path,
                language=language,
                symbol=symbol,
                start_line=start,
                end_line=end,
                content_hash=h,
                text=snippet,
            )
        )
    return chunks
