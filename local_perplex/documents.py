"""Private document vault: parsing, caching and query-focused retrieval."""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

from .retrieval import Chunk, ChunkIndex

logger = logging.getLogger(__name__)

TEXT_EXTS = {".txt", ".md", ".markdown", ".rst", ".log", ".json", ".yaml", ".yml", ".csv",
             ".py", ".js", ".ts", ".java", ".c", ".cpp", ".go", ".rs", ".tex"}
BINARY_EXTS = {".pdf", ".docx", ".xlsx"}
HTML_EXTS = {".html", ".htm"}
SUPPORTED_EXTS = TEXT_EXTS | BINARY_EXTS | HTML_EXTS
MAX_PDF_PAGES = 60
MAX_SHEET_ROWS = 500


class DocumentManager:
    def __init__(self, doc_dir: str | Path | None = None):
        if doc_dir is None:
            doc_dir = os.getenv("LOCAL_PERPLEX_DOCUMENT_DIR") or Path(__file__).parent.parent / "documents"
        self.doc_dir = Path(doc_dir)
        self.doc_dir.mkdir(parents=True, exist_ok=True)
        self._text_cache: dict[str, tuple[float, int, str]] = {}
        self._index: ChunkIndex | None = None
        self._index_key: tuple = ()
        self._lock = threading.Lock()

    # ── files ────────────────────────────────────────────────────────
    def files(self) -> list[Path]:
        return sorted((f for f in self.doc_dir.glob("*")
                       if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS and not f.name.startswith(".")),
                      key=lambda f: f.name.lower())

    def add_document(self, filename: str, content: str) -> Path:
        path = self.doc_dir / os.path.basename(filename)
        path.write_text(content, encoding="utf-8")
        return path

    def add_bytes(self, filename: str, data: bytes) -> Path:
        path = self.doc_dir / os.path.basename(filename)
        path.write_bytes(data)
        return path

    # ── parsing ──────────────────────────────────────────────────────
    def extract_document_text(self, file: str | Path) -> str:
        file = Path(file)
        try:
            stat = file.stat()
        except OSError:
            return ""
        key = str(file)
        cached = self._text_cache.get(key)
        if cached and cached[0] == stat.st_mtime and cached[1] == stat.st_size:
            return cached[2]
        try:
            text = _parse(file)
        except Exception as exc:  # noqa: BLE001 - a corrupt file must not break research
            logger.warning("Could not parse %s: %s", file.name, exc)
            text = ""
        self._text_cache[key] = (stat.st_mtime, stat.st_size, text)
        return text

    # ── retrieval ────────────────────────────────────────────────────
    def index(self) -> ChunkIndex:
        files = self.files()
        key = tuple((f.name, f.stat().st_mtime, f.stat().st_size) for f in files)
        with self._lock:
            if self._index is None or key != self._index_key:
                self._index = ChunkIndex((f.name, self.extract_document_text(f)) for f in files)
                self._index_key = key
            return self._index

    def search(self, query: str, k: int = 6) -> list[Chunk]:
        if not self.files():
            return []
        return self.index().search(query, k=k)

    def get_local_context(self, query: str | None = None, max_chars: int = 4000) -> str:
        """Context block for the LLM.

        With a ``query`` only the most relevant passages are included (RAG).
        Without one, falls back to the head of each document.
        """
        if query:
            chunks = self.search(query, k=8)
            parts, used = [], 0
            for c in chunks:
                block = f"[{c.source}] {c.text}"
                if used + len(block) > max_chars and parts:
                    break
                parts.append(block)
                used += len(block)
            return "\n\n".join(parts)
        parts, used = [], 0
        for f in self.files():
            text = self.extract_document_text(f)
            if not text:
                continue
            block = f"Local File: {f.name}\nContent: {text[:1500]}"
            if used + len(block) > max_chars and parts:
                break
            parts.append(block)
            used += len(block)
        return "\n\n".join(parts)

    def stats(self) -> dict:
        files = self.files()
        return {"files": len(files), "bytes": sum(f.stat().st_size for f in files),
                "chunks": len(self.index()) if files else 0}


def _parse(file: Path) -> str:
    ext = file.suffix.lower()
    if ext in TEXT_EXTS:
        return file.read_text(encoding="utf-8", errors="replace")
    if ext in HTML_EXTS:
        from .search import html_to_text
        return html_to_text(file.read_text(encoding="utf-8", errors="replace"))
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(file)
        return "\n\n".join((reader.pages[i].extract_text() or "")
                           for i in range(min(MAX_PDF_PAGES, len(reader.pages))))
    if ext == ".docx":
        import docx
        doc = docx.Document(file)
        parts = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text.strip() for cell in row.cells))
        return "\n\n".join(parts)
    if ext == ".xlsx":
        import openpyxl
        wb = openpyxl.load_workbook(file, data_only=True, read_only=True)
        out = []
        for ws in wb.worksheets:
            out.append(f"Sheet: {ws.title}")
            for row in ws.iter_rows(max_row=MAX_SHEET_ROWS, values_only=True):
                cells = [str(c) for c in row if c is not None]
                if cells:
                    out.append(", ".join(cells))
            out.append("")
        wb.close()
        return "\n".join(out)
    return ""
