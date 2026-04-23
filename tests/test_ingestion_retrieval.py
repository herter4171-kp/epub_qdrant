#!/usr/bin/env python3
"""Integration test harness for end-to-end ingestion + retrieval of books and papers.

Tests the full pipeline:
  1. Book ingestion (EPUB -> chunk_section -> embed dense + sparse -> upsert to test-books-named)
  2. Paper ingestion (PDF + JSON -> chunk_paper -> embed dense + sparse -> upsert to test-papers-named)
  3. Curl retrieval against Qdrant directly
  4. MCP server retrieval via JSON-RPC

All test collections are prefixed with ``test-`` and cleaned up after the suite.

Run:
    .venv/bin/python tests/test_ingestion_retrieval.py
"""

import json
import logging
import os
import sys
import unittest
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import requests
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    SparseVector,
    VectorParams,
    SparseVectorParams,
    Modifier,
    PointStruct,
)

# ---------------------------------------------------------------------------
# Paths and config
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Load .env before importing modules that read env vars at import time
from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")

_TEST_BOOKS = _PROJECT_ROOT / "test_books"
_DOWNLOADS = _PROJECT_ROOT / "downloads"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

QDRANT_HOST = os.getenv("QDRANT_HOST", "192.168.68.75")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_URL = f"http://{QDRANT_HOST}:{QDRANT_PORT}"
MCP_HOST = os.getenv("MCP_HOST", "localhost")
MCP_PORT = int(os.getenv("MCP_PORT", "8090"))
MCP_URL = f"http://{MCP_HOST}:{MCP_PORT}/mcp"

COLL_BOOKS = "test-books-named"
COLL_PAPERS = "test-papers-named"

BOOK_METADATA = {"publisher": "Apress", "language": "en", "isbn": "978-1-4842-0000-0"}
DENSE_SIZE = 768
SPARSE_WEIGHT_DEFAULT = 0.25

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from src.ingestion.epub_parser import parse_epub
from src.ingestion.chunker import chunk_section
from src.ingestion.paper_loader import chunk_paper
from servers.embedding_server.client import get_dense_vectors, get_sparse_vectors, health_check


# ===================================================================
# Helpers
# ===================================================================

def _first_epub() -> Optional[Path]:
    if not _TEST_BOOKS.exists():
        return None
    epubs = sorted(_TEST_BOOKS.glob("*.epub"))
    return epubs[0] if epubs else None


def _first_paper() -> tuple:
    if not _DOWNLOADS.exists():
        return None, None
    pdfs = sorted(_DOWNLOADS.glob("*.pdf"))
    for p in pdfs:
        jp = p.with_suffix(".json")
        if jp.exists():
            return p, jp
    return None, None


def _check_embedding_server_health() -> bool:
    try:
        return health_check()
    except Exception as e:
        logger.warning(f"Embedding server health check failed: {e}")
        return False


def _dense_embed(texts: List[str]) -> List[List[float]]:
    return get_dense_vectors(texts)


def _sparse_embed(texts: List[str], is_query: bool = False) -> List[Dict]:
    return get_sparse_vectors(texts, is_query=is_query)


def _delete_collection(client: QdrantClient, name: str) -> None:
    try:
        client.delete_collection(collection_name=name)
    except Exception:
        pass


def _create_named_coll(client: QdrantClient, name: str) -> None:
    client.create_collection(
        collection_name=name,
        vectors_config={"dense": VectorParams(size=DENSE_SIZE, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams(modifier=Modifier.IDF)},
    )
    for fld in ["source_file", "book_title", "section_title", "publisher", "language",
                "isbn", "arxiv_id", "category", "title", "doc_type"]:
        try:
            client.create_payload_index(collection_name=name, field_name=fld, field_schema="keyword")
        except Exception:
            pass


def _dense_embed(texts: List[str]) -> List[List[float]]:
    return get_dense_vectors(texts)


def _sparse_embed(texts: List[str], is_query: bool = False) -> List[Dict]:
    return get_sparse_vectors(texts, is_query=is_query)


def _mcp_query(mcp_url: str, name: str, args: dict, timeout: int = 60) -> dict:
    """Send an MCP tools/call request, parse JSON-RPC wrapper, return result dict."""
    payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {"name": name, "arguments": args},
        "id": 1,
    }
    r = requests.post(mcp_url, json=payload, timeout=timeout)
    r.raise_for_status()
    body = r.json()
    result = body.get("result", body)
    if isinstance(result, dict) and "content" in result:
        content = result["content"]
        if isinstance(content, list) and content:
            txt = content[0].get("text", "{}")
            result = json.loads(txt)
    return result


# ===================================================================
# Stage 1: Book Ingestion
# ===================================================================

class TestBookIngestion(unittest.TestCase):
    """Ingest one book into test-books-named with dense + sparse vectors."""

    @classmethod
    def setUpClass(cls):
        cls.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        cls.epub_path = _first_epub()
        if cls.epub_path is None:
            raise unittest.SkipTest("No EPUB files in test_books/")
        cls.book = parse_epub(str(cls.epub_path))
        assert len(cls.book.sections) > 0
        _delete_collection(cls.client, COLL_BOOKS)
        _create_named_coll(cls.client, COLL_BOOKS)
        cls.chunks: List = []
        for sec in cls.book.sections[:3]:
            chs = chunk_section(sec, chunk_size=500, chunk_overlap=100,
                                book_title=cls.book.title or "test-book",
                                publisher=BOOK_METADATA["publisher"],
                                language=BOOK_METADATA["language"],
                                isbn=BOOK_METADATA["isbn"])
            cls.chunks.extend(chs)
            if len(cls.chunks) >= 100:
                cls.chunks = cls.chunks[:100]
                break

    # NOTE: No tearDownClass — Stage 5 (Cleanup) owns all teardown.

    def test_01_epub_parsed(self):
        self.assertIsNotNone(self.book.title)
        self.assertTrue(len(self.book.sections) > 0)

    def test_02_chunks_produced(self):
        self.assertGreater(len(self.chunks), 0)
        self.assertLess(len(self.chunks), 500)
        self.assertIn("text", self.chunks[0].__dict__)

    def test_03_dense_embeds(self):
        texts = [c.text for c in self.chunks]
        vecs = _dense_embed(texts)
        self.assertEqual(len(vecs), len(texts))
        for v in vecs:
            self.assertEqual(len(v), DENSE_SIZE)

    def test_04_sparse_embeds(self):
        self.assertTrue(_check_embedding_server_health(), "Embedding server must be reachable")
        texts = [c.text for c in self.chunks[:10]]
        vecs = _sparse_embed(texts, is_query=False)
        self.assertEqual(len(vecs), len(texts))
        for sv in vecs:
            self.assertIn("indices", sv)
            self.assertIn("values", sv)

    def test_05_upsert(self):
        texts = [c.text for c in self.chunks]
        dv = _dense_embed(texts)
        sv = _sparse_embed(texts, is_query=False)
        points = []
        for i, (ch, d, s) in enumerate(zip(self.chunks, dv, sv)):
            points.append(PointStruct(
                id=i,
                vector={"dense": d, "sparse": SparseVector(indices=s["indices"], values=s["values"])},
                payload={
                    "text": ch.text, "book_title": ch.book_title,
                    "section_title": ch.section_title, "chapter_index": ch.chapter_index,
                    "section_index": ch.section_index, "chunk_index": ch.chunk_index,
                    "token_count": ch.token_count, "source_file": self.epub_path.name,
                    "publisher": ch.publisher, "language": ch.language, "isbn": ch.isbn,
                    "doc_type": "book",
                }))
        self.client.upsert(collection_name=COLL_BOOKS, points=points)
        info = self.client.get_collection(COLL_BOOKS)
        self.assertGreater(info.points_count, 0)
        logger.info(f"  Book ingestion: {info.points_count} points in {COLL_BOOKS}")

    def test_06_payload_fields(self):
        pts, _ = self.client.scroll(COLL_BOOKS, limit=5, with_payload=True, with_vectors=False)
        self.assertGreater(len(pts), 0)
        for fld in ["text", "book_title", "source_file", "publisher", "language", "isbn", "doc_type"]:
            for p in pts:
                self.assertIn(fld, p.payload, f"Point {p.id} missing {fld}")


# ===================================================================
# Stage 2: Paper Ingestion
# ===================================================================

class TestPaperIngestion(unittest.TestCase):
    """Ingest one paper into test-papers-named with dense + sparse vectors."""

    @classmethod
    def setUpClass(cls):
        cls.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        cls.pdf_path, cls.json_path = _first_paper()
        if cls.pdf_path is None:
            raise unittest.SkipTest("No PDF+JSON pair in downloads/")
        _delete_collection(cls.client, COLL_PAPERS)
        _create_named_coll(cls.client, COLL_PAPERS)
        with open(cls.json_path) as f:
            raw = json.load(f)
        cls.paper_meta = {}
        for attr in raw.get("metadataAttributes", []):
            if ": " in attr:
                k, v = attr.split(": ", 1)
                cls.paper_meta[k.lower()] = v
        import pypdf
        reader = pypdf.PdfReader(str(cls.pdf_path))
        cls.paper_text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        cls.chunks = chunk_paper(
            text=cls.paper_text, arxiv_id=cls.paper_meta.get("arxiv_id", "test"),
            title=cls.paper_meta.get("title", "Test Paper"),
            category=cls.paper_meta.get("category", "cs.AI"),
            subcategory=cls.paper_meta.get("subcategory", ""),
            authors=cls.paper_meta.get("authors", "Test Author"),
            publish_date=cls.paper_meta.get("publish_date", ""),
            abstract=cls.paper_meta.get("abstract", ""),
            source_file=cls.pdf_path.name, chunk_size=500, chunk_overlap=100)

    # NOTE: No tearDownClass — Stage 5 (Cleanup) owns all teardown.

    def test_01_metadata(self):
        self.assertIn("arxiv_id", self.paper_meta)
        self.assertIn("title", self.paper_meta)
        self.assertIn("category", self.paper_meta)

    def test_02_chunked(self):
        self.assertGreater(len(self.chunks), 0)
        self.assertLess(len(self.chunks), 500)
        self.assertEqual(self.chunks[0].arxiv_id, self.paper_meta.get("arxiv_id", "test"))

    def test_03_dense(self):
        texts = [c.text for c in self.chunks[:20]]
        vecs = _dense_embed(texts)
        self.assertEqual(len(vecs), len(texts))

    def test_04_sparse(self):
        self.assertTrue(_check_embedding_server_health(), "Embedding server must be reachable")
        texts = [c.text for c in self.chunks[:10]]
        vecs = _sparse_embed(texts, is_query=False)
        self.assertEqual(len(vecs), len(texts))

    def test_05_upsert(self):
        texts = [c.text for c in self.chunks]
        dv = _dense_embed(texts)
        sv = _sparse_embed(texts, is_query=False)
        points = []
        for i, (ch, d, s) in enumerate(zip(self.chunks, dv, sv)):
            points.append(PointStruct(
                id=i,
                vector={"dense": d, "sparse": SparseVector(indices=s["indices"], values=s["values"])},
                payload={
                    "text": ch.text, "arxiv_id": ch.arxiv_id, "title": ch.title,
                    "category": ch.category, "subcategory": ch.subcategory,
                    "authors": ch.authors, "publish_date": ch.publish_date,
                    "chunk_index": ch.chunk_index, "chunk_count": ch.chunk_count,
                    "token_count": ch.token_count, "source_file": ch.source_file,
                    "doc_type": "paper",
                }))
        self.client.upsert(collection_name=COLL_PAPERS, points=points)
        info = self.client.get_collection(COLL_PAPERS)
        self.assertGreater(info.points_count, 0)
        logger.info(f"  Paper ingestion: {info.points_count} points in {COLL_PAPERS}")

    def test_06_payload_fields(self):
        pts, _ = self.client.scroll(COLL_PAPERS, limit=5, with_payload=True, with_vectors=False)
        self.assertGreater(len(pts), 0)
        for fld in ["text", "arxiv_id", "title", "category", "source_file", "doc_type"]:
            for p in pts:
                self.assertIn(fld, p.payload, f"Point {p.id} missing {fld}")


# ===================================================================
# Stage 3: Curl Retrieval (Qdrant Direct)
# ===================================================================

class TestCurlRetrieval(unittest.TestCase):
    """Test retrieval against Qdrant directly via HTTP."""

    @classmethod
    def setUpClass(cls):
        cls.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    def _embed(self, q):
        return _dense_embed([q])[0]

    def test_01_books_vector(self):
        v = _dense_embed(["agentic AI enterprise"])[0]
        res = self.client.query_points(COLL_BOOKS, query=v, using="dense", limit=5)
        self.assertGreater(len(res.points), 0)
        self.assertIn("book_title", res.points[0].payload)

    def test_02_papers_vector(self):
        v = _dense_embed(["ALFWorld embodied agents"])[0]
        res = self.client.query_points(COLL_PAPERS, query=v, using="dense", limit=5)
        self.assertGreater(len(res.points), 0)
        self.assertIn("arxiv_id", res.points[0].payload)

    def test_03_books_hybrid(self):
        self.assertTrue(_check_embedding_server_health())
        d = _dense_embed(["agentic AI"])[0]
        s = _sparse_embed(["agentic AI"], is_query=True)[0]
        dr = self.client.query_points(COLL_BOOKS, query=d, using="dense", limit=5)
        sr = self.client.query_points(COLL_BOOKS, query=SparseVector(indices=s["indices"], values=s["values"]), using="sparse", limit=5)
        self.assertGreater(len(dr.points) + len(sr.points), 0)

    def test_04_papers_hybrid(self):
        self.assertTrue(_check_embedding_server_health())
        d = _dense_embed(["ALFWorld"])[0]
        s = _sparse_embed(["ALFWorld"], is_query=True)[0]
        dr = self.client.query_points(COLL_PAPERS, query=d, using="dense", limit=5)
        sr = self.client.query_points(COLL_PAPERS, query=SparseVector(indices=s["indices"], values=s["values"]), using="sparse", limit=5)
        self.assertGreater(len(dr.points) + len(sr.points), 0)

    def test_05_rrf_fusion(self):
        self.assertTrue(_check_embedding_server_health())
        d = _dense_embed(["agentic"])[0]
        s = _sparse_embed(["agentic"], is_query=True)[0]
        dh = self.client.query_points(COLL_BOOKS, query=d, using="dense", limit=10)
        sh = self.client.query_points(COLL_BOOKS, query=SparseVector(indices=s["indices"], values=s["values"]), using="sparse", limit=10)
        scores = defaultdict(float)
        for i, h in enumerate(dh.points):
            scores[h.id] += 1.0 / (60 + i + 1)
        for i, h in enumerate(sh.points):
            scores[h.id] += SPARSE_WEIGHT_DEFAULT * (1.0 / (60 + i + 1))
        merged = sorted(scores.items(), key=lambda x: -x[1])[:5]
        self.assertGreater(len(merged), 0)
        logger.info(f"  RRF fusion: {len(merged)} merged results")


# ===================================================================
# Stage 4: MCP Server Retrieval
# ===================================================================

class TestMCPRetrieval(unittest.TestCase):
    """Test retrieval via the MCP server JSON-RPC API."""

    @classmethod
    def setUpClass(cls):
        cls.mcp_url = MCP_URL

    def test_01_health(self):
        try:
            # Health endpoint is at /health, not /mcp/health
            base_url = self.mcp_url.rsplit("/mcp", 1)[0]
            r = requests.get(f"{base_url}/health", timeout=5)
            self.assertEqual(r.status_code, 200)
        except requests.ConnectionError:
            self.skipTest(f"MCP server not reachable at {self.mcp_url}")

    def test_02_query_books(self):
        try:
            res = _mcp_query(self.mcp_url, "query",
                             {"query": "agentic AI enterprise", "top_k": 5,
                              "collection": COLL_BOOKS, "mode": "search"})
            self.assertIn("groups", res)
            logger.info(f"  MCP books: {res.get('total_chunks', '?')} chunks")
        except requests.ConnectionError:
            self.skipTest("MCP server not reachable")

    def test_03_query_papers(self):
        try:
            res = _mcp_query(self.mcp_url, "query",
                             {"query": "ALFWorld embodied", "top_k": 5,
                              "collection": COLL_PAPERS, "mode": "search"})
            self.assertIn("groups", res)
            logger.info("  MCP papers: OK")
        except requests.ConnectionError:
            self.skipTest("MCP server not reachable")

    def test_04_list_collections(self):
        try:
            res = _mcp_query(self.mcp_url, "list_collections", {})
            names = {c.get("name") for c in res.get("collections", [])}
            logger.info(f"  MCP collections: {names}")
        except requests.ConnectionError:
            self.skipTest("MCP server not reachable")

    def test_05_cross_collection(self):
        try:
            res = _mcp_query(self.mcp_url, "query",
                             {"query": "AI agents",
                              "collections": f"{COLL_BOOKS},{COLL_PAPERS}",
                              "top_k": 5, "mode": "search"}, timeout=120)
            self.assertIn("groups", res)
            logger.info(f"  MCP cross-collection: {res.get('total_chunks', '?')} chunks")
        except requests.ConnectionError:
            self.skipTest("MCP server not reachable")


# ===================================================================
# Stage 5: Cleanup
# ===================================================================

class TestCleanup(unittest.TestCase):
    """Delete test collections and verify no orphans remain."""

    @classmethod
    def setUpClass(cls):
        cls.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    def test_00_exist(self):
        names = {c.name for c in self.client.get_collections().collections}
        self.assertIn(COLL_BOOKS, names)
        self.assertIn(COLL_PAPERS, names)

    def test_01_books(self):
        _delete_collection(self.client, COLL_BOOKS)
        names = {c.name for c in self.client.get_collections().collections}
        self.assertNotIn(COLL_BOOKS, names)

    def test_02_papers(self):
        _delete_collection(self.client, COLL_PAPERS)
        names = {c.name for c in self.client.get_collections().collections}
        self.assertNotIn(COLL_PAPERS, names)

    def test_03_no_orphans(self):
        names = {c.name for c in self.client.get_collections().collections}
        orphans = {c for c in names if c.startswith("test-")}
        self.assertEqual(len(orphans), 0, f"Orphans: {orphans}")


# ===================================================================
# Suite Runner
# ===================================================================

def run_all():
    """Run the full test suite and print a summary."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [TestBookIngestion, TestPaperIngestion, TestCurlRetrieval,
                TestMCPRetrieval, TestCleanup]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


if __name__ == "__main__":
    result = run_all()
    sys.exit(0 if result.wasSuccessful() else 1)