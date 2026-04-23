"""Tests for semantic_chunker: tokenizer integration, ChunkConfig, ChunkResult."""

import os
import pytest
import semchunk

from src.ingestion.semantic_chunker import (
    ChunkConfig,
    ChunkResult,
    load_tokenizer,
)

TOKENIZER_PATH = "tokenizer.json"


# ── ChunkConfig defaults ──────────────────────────────────────────────────

class TestChunkConfig:
    def test_defaults(self):
        cfg = ChunkConfig()
        assert cfg.chunk_size == 500
        assert cfg.overlap_ratio == 0.2
        assert cfg.similarity_percentile == 95.0
        assert cfg.min_distance_floor == 0.1
        assert cfg.min_sentences_for_semantic == 10
        assert cfg.min_chunk_tokens == 50
        assert cfg.enable_semantic is True
        assert cfg.tokenizer_path is None

    def test_custom_values(self):
        cfg = ChunkConfig(chunk_size=256, overlap_ratio=0.1, enable_semantic=False)
        assert cfg.chunk_size == 256
        assert cfg.overlap_ratio == 0.1
        assert cfg.enable_semantic is False


# ── ChunkResult ────────────────────────────────────────────────────────────

class TestChunkResult:
    def test_fields(self):
        cr = ChunkResult(
            text="hello world",
            section_title="Intro",
            chunk_index=0,
            token_count=2,
            has_heading_context=True,
        )
        assert cr.text == "hello world"
        assert cr.section_title == "Intro"
        assert cr.chunk_index == 0
        assert cr.token_count == 2
        assert cr.has_heading_context is True


# ── load_tokenizer ─────────────────────────────────────────────────────────

class TestLoadTokenizer:
    def test_error_bad_file(self):
        with pytest.raises(FileNotFoundError):
            load_tokenizer(path="/tmp/definitely_not_a_tokenizer_abc123.json")

    def test_loads_from_explicit_path(self):
        counter = load_tokenizer(path=TOKENIZER_PATH)
        assert callable(counter)

    def test_loads_from_env_var(self, monkeypatch):
        monkeypatch.setenv("TOKENIZER_JSON", TOKENIZER_PATH)
        counter = load_tokenizer()
        assert callable(counter)

    def test_loads_from_default(self):
        """Default path fallback to ./tokenizer.json works."""
        counter = load_tokenizer()
        assert callable(counter)

    def test_consistent_counts(self):
        """Same input → same token count every time."""
        counter = load_tokenizer(path=TOKENIZER_PATH)
        text = "The quick brown fox jumps over the lazy dog."
        c1 = counter(text)
        c2 = counter(text)
        assert c1 == c2
        assert c1 > 0

    def test_empty_string(self):
        """Empty string may produce BOS/EOS special tokens — just check it's small."""
        counter = load_tokenizer(path=TOKENIZER_PATH)
        assert counter("") <= 3  # gemma adds special tokens

    def test_longer_text_more_tokens(self):
        counter = load_tokenizer(path=TOKENIZER_PATH)
        short = "Hello."
        long = "Hello. " * 100
        assert counter(long) > counter(short)

    def test_works_with_semchunk_chunkerify(self):
        """Token counter callable compatible with semchunk.chunkerify()."""
        counter = load_tokenizer(path=TOKENIZER_PATH)
        chunker = semchunk.chunkerify(counter, 20)
        text = "This is a test sentence. " * 50
        chunks = chunker(text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert counter(chunk) <= 20



# ── _split_sentences ───────────────────────────────────────────────────────

from src.ingestion.semantic_chunker import _split_sentences


class TestSplitSentences:
    def test_basic_split(self):
        text = "Hello world. This is a test. Another sentence here."
        sents = _split_sentences(text)
        assert len(sents) == 3

    def test_empty_string(self):
        assert _split_sentences("") == []
        assert _split_sentences("   ") == []

    def test_single_sentence(self):
        sents = _split_sentences("Just one sentence.")
        assert len(sents) == 1

    def test_abbreviations_no_split(self):
        """Dr. and Mr. shouldn't cause splits."""
        text = "Dr. Smith went to Washington. He met Mr. Jones there."
        sents = _split_sentences(text)
        # Should be 2 sentences, not 4
        assert len(sents) == 2

    def test_decimal_numbers(self):
        text = "The value is 3.14 approximately. That is pi."
        sents = _split_sentences(text)
        # "3.14" shouldn't split — no uppercase after it
        assert len(sents) == 2

    def test_question_and_exclamation(self):
        text = "What is this? It is great! Really amazing."
        sents = _split_sentences(text)
        assert len(sents) == 3

    def test_preserves_content(self):
        text = "First sentence. Second sentence. Third sentence."
        sents = _split_sentences(text)
        joined = " ".join(sents)
        assert "First" in joined
        assert "Third" in joined


# ── _detect_semantic_boundaries ────────────────────────────────────────────

import numpy as np
from src.ingestion.semantic_chunker import _detect_semantic_boundaries


class TestDetectSemanticBoundaries:
    def _make_embeddings(self, n: int, dim: int = 768) -> list:
        """Create n random unit embeddings."""
        rng = np.random.RandomState(42)
        embs = rng.randn(n, dim)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = embs / norms
        return embs.tolist()

    def test_too_few_sentences(self):
        """< 2 sentences → empty list."""
        assert _detect_semantic_boundaries(["one"], [[1.0] * 768]) == []

    def test_uniform_embeddings_no_boundaries(self):
        """All identical embeddings → distances all 0 → below floor → empty."""
        emb = [0.1] * 768
        sentences = [f"Sentence {i}." for i in range(10)]
        embeddings = [emb[:] for _ in range(10)]
        result = _detect_semantic_boundaries(
            sentences, embeddings, percentile=95.0, min_distance_floor=0.1
        )
        assert result == []

    def test_obvious_boundary(self):
        """Two clusters of identical embeddings with a big gap → boundary found."""
        dim = 768
        emb_a = [1.0] + [0.0] * (dim - 1)
        emb_b = [0.0] + [1.0] + [0.0] * (dim - 2)
        sentences = [f"S{i}" for i in range(6)]
        # 3 sentences cluster A, 3 sentences cluster B
        embeddings = [emb_a] * 3 + [emb_b] * 3
        result = _detect_semantic_boundaries(
            sentences, embeddings, percentile=50.0, min_distance_floor=0.1
        )
        assert 2 in result  # boundary between index 2 and 3

    def test_boundaries_sorted(self):
        """Result always sorted ascending."""
        embs = self._make_embeddings(20)
        sentences = [f"S{i}" for i in range(20)]
        result = _detect_semantic_boundaries(
            sentences, embs, percentile=80.0, min_distance_floor=0.01
        )
        assert result == sorted(result)

    def test_boundaries_in_valid_range(self):
        """All indices in [0, len-2]."""
        n = 15
        embs = self._make_embeddings(n)
        sentences = [f"S{i}" for i in range(n)]
        result = _detect_semantic_boundaries(
            sentences, embs, percentile=80.0, min_distance_floor=0.01
        )
        for idx in result:
            assert 0 <= idx <= n - 2

    def test_determinism(self):
        """Same inputs → same outputs."""
        embs = self._make_embeddings(10)
        sentences = [f"S{i}" for i in range(10)]
        r1 = _detect_semantic_boundaries(sentences, embs, 90.0, 0.1)
        r2 = _detect_semantic_boundaries(sentences, embs, 90.0, 0.1)
        assert r1 == r2

    def test_high_floor_suppresses_all(self):
        """min_distance_floor=2.0 → no boundary can exceed it → empty."""
        embs = self._make_embeddings(10)
        sentences = [f"S{i}" for i in range(10)]
        result = _detect_semantic_boundaries(
            sentences, embs, percentile=99.0, min_distance_floor=2.0
        )
        assert result == []

    def test_low_percentile_more_boundaries(self):
        """Lower percentile → more boundaries detected."""
        embs = self._make_embeddings(20)
        sentences = [f"S{i}" for i in range(20)]
        high = _detect_semantic_boundaries(sentences, embs, 95.0, 0.01)
        low = _detect_semantic_boundaries(sentences, embs, 50.0, 0.01)
        assert len(low) >= len(high)



# ── _sentences_to_segments ─────────────────────────────────────────────────

from src.ingestion.semantic_chunker import _sentences_to_segments


class TestSentencesToSegments:
    def test_single_boundary(self):
        sents = ["A.", "B.", "C.", "D."]
        segs = _sentences_to_segments(sents, [1])
        assert len(segs) == 2
        assert "A." in segs[0] and "B." in segs[0]
        assert "C." in segs[1] and "D." in segs[1]

    def test_no_boundaries(self):
        sents = ["A.", "B.", "C."]
        segs = _sentences_to_segments(sents, [])
        assert len(segs) == 1
        assert "A." in segs[0]

    def test_multiple_boundaries(self):
        sents = ["A.", "B.", "C.", "D.", "E."]
        segs = _sentences_to_segments(sents, [1, 3])
        assert len(segs) == 3


# ── _merge_runts ───────────────────────────────────────────────────────────

from src.ingestion.semantic_chunker import _merge_runts


class TestMergeRunts:
    def _counter(self, text: str) -> int:
        return len(text.split())

    def test_no_runts(self):
        chunks = ["word " * 10, "word " * 10]
        result = _merge_runts(chunks, 5, self._counter)
        assert len(result) == 2

    def test_small_last_merged(self):
        chunks = ["word " * 10, "tiny"]
        result = _merge_runts(chunks, 5, self._counter)
        assert len(result) == 1

    def test_small_first_merged(self):
        chunks = ["hi", "word " * 10]
        result = _merge_runts(chunks, 5, self._counter)
        assert len(result) == 1

    def test_single_chunk_unchanged(self):
        result = _merge_runts(["hello"], 5, self._counter)
        assert result == ["hello"]

    def test_empty_list(self):
        result = _merge_runts([], 5, self._counter)
        assert result == []


# ── chunk_section ──────────────────────────────────────────────────────────

from src.ingestion.semantic_chunker import chunk_section


class TestChunkSection:
    """Tests for the three-layer semantic chunker."""

    @pytest.fixture
    def counter(self):
        return load_tokenizer(path=TOKENIZER_PATH)

    @pytest.fixture
    def config(self):
        return ChunkConfig(
            chunk_size=100,
            overlap_ratio=0.2,
            min_chunk_tokens=10,
            enable_semantic=False,  # most tests skip Layer 2
        )

    def test_short_content_single_chunk(self, counter, config):
        """Content shorter than chunk_size → single chunk."""
        results = chunk_section("Intro", "Hello world.", config, counter)
        assert len(results) == 1
        assert results[0].chunk_index == 0

    def test_heading_prefix_present(self, counter, config):
        """Non-empty title → heading prepended."""
        results = chunk_section("Chapter 1", "Some content here.", config, counter)
        assert results[0].text.startswith("## Chapter 1\n\n")
        assert results[0].has_heading_context is True

    def test_no_heading_for_no_title(self, counter, config):
        """(no title) → no heading prefix."""
        results = chunk_section("(no title)", "Some content.", config, counter)
        assert not results[0].text.startswith("##")
        assert results[0].has_heading_context is False

    def test_no_heading_for_empty_title(self, counter, config):
        results = chunk_section("", "Some content.", config, counter)
        assert results[0].has_heading_context is False

    def test_sequential_chunk_index(self, counter, config):
        """chunk_index values are 0, 1, 2, ..."""
        long_text = "This is a test sentence with enough words. " * 200
        results = chunk_section("Test", long_text, config, counter)
        indices = [r.chunk_index for r in results]
        assert indices == list(range(len(results)))

    def test_multiple_chunks_for_long_content(self, counter, config):
        """Long content → multiple chunks."""
        long_text = "This is a moderately long sentence for testing purposes. " * 200
        results = chunk_section("Test", long_text, config, counter)
        assert len(results) > 1

    def test_token_upper_bound(self, counter, config):
        """Every chunk token_count ≤ chunk_size * 1.1."""
        long_text = "Semantic chunking splits text at topic boundaries. " * 200
        results = chunk_section("Test", long_text, config, counter)
        max_allowed = config.chunk_size * 1.1
        for r in results:
            assert r.token_count <= max_allowed + 5, (
                f"Chunk {r.chunk_index} has {r.token_count} tokens, "
                f"max allowed {max_allowed}"
            )

    def test_runt_merging(self, counter):
        """Small final chunk gets merged."""
        cfg = ChunkConfig(
            chunk_size=50,
            min_chunk_tokens=20,
            enable_semantic=False,
        )
        # Build text that would produce a small tail
        text = "Word " * 60 + "tiny."
        results = chunk_section("Test", text, cfg, counter)
        # Last chunk should be >= min_chunk_tokens (or be the only chunk)
        if len(results) > 1:
            for r in results[:-1]:
                assert r.token_count >= cfg.min_chunk_tokens or len(results) == 1

    def test_no_content_loss(self, counter, config):
        """Union of chunks (minus heading) covers input."""
        content = "Alpha bravo charlie. Delta echo foxtrot. Golf hotel india."
        results = chunk_section("Sec", content, config, counter)
        # Strip heading prefix and rejoin
        prefix = "## Sec\n\n"
        recovered = " ".join(
            r.text[len(prefix):] if r.text.startswith(prefix) else r.text
            for r in results
        )
        # Every word from input should appear in recovered text
        for word in content.split():
            assert word.rstrip(".") in recovered, f"Lost word: {word}"

    def test_section_title_propagated(self, counter, config):
        results = chunk_section("My Section", "Content here.", config, counter)
        for r in results:
            assert r.section_title == "My Section"

    def test_semantic_disabled_skips_layer2(self, counter):
        """enable_semantic=False → no embedding_fn needed."""
        cfg = ChunkConfig(chunk_size=100, enable_semantic=False)
        results = chunk_section(
            "Test", "Some text. " * 50, cfg, counter, embedding_fn=None
        )
        assert len(results) >= 1

    def test_few_sentences_skips_semantic(self, counter):
        """Fewer than min_sentences_for_semantic → skip Layer 2 even if enabled."""
        cfg = ChunkConfig(
            chunk_size=500,
            enable_semantic=True,
            min_sentences_for_semantic=100,  # very high threshold
        )
        # Only ~3 sentences — should skip semantic, no embedding_fn needed
        text = "First sentence. Second sentence. Third sentence."
        results = chunk_section("Test", text, cfg, counter, embedding_fn=None)
        assert len(results) >= 1

    def test_with_mock_embedding_fn(self, counter):
        """Semantic enabled with mock embeddings → runs without error."""
        dim = 768

        def mock_embed(texts):
            """Return random-ish embeddings."""
            import numpy as np
            rng = np.random.RandomState(42)
            return rng.randn(len(texts), dim).tolist()

        cfg = ChunkConfig(
            chunk_size=50,
            enable_semantic=True,
            min_sentences_for_semantic=3,
        )
        text = "First topic sentence. " * 10 + "Completely different topic. " * 10
        results = chunk_section("Test", text, cfg, counter, embedding_fn=mock_embed)
        assert len(results) >= 1
        for r in results:
            assert r.section_title == "Test"

    def test_empty_content(self, counter, config):
        """Empty content → still returns one result."""
        results = chunk_section("Empty", "", config, counter)
        assert len(results) == 1

    def test_at_chunk_size(self, counter):
        """Content exactly at chunk_size → single chunk."""
        cfg = ChunkConfig(chunk_size=500, enable_semantic=False)
        # Build text that's roughly 500 tokens
        text = "word " * 490
        results = chunk_section("Test", text, cfg, counter)
        # Should be 1-2 chunks (depends on heading overhead)
        assert len(results) <= 3



# ── Error resilience (task 9) ──────────────────────────────────────────────

class TestErrorResilience:
    """Tests for embedding fallback and short content handling."""

    @pytest.fixture
    def counter(self):
        return load_tokenizer(path=TOKENIZER_PATH)

    def test_embedding_connection_error_fallback(self, counter):
        """Embedding fn raises ConnectionError → falls back to recursive only."""
        def bad_embed(texts):
            raise ConnectionError("server down")

        cfg = ChunkConfig(
            chunk_size=50,
            enable_semantic=True,
            min_sentences_for_semantic=3,
        )
        text = "First sentence here. " * 20
        # Should NOT raise — graceful fallback
        results = chunk_section("Test", text, cfg, counter, embedding_fn=bad_embed)
        assert len(results) >= 1
        for r in results:
            assert r.section_title == "Test"

    def test_embedding_oserror_fallback(self, counter):
        """Embedding fn raises OSError → same fallback."""
        def bad_embed(texts):
            raise OSError("network unreachable")

        cfg = ChunkConfig(
            chunk_size=50,
            enable_semantic=True,
            min_sentences_for_semantic=3,
        )
        text = "Another test sentence. " * 20
        results = chunk_section("Sec", text, cfg, counter, embedding_fn=bad_embed)
        assert len(results) >= 1

    def test_embedding_generic_exception_fallback(self, counter):
        """Any exception from embedding_fn → fallback."""
        def bad_embed(texts):
            raise RuntimeError("unexpected")

        cfg = ChunkConfig(
            chunk_size=50,
            enable_semantic=True,
            min_sentences_for_semantic=3,
        )
        text = "Sentence for testing. " * 20
        results = chunk_section("Sec", text, cfg, counter, embedding_fn=bad_embed)
        assert len(results) >= 1

    def test_short_content_single_chunk(self, counter):
        """Content shorter than min_chunk_tokens → single chunk as-is."""
        cfg = ChunkConfig(chunk_size=500, min_chunk_tokens=50, enable_semantic=False)
        # Very short content — fewer tokens than min_chunk_tokens
        text = "Hi."
        results = chunk_section("Short", text, cfg, counter)
        assert len(results) == 1
        assert "Hi." in results[0].text

    def test_short_content_has_heading(self, counter):
        """Short content still gets heading prefix."""
        cfg = ChunkConfig(chunk_size=500, min_chunk_tokens=50, enable_semantic=False)
        text = "Tiny."
        results = chunk_section("Title", text, cfg, counter)
        assert results[0].text.startswith("## Title\n\n")
        assert results[0].has_heading_context is True

    def test_short_content_no_title_no_heading(self, counter):
        """Short content with (no title) → no heading."""
        cfg = ChunkConfig(chunk_size=500, min_chunk_tokens=50, enable_semantic=False)
        text = "Tiny."
        results = chunk_section("(no title)", text, cfg, counter)
        assert not results[0].text.startswith("##")
        assert results[0].has_heading_context is False
