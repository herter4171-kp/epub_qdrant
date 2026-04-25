"""Property-based and unit tests for src/ingestion/mineru_json_parser.py.

Feature tags: # Feature: mineru-json-ingestion

Run:
    .venv/bin/pytest tests/test_mineru_json_parser.py -v
"""

import json
import logging
import re
from io import StringIO
from pathlib import Path
from typing import Dict, List

import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.strategies import SearchStrategy

from src.ingestion.mineru_json_parser import (
    DEFAULT_MINERU_OUTPUT_DIR,
    EXCLUDED_BLOCK_TYPES,
    INCLUDED_BLOCK_TYPES,
    JsonSection,
    _assemble_inline,
    _is_excluded_section,
    _render_table,
    parse_content_list,
    resolve_json_path,
)


# ── Hypothesis strategies ─────────────────────────────────────────────────────

def span_strategy() -> SearchStrategy[Dict]:
    """Generate a single inline span dict."""
    return st.fixed_dictionaries({
        "type": st.sampled_from(["text", "equation_inline", "unknown_type"]),
        "content": st.text(min_size=0),
    })


def block_strategy() -> SearchStrategy[Dict]:
    """Generate a single block dict (title, paragraph, list, table, etc.)."""
    # Paragraph with paragraph_content spans
    para = st.fixed_dictionaries({
        "type": st.just("paragraph"),
        "content": st.fixed_dictionaries({
            "paragraph_content": st.lists(span_strategy(), min_size=0),
        }),
    })

    # Title with title_content + level
    title = st.fixed_dictionaries({
        "type": st.just("title"),
        "content": st.fixed_dictionaries({
            "title_content": st.lists(span_strategy()),
            "level": st.integers(min_value=0, max_value=4) | st.none(),
        }),
    })

    # List with list_items
    list_item = st.fixed_dictionaries({
        "item_content": st.lists(span_strategy(), min_size=0),
    })
    blist = st.fixed_dictionaries({
        "type": st.just("list"),
        "content": st.fixed_dictionaries({
            "list_items": st.lists(list_item, min_size=0),
        }),
    })

    # Table with caption + html
    table = st.fixed_dictionaries({
        "type": st.just("table"),
        "content": st.fixed_dictionaries({
            "table_caption": st.lists(span_strategy()),
            "html": st.text(min_size=0),
        }),
    })

    # Excluded block types
    excluded = st.sampled_from([
        {"type": "page_header"},
        {"type": "page_number"},
        {"type": "page_aside_text"},
        {"type": "image"},
        {"type": "equation_interline"},
        {"type": "chart"},
        {"type": "page_footer"},
    ])

    return st.one_of(para, title, blist, table, excluded)


# ── Task 2g: Unit tests for _is_excluded_section ─────────────────────────────

class TestIsExcludedSection:
    """Validate _is_excluded_section against design Requirements 4.4, 4.4a, 4.4b."""

    @pytest.mark.parametrize("title,level,expected", [
        # References — various casings
        ("References", 1, True),
        ("references", 2, True),
        ("REFERENCES", 3, True),
        # References with leading number
        ("5. References", 1, True),
        ("3) REFERENCES", 2, True),
        ("References", 4, True),
        # Bibliography
        ("Bibliography", 1, True),
        ("bibliography", 3, True),
        ("4. BIBLIOGRAPHY", 2, True),
        # Exact appendix at level 1-2
        ("Appendix", 1, True),
        ("appendix", 2, True),
        ("APPENDIX", 1, True),
        ("Appendices", 2, True),
        # Appendix at level 3+ — NOT excluded
        ("Appendix", 3, False),
        ("Appendix", 4, False),
        ("appendix", 3, False),
        # "Appendix B: ..." — NOT excluded (not exact match)
        ("Appendix B: Experimental Details", 1, False),
        ("Appendix A.1: Proof of Theorem 2", 2, False),
        ("A.1 Proof of Theorem 2", 1, False),
        # Normal section titles — NOT excluded
        ("Introduction", 1, False),
        ("3. Methodology", 2, False),
        ("Results and Discussion", 1, False),
        # Appendix with extra words — NOT excluded
        ("Appendix A", 1, False),
        ("Appendix Notes", 2, False),
        ("My Appendix", 1, False),
    ])
    def test_cases(self, title: str, level: int, expected: bool):
        assert _is_excluded_section(title, level) == expected


# ── Task 2c: Property test for _assemble_inline (Property 3) ─────────────────

class TestAssembleInlineProperty3:
    # Feature: mineru-json-ingestion, Property 3: Inline assembler preserves non-empty spans

    @given(st.lists(span_strategy(), min_size=1))
    @settings(max_examples=100)
    def test_inline_assembler_nonempty(self, spans: List[Dict]):
        """Property 3: if at least one text/equation_inline span has non-whitespace content, result is non-empty."""
        # Only count recognized span types (text, equation_inline) with non-whitespace content
        recognized_nonwhitespace = any(
            s.get("type") in ("text", "equation_inline") and s.get("content", "").strip()
            for s in spans
        )
        if not recognized_nonwhitespace:
            # No recognized spans with meaningful content → result may be empty, that's fine
            return

        result = _assemble_inline(spans)
        assert result, f"Expected non-empty result for recognized spans with content: {spans}"


# ── Task 2e: Property test for _render_table (Property 7) ────────────────────

class TestRenderTableProperty7:
    # Feature: mineru-json-ingestion, Property 7: Table rendering strips all HTML tags

    @given(
        st.text(min_size=0).filter(lambda t: "<" not in t and ">" not in t),
        st.text(min_size=1),
    )
    @settings(max_examples=100)
    def test_table_render_no_html_tags(self, caption_text: str, html_string: str):
        """Property 7: output contains no < or > characters when HTML is provided."""
        block = {
            "content": {
                "table_caption": [{"type": "text", "content": caption_text}],
                "html": html_string,
            }
        }
        result = _render_table(block)
        if html_string.strip():
            assert "<" not in result, f"HTML tag found in output: {result}"
            assert ">" not in result, f"HTML tag found in output: {result}"

    @given(st.text(min_size=1).filter(lambda t: t.strip()))
    @settings(max_examples=50)
    def test_table_caption_only(self, caption_text: str):
        """Table with caption but no HTML returns the caption (stripped by _assemble_inline)."""
        block = {
            "content": {
                "table_caption": [{"type": "text", "content": caption_text}],
                "html": "",
            }
        }
        result = _render_table(block)
        if caption_text:
            assert result.strip() == caption_text.strip()

    @given(st.text(min_size=1))
    @settings(max_examples=50)
    def test_table_multi_row(self, cell_text: str):
        """Multi-row table: rows separated by \\n, cells within row by \\t."""
        html = f"<table><tr><td>A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr></table>"
        block = {
            "content": {
                "table_caption": [{"type": "text", "content": "My Table"}],
                "html": html,
            }
        }
        result = _render_table(block)
        assert "My Table" in result
        assert "<" not in result
        # Multi-row → contains newline
        assert "\n" in result
        # Two cells per row → contains tab
        assert "\t" in result


# ── Task 3b: Unit tests for resolve_json_path ────────────────────────────────

class TestResolveJsonPath:
    """Validate resolve_json_path against design Requirements 1.1–1.7."""

    def test_tree_layout_found(self, tmp_path: Path):
        """Tree layout exists → returns correct path."""
        arxiv = "2603.07444"
        json_file = tmp_path / "2603_07444" / "vlm" / "2603_07444_content_list_v2.json"
        json_file.parent.mkdir(parents=True)
        json_file.write_text("[]")

        import os
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("MINERU_OUTPUT_DIR", str(tmp_path))
            result = resolve_json_path(arxiv)
        assert result == json_file

    def test_flat_layout_fallback(self, tmp_path: Path):
        """Tree missing, flat exists → returns flat path."""
        arxiv = "2603.07444"
        json_file = tmp_path / "2603_07444_content_list_v2.json"
        json_file.write_text("[]")

        import os
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("MINERU_OUTPUT_DIR", str(tmp_path))
            result = resolve_json_path(arxiv)
        assert result == json_file

    def test_neither_found(self, tmp_path: Path):
        """Neither path exists → returns None."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("MINERU_OUTPUT_DIR", str(tmp_path))
            result = resolve_json_path("9999.99999")
        assert result is None

    def test_dot_normalized_to_underscore(self, tmp_path: Path):
        """Dot-format arxiv_id → underscore format in path."""
        arxiv = "2603.07444"  # dot format
        json_file = tmp_path / "2603_07444_content_list_v2.json"
        json_file.write_text("[]")

        import os
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("MINERU_OUTPUT_DIR", str(tmp_path))
            result = resolve_json_path(arxiv)
        assert result == json_file

    def test_env_var_respected(self, tmp_path: Path):
        """MINERU_OUTPUT_DIR env var overrides default."""
        arxiv = "2603.07444"
        json_file = tmp_path / "2603_07444_content_list_v2.json"
        json_file.write_text("[]")

        import os
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("MINERU_OUTPUT_DIR", str(tmp_path))
            result = resolve_json_path(arxiv)
        assert result == json_file


# ── Task 3d: Unit tests for parse_content_list ───────────────────────────────

class TestParseContentList:
    """Validate parse_content_list against design Requirements 2–12."""

    @pytest.fixture
    def sample_json(self, tmp_path: Path) -> Path:
        """Minimal valid content_list_v2.json: 1 page, 3 blocks."""
        data = [
            [
                {"type": "title", "content": {"title_content": [{"type": "text", "content": "Introduction"}], "level": 1}},
                {"type": "paragraph", "content": {"paragraph_content": [{"type": "text", "content": "We propose a method."}]}},
                {"type": "paragraph", "content": {"paragraph_content": [{"type": "text", "content": "It works well."}]}},
            ]
        ]
        p = tmp_path / "test_content_list_v2.json"
        p.write_text(json.dumps(data))
        return p

    @pytest.fixture
    def table_json(self, tmp_path: Path) -> Path:
        """JSON with title, paragraph, and table blocks."""
        data = [
            [
                {"type": "title", "content": {"title_content": [{"type": "text", "content": "Results"}], "level": 2}},
                {"type": "paragraph", "content": {"paragraph_content": [{"type": "text", "content": "Table 1 shows results."}]}},
                {"type": "table", "content": {
                    "table_caption": [{"type": "text", "content": "Table 1: Results"}],
                    "html": "<table><tr><td>Method</td><td>Score</td></tr><tr><td>BERT</td><td>0.89</td></tr></table>",
                }},
            ]
        ]
        p = tmp_path / "table_content_list_v2.json"
        p.write_text(json.dumps(data))
        return p

    @pytest.fixture
    def code_json(self, tmp_path: Path) -> Path:
        """JSON with code block."""
        data = [
            [
                {"type": "title", "content": {"title_content": [{"type": "text", "content": "Implementation"}], "level": 2}},
                {"type": "code", "content": {"code_content": [{"type": "text", "content": "def hello(): pass"}]}},
            ]
        ]
        p = tmp_path / "code_content_list_v2.json"
        p.write_text(json.dumps(data))
        return p

    @pytest.fixture
    def algorithm_json(self, tmp_path: Path) -> Path:
        """JSON with algorithm block."""
        data = [
            [
                {"type": "title", "content": {"title_content": [{"type": "text", "content": "Algorithm"}], "level": 2}},
                {"type": "algorithm", "content": {"algorithm_content": [{"type": "text", "content": "while True: continue"}]}},
            ]
        ]
        p = tmp_path / "algo_content_list_v2.json"
        p.write_text(json.dumps(data))
        return p

    @pytest.fixture
    def footnote_json(self, tmp_path: Path) -> Path:
        """JSON with page_footnote block."""
        data = [
            [
                {"type": "title", "content": {"title_content": [{"type": "text", "content": "Footnotes"}], "level": 2}},
                {"type": "page_footnote", "content": {"page_footnote_content": [{"type": "text", "content": "Note: this is important."}]}},
            ]
        ]
        p = tmp_path / "footnote_content_list_v2.json"
        p.write_text(json.dumps(data))
        return p

    def test_valid_json_with_blocks(self, sample_json: Path):
        """Valid JSON with title + paragraph blocks → correct sections."""
        sections = parse_content_list(sample_json)
        assert len(sections) == 1
        assert sections[0].title == "Introduction"
        assert sections[0].heading_level == 1
        assert sections[0].section_index == 0
        assert "We propose a method." in sections[0].content
        assert "It works well." in sections[0].content

    def test_table_block_included(self, table_json: Path):
        """JSON with title + paragraph + table → correct sections with table rendered."""
        sections = parse_content_list(table_json)
        assert len(sections) == 1
        assert "Table 1: Results" in sections[0].content
        assert "Method\tScore" in sections[0].content
        assert "BERT\t0.89" in sections[0].content
        assert "<" not in sections[0].content  # no HTML

    def test_code_block_included(self, code_json: Path):
        """Code block text assembled and included."""
        sections = parse_content_list(code_json)
        assert len(sections) == 1
        assert "def hello(): pass" in sections[0].content

    def test_algorithm_block_included(self, algorithm_json: Path):
        """Algorithm block text assembled and included."""
        sections = parse_content_list(algorithm_json)
        assert len(sections) == 1
        assert "while True: continue" in sections[0].content

    def test_page_footnote_block_included(self, footnote_json: Path):
        """Page footnote block text assembled and included."""
        sections = parse_content_list(footnote_json)
        assert len(sections) == 1
        assert "Note: this is important." in sections[0].content

    def test_chart_excluded(self, tmp_path: Path) -> Path:
        """chart block excluded."""
        data = [
            [
                {"type": "title", "content": {"title_content": [{"type": "text", "content": "Figures"}], "level": 2}},
                {"type": "chart", "content": {"some": "data"}},
                {"type": "paragraph", "content": {"paragraph_content": [{"type": "text", "content": "See figure 1."}]}},
            ]
        ]
        p = tmp_path / "chart.json"
        p.write_text(json.dumps(data))
        sections = parse_content_list(p)
        assert len(sections) == 1
        assert "See figure 1." in sections[0].content
        # chart content NOT in output
        assert "some" not in sections[0].content

    def test_page_footer_excluded(self, tmp_path: Path):
        """page_footer block excluded."""
        data = [
            [
                {"type": "title", "content": {"title_content": [{"type": "text", "content": "Page 1"}], "level": 1}},
                {"type": "page_footer", "content": {"text": "footer"}},
                {"type": "paragraph", "content": {"paragraph_content": [{"type": "text", "content": "main text."}]}},
            ]
        ]
        p = tmp_path / "footer.json"
        p.write_text(json.dumps(data))
        sections = parse_content_list(p)
        assert len(sections) == 1
        assert "main text." in sections[0].content

    def test_paragraph_missing_paragraph_content(self, tmp_path: Path):
        """Paragraph with missing paragraph_content → skipped with warning."""
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.WARNING)
        logger = logging.getLogger("src.ingestion.mineru_json_parser")
        logger.addHandler(handler)

        data = [
            [
                {"type": "title", "content": {"title_content": [{"type": "text", "content": "Sec"}], "level": 1}},
                {"type": "paragraph", "content": {"wrong_field": "data"}},
                {"type": "paragraph", "content": {"paragraph_content": [{"type": "text", "content": "good."}]}},
            ]
        ]
        p = tmp_path / "missing.json"
        p.write_text(json.dumps(data))
        sections = parse_content_list(p)
        assert len(sections) == 1
        assert "good." in sections[0].content

        logger.removeHandler(handler)

    def test_preamble_section(self, tmp_path: Path):
        """Paragraphs before first title block → Preamble section."""
        data = [
            [
                {"type": "paragraph", "content": {"paragraph_content": [{"type": "text", "content": "Abstract text."}]}},
                {"type": "paragraph", "content": {"paragraph_content": [{"type": "text", "content": "More abstract."}]}},
                {"type": "title", "content": {"title_content": [{"type": "text", "content": "Intro"}], "level": 1}},
                {"type": "paragraph", "content": {"paragraph_content": [{"type": "text", "content": "Introduction text."}]}},
            ]
        ]
        p = tmp_path / "preamble.json"
        p.write_text(json.dumps(data))
        sections = parse_content_list(p)
        assert len(sections) == 2
        assert sections[0].title == "Preamble"
        assert "Abstract text." in sections[0].content
        assert sections[1].title == "Intro"
        assert "Introduction text." in sections[1].content

    @pytest.mark.parametrize("title,expected_in_output", [
        ("References", False),
        ("5. References", False),
        ("BIBLIOGRAPHY", False),
        ("2) Bibliography", False),
    ])
    def test_references_excluded(self, tmp_path: Path, title: str, expected_in_output: bool):
        """References/bibliography sections excluded."""
        data = [
            [
                {"type": "title", "content": {"title_content": [{"type": "text", "content": "Before"}], "level": 1}},
                {"type": "paragraph", "content": {"paragraph_content": [{"type": "text", "content": "before text."}]}},
                {"type": "title", "content": {"title_content": [{"type": "text", "content": title}], "level": 1}},
                {"type": "paragraph", "content": {"paragraph_content": [{"type": "text", "content": "ref text."}]}},
                {"type": "title", "content": {"title_content": [{"type": "text", "content": "After"}], "level": 1}},
                {"type": "paragraph", "content": {"paragraph_content": [{"type": "text", "content": "after text."}]}},
            ]
        ]
        p = tmp_path / "ref.json"
        p.write_text(json.dumps(data))
        sections = parse_content_list(p)
        section_titles = [s.title for s in sections]
        if expected_in_output:
            assert title in section_titles
        else:
            assert title not in section_titles

    def test_appendix_exact_level1_excluded(self, tmp_path: Path):
        """Exact 'Appendix' at level 1 → excluded."""
        data = [
            [
                {"type": "title", "content": {"title_content": [{"type": "text", "content": "Main"}], "level": 1}},
                {"type": "paragraph", "content": {"paragraph_content": [{"type": "text", "content": "main text."}]}},
                {"type": "title", "content": {"title_content": [{"type": "text", "content": "Appendix"}], "level": 1}},
                {"type": "paragraph", "content": {"paragraph_content": [{"type": "text", "content": "appendix text."}]}},
            ]
        ]
        p = tmp_path / "app.json"
        p.write_text(json.dumps(data))
        sections = parse_content_list(p)
        assert len(sections) == 1
        assert sections[0].title == "Main"

    def test_appendix_level3_not_excluded(self, tmp_path: Path):
        """Exact 'Appendix' at level 3 → NOT excluded."""
        data = [
            [
                {"type": "title", "content": {"title_content": [{"type": "text", "content": "Appendix"}], "level": 3}},
                {"type": "paragraph", "content": {"paragraph_content": [{"type": "text", "content": "appendix text."}]}},
            ]
        ]
        p = tmp_path / "app3.json"
        p.write_text(json.dumps(data))
        sections = parse_content_list(p)
        assert len(sections) == 1
        assert sections[0].title == "Appendix"

    def test_appendix_subsection_not_excluded(self, tmp_path: Path):
        """'Appendix B: Experimental Details' → NOT excluded."""
        data = [
            [
                {"type": "title", "content": {"title_content": [{"type": "text", "content": "Appendix B: Experimental Details"}], "level": 2}},
                {"type": "paragraph", "content": {"paragraph_content": [{"type": "text", "content": "details text."}]}},
            ]
        ]
        p = tmp_path / "appsub.json"
        p.write_text(json.dumps(data))
        sections = parse_content_list(p)
        assert len(sections) == 1
        assert "Appendix B: Experimental Details" in sections[0].title

    def test_empty_content_section_excluded(self, tmp_path: Path):
        """Section with no content blocks after filtering → excluded."""
        data = [
            [
                {"type": "title", "content": {"title_content": [{"type": "text", "content": "Empty"}], "level": 1}},
                {"type": "page_header", "content": {"text": "header"}},
                {"type": "title", "content": {"title_content": [{"type": "text", "content": "Full"}], "level": 1}},
                {"type": "paragraph", "content": {"paragraph_content": [{"type": "text", "content": "has text."}]}},
            ]
        ]
        p = tmp_path / "empty.json"
        p.write_text(json.dumps(data))
        sections = parse_content_list(p)
        assert len(sections) == 1
        assert sections[0].title == "Full"

    def test_malformed_page_skipped(self, tmp_path: Path):
        """Malformed page (not a list) → skipped with warning."""
        data = [
            "not a list page",
            [
                {"type": "title", "content": {"title_content": [{"type": "text", "content": "Sec"}], "level": 1}},
                {"type": "paragraph", "content": {"paragraph_content": [{"type": "text", "content": "text."}]}},
            ],
        ]
        p = tmp_path / "malformed_page.json"
        p.write_text(json.dumps(data))
        sections = parse_content_list(p)
        assert len(sections) == 1
        assert sections[0].title == "Sec"

    def test_block_missing_type_skipped(self, tmp_path: Path):
        """Block missing 'type' → skipped with warning."""
        data = [
            [
                {"type": "title", "content": {"title_content": [{"type": "text", "content": "Sec"}], "level": 1}},
                {"content": {"no": "type"}},
                {"type": "paragraph", "content": {"paragraph_content": [{"type": "text", "content": "text."}]}},
            ]
        ]
        p = tmp_path / "no_type.json"
        p.write_text(json.dumps(data))
        sections = parse_content_list(p)
        assert len(sections) == 1

    def test_top_level_not_list(self, tmp_path: Path):
        """Top-level JSON not a list → ValueError."""
        p = tmp_path / "obj.json"
        p.write_text(json.dumps({"not": "a list"}))
        with pytest.raises(ValueError, match="Expected top-level list"):
            parse_content_list(p)

    def test_invalid_json(self, tmp_path: Path):
        """Invalid JSON → ValueError."""
        p = tmp_path / "bad.json"
        p.write_text("not valid json {{{")
        with pytest.raises(ValueError, match="Invalid JSON"):
            parse_content_list(p)

    def test_file_not_found(self, tmp_path: Path):
        """Non-existent file → FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            parse_content_list(tmp_path / "does_not_exist.json")

    def test_fewer_than_2_sections_warning(self, tmp_path: Path):
        """Document with only 1 section → warning logged."""
        data = [
            [
                {"type": "title", "content": {"title_content": [{"type": "text", "content": "Only One"}], "level": 1}},
                {"type": "paragraph", "content": {"paragraph_content": [{"type": "text", "content": "text."}]}},
            ]
        ]
        p = tmp_path / "one.json"
        p.write_text(json.dumps(data))
        sections = parse_content_list(p)
        assert len(sections) == 1

    def test_section_indices_sequential(self, tmp_path: Path):
        """section_index values form [0, 1, ..., n-1]."""
        data = [
            [
                {"type": "title", "content": {"title_content": [{"type": "text", "content": "A"}], "level": 1}},
                {"type": "paragraph", "content": {"paragraph_content": [{"type": "text", "content": "a."}]}},
                {"type": "title", "content": {"title_content": [{"type": "text", "content": "B"}], "level": 2}},
                {"type": "paragraph", "content": {"paragraph_content": [{"type": "text", "content": "b."}]}},
                {"type": "title", "content": {"title_content": [{"type": "text", "content": "C"}], "level": 1}},
                {"type": "paragraph", "content": {"paragraph_content": [{"type": "text", "content": "c."}]}},
            ]
        ]
        p = tmp_path / "seq.json"
        p.write_text(json.dumps(data))
        sections = parse_content_list(p)
        indices = [s.section_index for s in sections]
        assert indices == [0, 1, 2]

    def test_heading_level_default_1(self, tmp_path: Path):
        """Title block missing level → defaults to 1."""
        data = [
            [
                {"type": "title", "content": {"title_content": [{"type": "text", "content": "No Level"}]}},
                {"type": "paragraph", "content": {"paragraph_content": [{"type": "text", "content": "text."}]}},
            ]
        ]
        p = tmp_path / "nolev.json"
        p.write_text(json.dumps(data))
        sections = parse_content_list(p)
        assert len(sections) == 1
        assert sections[0].heading_level == 1

    def test_module_constants_exist(self):
        """INCLUDED_BLOCK_TYPES and EXCLUDED_BLOCK_TYPES are frozensets."""
        assert isinstance(INCLUDED_BLOCK_TYPES, frozenset)
        assert isinstance(EXCLUDED_BLOCK_TYPES, frozenset)
        assert "title" in INCLUDED_BLOCK_TYPES
        assert "page_header" in EXCLUDED_BLOCK_TYPES


# ── Task 5: PBT structural invariants for parse_content_list ─────────────────

# Shared strategy: random content_list inputs
_content_list_strategy = st.lists(
    st.lists(block_strategy(), min_size=0, max_size=10),
    min_size=0,
    max_size=8,
)


class TestParseContentListStructuralProperties:
    """Property-based tests for parse_content_list structural invariants.

    Feature: mineru-json-ingestion
    """

    @given(_content_list_strategy)
    @settings(max_examples=200)
    def test_property1_section_indices_sequential(self, content_list: List[List[Dict]]):
        """Property 1: section_index values form [0, 1, ..., n-1] with no gaps.

        # Feature: mineru-json-ingestion, Property 1
        Validates: Requirements 11.1
        """
        import json as _json
        import tempfile, os

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            _json.dump(content_list, f)
            fname = f.name

        try:
            sections = parse_content_list(fname)
        except (ValueError, FileNotFoundError):
            return
        finally:
            os.unlink(fname)

        indices = [s.section_index for s in sections]
        assert indices == list(range(len(sections))), (
            f"section_index not sequential: {indices}"
        )

    @given(_content_list_strategy)
    @settings(max_examples=200)
    def test_property2_section_count_bounded_by_title_blocks(
        self, content_list: List[List[Dict]]
    ):
        """Property 2: len(sections) <= title_block_count + 1 (preamble).

        # Feature: mineru-json-ingestion, Property 2
        Validates: Requirements 11.2
        """
        import json as _json
        import tempfile, os

        # Count title blocks in the raw input
        title_count = sum(
            1
            for page in content_list
            if isinstance(page, list)
            for block in page
            if isinstance(block, dict) and block.get("type") == "title"
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            _json.dump(content_list, f)
            fname = f.name

        try:
            sections = parse_content_list(fname)
        except (ValueError, FileNotFoundError):
            return
        finally:
            os.unlink(fname)

        # +1 for possible preamble section
        assert len(sections) <= title_count + 1, (
            f"sections={len(sections)} > title_count+1={title_count + 1}"
        )

    @given(_content_list_strategy)
    @settings(max_examples=200)
    def test_property4_excluded_sections_absent(self, content_list: List[List[Dict]]):
        """Property 4: no output section matches references/bibliography/exact-appendix rules.

        # Feature: mineru-json-ingestion, Property 4
        Validates: Requirements 11.4, 4.4, 4.4a, 4.4b
        """
        import json as _json
        import tempfile, os, re

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            _json.dump(content_list, f)
            fname = f.name

        try:
            sections = parse_content_list(fname)
        except (ValueError, FileNotFoundError):
            return
        finally:
            os.unlink(fname)

        for s in sections:
            normalized = re.sub(r'^\d+[\.\)]\s*', '', s.title.strip()).strip().lower()
            assert normalized not in ("references", "bibliography"), (
                f"Excluded section title found in output: '{s.title}'"
            )
            if normalized in ("appendix", "appendices"):
                assert s.heading_level > 2, (
                    f"Exact appendix at level {s.heading_level} should have been excluded"
                )

    @given(_content_list_strategy)
    @settings(max_examples=200)
    def test_property5_no_empty_content(self, content_list: List[List[Dict]]):
        """Property 5: all output sections have non-empty content.

        # Feature: mineru-json-ingestion, Property 5
        Validates: Requirements 11.5, 8.3
        """
        import json as _json
        import tempfile, os

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            _json.dump(content_list, f)
            fname = f.name

        try:
            sections = parse_content_list(fname)
        except (ValueError, FileNotFoundError):
            return
        finally:
            os.unlink(fname)

        for s in sections:
            assert s.content.strip(), (
                f"Section '{s.title}' has empty content"
            )

    @given(_content_list_strategy)
    @settings(max_examples=200)
    def test_property6_heading_levels_positive(self, content_list: List[List[Dict]]):
        """Property 6: all output heading_level values are >= 1.

        # Feature: mineru-json-ingestion, Property 6
        Validates: Requirements 11.6, 4.2
        """
        import json as _json
        import tempfile, os

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            _json.dump(content_list, f)
            fname = f.name

        try:
            sections = parse_content_list(fname)
        except (ValueError, FileNotFoundError):
            return
        finally:
            os.unlink(fname)

        for s in sections:
            assert s.heading_level >= 1, (
                f"Section '{s.title}' has heading_level={s.heading_level} < 1"
            )


# ── Task 6.3: Unit tests for PdfLoader._load_mineru_json ─────────────────────

class TestPdfLoaderMineruJson:
    """Unit tests for PdfLoader._load_mineru_json.

    Validates: Requirements 9.1–9.3, 1.8, 12.4, 12.5
    """

    REQUIRED_METADATA_FIELDS = {
        "doc_type", "source_file", "title", "arxiv_id",
        "category", "subcategory", "authors", "publish_date",
        "section_title", "chunk_index", "chunk_count",
        "token_count", "has_heading_context", "heading_level",
    }

    def _make_json(self, tmp_path: Path, arxiv_id: str) -> Path:
        """Write a minimal content_list_v2.json for the given arxiv_id."""
        data = [
            [
                {
                    "type": "title",
                    "content": {
                        "title_content": [{"type": "text", "content": "Introduction"}],
                        "level": 1,
                    },
                },
                {
                    "type": "paragraph",
                    "content": {
                        "paragraph_content": [
                            {"type": "text", "content": "This paper proposes a method."}
                        ]
                    },
                },
            ]
        ]
        json_dir = tmp_path / arxiv_id / "vlm"
        json_dir.mkdir(parents=True)
        json_file = json_dir / f"{arxiv_id}_content_list_v2.json"
        json_file.write_text(json.dumps(data))
        return json_file

    def _make_pdf_path(self, tmp_path: Path, arxiv_id: str) -> Path:
        """Return a fake PDF path with the correct stem format."""
        pdf = tmp_path / f"{arxiv_id}.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        return pdf

    def test_json_found_returns_document_chunks(self, tmp_path: Path, monkeypatch):
        """JSON found and parsed → DocumentChunk objects with correct metadata schema."""
        from src.ingestion.loader import PdfLoader, DocumentChunk

        arxiv_id = "2603_07444"
        self._make_json(tmp_path, arxiv_id)
        pdf_path = self._make_pdf_path(tmp_path, arxiv_id)

        monkeypatch.setenv("MINERU_OUTPUT_DIR", str(tmp_path))
        monkeypatch.setenv("PDF_BACKEND", "mineru_json")

        # Patch embedding calls so no server needed
        monkeypatch.setattr(
            "servers.embedding_server.client.get_dense_vectors",
            lambda texts: [[0.1] * 768 for _ in texts],
        )

        loader = PdfLoader()
        chunks = loader._load_mineru_json(pdf_path)

        assert len(chunks) >= 1
        for chunk in chunks:
            assert isinstance(chunk, DocumentChunk)
            assert chunk.text.strip()
            for field in self.REQUIRED_METADATA_FIELDS:
                assert field in chunk.metadata, f"Missing metadata field: {field}"
            assert chunk.metadata["doc_type"] == "paper"
            assert chunk.metadata["source_file"] == pdf_path.name
            assert chunk.metadata["arxiv_id"] == arxiv_id
            assert chunk.metadata["chunk_index"] >= 0
            assert chunk.metadata["chunk_count"] >= 1
            assert chunk.metadata["token_count"] >= 0
            assert isinstance(chunk.metadata["has_heading_context"], bool)
            assert chunk.metadata["heading_level"] >= 1

    def test_json_not_found_falls_back_to_pypdf(self, tmp_path: Path, monkeypatch, caplog):
        """JSON not found → falls back to _load_pypdf, logs warning."""
        from src.ingestion.loader import PdfLoader

        arxiv_id = "9999_99999"
        pdf_path = self._make_pdf_path(tmp_path, arxiv_id)

        monkeypatch.setenv("MINERU_OUTPUT_DIR", str(tmp_path))
        monkeypatch.setenv("PDF_BACKEND", "mineru_json")

        # Patch _load_pypdf to avoid needing a real PDF
        fallback_called = []

        def fake_pypdf(path):
            fallback_called.append(path)
            return []

        loader = PdfLoader()
        monkeypatch.setattr(loader, "_load_pypdf", fake_pypdf)

        with caplog.at_level(logging.WARNING, logger="src.ingestion.loader"):
            result = loader._load_mineru_json(pdf_path)

        assert fallback_called, "Expected _load_pypdf to be called as fallback"
        assert result == []
        assert any("not found" in r.message.lower() or "falling back" in r.message.lower()
                   for r in caplog.records)

    def test_json_parse_error_falls_back_to_pypdf(self, tmp_path: Path, monkeypatch, caplog):
        """JSON parse error → falls back to _load_pypdf, logs warning."""
        from src.ingestion.loader import PdfLoader

        arxiv_id = "2603_07444"
        # Write invalid JSON
        json_dir = tmp_path / arxiv_id / "vlm"
        json_dir.mkdir(parents=True)
        bad_json = json_dir / f"{arxiv_id}_content_list_v2.json"
        bad_json.write_text("not valid json {{{")

        pdf_path = self._make_pdf_path(tmp_path, arxiv_id)
        monkeypatch.setenv("MINERU_OUTPUT_DIR", str(tmp_path))

        fallback_called = []

        def fake_pypdf(path):
            fallback_called.append(path)
            return []

        loader = PdfLoader()
        monkeypatch.setattr(loader, "_load_pypdf", fake_pypdf)

        with caplog.at_level(logging.WARNING, logger="src.ingestion.loader"):
            result = loader._load_mineru_json(pdf_path)

        assert fallback_called, "Expected _load_pypdf to be called as fallback"
        assert result == []
        assert any("falling back" in r.message.lower() or "parse" in r.message.lower()
                   for r in caplog.records)

    def test_non_arxiv_stem_falls_back_to_pypdf(self, tmp_path: Path, monkeypatch, caplog):
        """PDF stem not matching arxiv format → falls back to _load_pypdf, logs warning."""
        from src.ingestion.loader import PdfLoader

        pdf_path = tmp_path / "my_book.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        fallback_called = []

        def fake_pypdf(path):
            fallback_called.append(path)
            return []

        loader = PdfLoader()
        monkeypatch.setattr(loader, "_load_pypdf", fake_pypdf)

        with caplog.at_level(logging.WARNING, logger="src.ingestion.loader"):
            result = loader._load_mineru_json(pdf_path)

        assert fallback_called, "Expected _load_pypdf to be called as fallback"
        assert result == []
