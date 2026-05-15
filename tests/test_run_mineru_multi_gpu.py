"""Unit tests for multi-GPU MinerU orchestration.

Feature: mineru-multi-gpu-conversion
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from scripts.run_mineru_multi_gpu import (
    DEFAULT_PARSE_METHOD,
    GPUDetection,
    MinerUProcess,
    _extract_content_list,
    _copy_marked_pdf_if_available,
    create_marked_pdf_from_content_list,
    assign_documents,
    build_mineru_command,
    build_worker_env,
    convert_pdf_via_mineru_api,
    detect_gpus,
    discover_pdfs,
    output_marked_pdf_path,
    output_json_path,
    parse_api_extra,
    port_for_gpu,
    process_worker_documents,
    start_mineru_processes,
    ensure_marked_pdf,
)


class _FakeCuda:
    def __init__(self, available: bool, count: int):
        self._available = available
        self._count = count

    def is_available(self):
        return self._available

    def device_count(self):
        return self._count


class _FakeTorch:
    def __init__(self, available: bool, count: int):
        self.cuda = _FakeCuda(available, count)


class _FakeProc:
    def __init__(self, pid: int = 12345, running: bool = True):
        self.pid = pid
        self._running = running
        self.terminated = False
        self.killed = False

    def poll(self):
        return None if self._running else 0

    def wait(self, timeout=None):
        self._running = False
        return 0

    def terminate(self):
        self.terminated = True
        self._running = False

    def kill(self):
        self.killed = True
        self._running = False


def test_detect_gpus_multiple_cuda_devices():
    detection = detect_gpus(_FakeTorch(available=True, count=4))
    assert detection.gpu_count == 4
    assert detection.cuda_available is True
    assert "Detected 4 CUDA GPU" in detection.reason


def test_detect_gpus_zero_or_unavailable_falls_back_to_one():
    zero = detect_gpus(_FakeTorch(available=True, count=0))
    unavailable = detect_gpus(_FakeTorch(available=False, count=8))

    assert zero.gpu_count == 1
    assert zero.cuda_available is False
    assert unavailable.gpu_count == 1
    assert unavailable.cuda_available is False


def test_detect_gpus_exception_falls_back_to_one():
    class BrokenTorch:
        cuda = object()

    detection = detect_gpus(BrokenTorch())
    assert detection.gpu_count == 1
    assert detection.cuda_available is False
    assert "GPU detection failed" in detection.reason


def test_assign_documents_uses_source_index_mod_gpu_count():
    docs = [Path(f"doc_{i}.pdf") for i in range(10)]
    assignments = assign_documents(docs, gpu_count=3)

    assert assignments == [
        [Path("doc_0.pdf"), Path("doc_3.pdf"), Path("doc_6.pdf"), Path("doc_9.pdf")],
        [Path("doc_1.pdf"), Path("doc_4.pdf"), Path("doc_7.pdf")],
        [Path("doc_2.pdf"), Path("doc_5.pdf"), Path("doc_8.pdf")],
    ]


def test_assign_documents_requires_positive_gpu_count():
    with pytest.raises(ValueError, match="gpu_count"):
        assign_documents([Path("a.pdf")], gpu_count=0)


def test_port_for_gpu_offsets_base_port():
    assert port_for_gpu(8010, 0) == 8010
    assert port_for_gpu(8010, 1) == 8011
    assert port_for_gpu(8010, 7) == 8017


def test_discover_pdfs_skips_hidden_appledouble_files(tmp_path):
    visible = tmp_path / "2207_01206.pdf"
    hidden = tmp_path / "._2207_01206.pdf"
    other = tmp_path / "notes.txt"
    visible.write_bytes(b"%PDF-1.4 visible")
    hidden.write_bytes(b"AppleDouble metadata, not a real PDF")
    other.write_text("not a pdf", encoding="utf-8")

    assert discover_pdfs(tmp_path) == [visible]


def test_build_mineru_command_appends_extra_args():
    cmd = build_mineru_command(
        "mineru-api",
        "0.0.0.0",
        8020,
        extra_args=["--log-level", "debug"],
    )
    assert cmd == ["mineru-api", "--host", "0.0.0.0", "--port", "8020", "--log-level", "debug"]


def test_build_worker_env_sets_cuda_visible_devices_when_cuda_available():
    env = build_worker_env({"PATH": "/bin", "CUDA_VISIBLE_DEVICES": "old"}, 2, True)
    assert env["PATH"] == "/bin"
    assert env["CUDA_VISIBLE_DEVICES"] == "2"


def test_build_worker_env_sets_mineru_api_output_root_per_worker(tmp_path):
    env = build_worker_env(
        {"PATH": "/bin"},
        gpu_index=3,
        cuda_available=True,
        mineru_api_output_root=tmp_path / "api-output",
    )

    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    assert env["MINERU_API_OUTPUT_ROOT"] == str(tmp_path / "api-output" / "worker_3")
    assert (tmp_path / "api-output" / "worker_3").is_dir()


def test_build_worker_env_does_not_inject_cuda_mask_for_cpu_fallback():
    env = build_worker_env({"PATH": "/bin"}, 0, False)
    assert env == {"PATH": "/bin"}


def test_start_mineru_processes_starts_one_process_per_gpu_with_port_and_env():
    calls = []

    def fake_popen(cmd, env, stdout, stderr, start_new_session):
        calls.append({
            "cmd": cmd,
            "env": env,
            "stdout": stdout,
            "stderr": stderr,
            "start_new_session": start_new_session,
        })
        return _FakeProc(pid=12000 + len(calls))

    ready_urls = []

    workers = start_mineru_processes(
        gpu_count=2,
        detection=GPUDetection(gpu_count=2, cuda_available=True, reason="test"),
        host="127.0.0.1",
        connect_host="127.0.0.1",
        base_port=8100,
        mineru_command="mineru-api",
        startup_timeout=5,
        popen_factory=fake_popen,
        wait_for_ready=lambda url, timeout: ready_urls.append((url, timeout)),
    )

    assert [w.gpu_index for w in workers] == [0, 1]
    assert [w.port for w in workers] == [8100, 8101]
    assert [w.url for w in workers] == ["http://127.0.0.1:8100", "http://127.0.0.1:8101"]
    assert calls[0]["cmd"] == ["mineru-api", "--host", "127.0.0.1", "--port", "8100"]
    assert calls[1]["cmd"] == ["mineru-api", "--host", "127.0.0.1", "--port", "8101"]
    assert calls[0]["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert calls[1]["env"]["CUDA_VISIBLE_DEVICES"] == "1"
    assert calls[0]["start_new_session"] is True
    assert ready_urls == [("http://127.0.0.1:8100", 5), ("http://127.0.0.1:8101", 5)]


def test_parse_api_extra_parses_repeated_key_values():
    assert parse_api_extra(["parse_method=ocr", "return_layout=true"]) == {
        "parse_method": "ocr",
        "return_layout": "true",
    }


def test_parse_api_extra_rejects_malformed_values():
    with pytest.raises(ValueError, match="KEY=VALUE"):
        parse_api_extra(["parse_method"])


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"content_list": [{"type": "text"}]}, [{"type": "text"}]),
        ({"content_list": '[{"type":"text","text":"from-json-string"}]'}, [{"type": "text", "text": "from-json-string"}]),
        ({"content_list_v2": [{"type": "table"}]}, [{"type": "table"}]),
        ({"results": [{"content_list": [{"type": "image"}]}]}, [{"type": "image"}]),
        (
            {"backend": "vlm-transformers", "version": "2.5", "results": {"2010_03768": {"content_list": '[{"type":"text"}]'}}},
            [{"type": "text"}],
        ),
        ([{"type": "text", "text": "hello"}], [{"type": "text", "text": "hello"}]),
    ],
)
def test_extract_content_list_handles_common_response_shapes(payload, expected):
    assert _extract_content_list(payload) == expected


def test_convert_pdf_via_mineru_api_handles_actual_mineru_results_shape(tmp_path):
    pdf = tmp_path / "2010_03768.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    output_dir = tmp_path / "mineru_output"

    resp = MagicMock()
    resp.status_code = 200
    resp.text = '{"backend":"vlm-transformers","results":{"2010_03768":{"content_list":"[...]"}}}'
    resp.json.return_value = {
        "task_id": "abc123",
        "status": "completed",
        "backend": "vlm-transformers",
        "version": "2.5",
        "results": {
            "2010_03768": {
                "content_list": '[{"type":"text","text":"paper text"}]',
            }
        },
    }

    with patch("scripts.run_mineru_multi_gpu.requests.post", return_value=resp):
        written = convert_pdf_via_mineru_api(
            pdf_path=pdf,
            api_url="http://127.0.0.1:8010",
            output_dir=output_dir,
            timeout=30,
        )

    assert written == output_json_path(output_dir, pdf)
    assert json.loads(written.read_text(encoding="utf-8")) == [
        {"type": "text", "text": "paper text"}
    ]


def test_copy_marked_pdf_if_available_normalizes_layout_pdf(tmp_path):
    pdf = tmp_path / "2010_03768.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    output_dir = tmp_path / "mineru_output"
    raw_marked = (
        output_dir
        / "_mineru_api"
        / "worker_0"
        / "task-123"
        / "2010_03768"
        / "hybrid_auto"
        / "2010_03768_layout.pdf"
    )
    raw_marked.parent.mkdir(parents=True)
    raw_marked.write_bytes(b"%PDF-1.4 marked layout")

    copied = _copy_marked_pdf_if_available(
        output_dir=output_dir,
        pdf_path=pdf,
        response_data={"task_id": "task-123"},
    )

    assert copied == output_marked_pdf_path(output_dir, pdf)
    assert copied.read_bytes() == b"%PDF-1.4 marked layout"


def test_convert_pdf_via_mineru_api_writes_json_and_copies_marked_pdf(tmp_path):
    pdf = tmp_path / "2010_03768.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    output_dir = tmp_path / "mineru_output"
    raw_marked = (
        output_dir
        / "_mineru_api"
        / "worker_1"
        / "task-456"
        / "2010_03768"
        / "hybrid_auto"
        / "2010_03768_layout.pdf"
    )
    raw_marked.parent.mkdir(parents=True)
    raw_marked.write_bytes(b"%PDF-1.4 marked layout")

    resp = MagicMock()
    resp.status_code = 200
    resp.text = "{}"
    resp.json.return_value = {
        "task_id": "task-456",
        "results": {
            "2010_03768": {
                "content_list": '[{"type":"text","text":"paper text"}]',
            }
        },
    }

    with patch("scripts.run_mineru_multi_gpu.requests.post", return_value=resp):
        written = convert_pdf_via_mineru_api(
            pdf_path=pdf,
            api_url="http://127.0.0.1:8010",
            output_dir=output_dir,
            timeout=30,
        )

    assert written == output_json_path(output_dir, pdf)
    assert output_marked_pdf_path(output_dir, pdf).read_bytes() == b"%PDF-1.4 marked layout"


def test_create_marked_pdf_from_content_list_renders_bbox_pdf(tmp_path):
    pytest.importorskip("pypdfium2")
    pytest.importorskip("PIL")
    from PIL import Image

    source_pdf = tmp_path / "source.pdf"
    image = Image.new("RGB", (200, 260), "white")
    image.save(source_pdf, "PDF")
    content_list = [[
        {"type": "title", "bbox": [10, 10, 180, 40]},
        {"type": "paragraph", "bbox": [20, 60, 170, 120]},
    ]]
    marked_pdf = tmp_path / "marked.pdf"

    result = create_marked_pdf_from_content_list(source_pdf, content_list, marked_pdf)

    assert result == marked_pdf
    assert marked_pdf.exists()
    assert marked_pdf.stat().st_size > 0


def test_ensure_marked_pdf_is_nonfatal_when_renderer_raises(tmp_path):
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    output_dir = tmp_path / "mineru_output"

    with patch(
        "scripts.run_mineru_multi_gpu.create_marked_pdf_from_content_list",
        side_effect=KeyError("JPEG"),
    ):
        result = ensure_marked_pdf(
            output_dir=output_dir,
            pdf_path=pdf,
            content_list=[[{"type": "paragraph", "bbox": [1, 2, 3, 4]}]],
        )

    assert result is None


def test_convert_pdf_via_mineru_api_writes_canonical_content_list_json(tmp_path):
    pdf = tmp_path / "2401_00001.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    output_dir = tmp_path / "mineru_output"

    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"content_list": [{"type": "text", "text": "hello"}]}

    with patch("scripts.run_mineru_multi_gpu.requests.post", return_value=resp) as post:
        written = convert_pdf_via_mineru_api(
            pdf_path=pdf,
            api_url="http://127.0.0.1:8010",
            output_dir=output_dir,
            timeout=30,
            extra_form_fields={"return_layout": "true"},
        )

    assert written == output_json_path(output_dir, pdf)
    assert json.loads(written.read_text(encoding="utf-8")) == [{"type": "text", "text": "hello"}]
    _, kwargs = post.call_args
    assert kwargs["data"]["return_content_list"] == "true"
    assert kwargs["data"]["return_md"] == "false"
    assert kwargs["data"]["output_dir"] == str(output_dir)
    assert kwargs["data"]["parse_method"] == DEFAULT_PARSE_METHOD
    assert kwargs["data"]["return_layout"] == "true"


def test_convert_pdf_via_mineru_api_allows_explicit_parse_method(tmp_path):
    pdf = tmp_path / "2401_00003.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    output_dir = tmp_path / "mineru_output"

    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"content_list": []}

    with patch("scripts.run_mineru_multi_gpu.requests.post", return_value=resp) as post:
        convert_pdf_via_mineru_api(
            pdf_path=pdf,
            api_url="http://127.0.0.1:8010",
            output_dir=output_dir,
            timeout=30,
            parse_method="ocr",
        )

    _, kwargs = post.call_args
    assert kwargs["data"]["parse_method"] == "ocr"


def test_convert_pdf_via_mineru_api_uses_existing_server_written_output(tmp_path):
    pdf = tmp_path / "2401_00002.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    output_dir = tmp_path / "mineru_output"
    expected = output_json_path(output_dir, pdf)
    expected.parent.mkdir(parents=True)
    expected.write_text("[]", encoding="utf-8")

    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"md_content": "# markdown only"}

    with patch("scripts.run_mineru_multi_gpu.requests.post", return_value=resp):
        written = convert_pdf_via_mineru_api(
            pdf_path=pdf,
            api_url="http://127.0.0.1:8010",
            output_dir=output_dir,
            timeout=30,
        )

    assert written == expected


def test_convert_pdf_via_mineru_api_normalizes_server_written_noncanonical_output(tmp_path):
    pdf = tmp_path / "2401_00004.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    output_dir = tmp_path / "mineru_output"
    server_written = output_dir / "2401_00004" / "auto" / "2401_00004_content_list_v2.json"
    server_written.parent.mkdir(parents=True)
    server_written.write_text('[{"type":"text"}]', encoding="utf-8")

    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"md_content": "# markdown only"}

    with patch("scripts.run_mineru_multi_gpu.requests.post", return_value=resp):
        written = convert_pdf_via_mineru_api(
            pdf_path=pdf,
            api_url="http://127.0.0.1:8010",
            output_dir=output_dir,
            timeout=30,
        )

    canonical = output_json_path(output_dir, pdf)
    assert written == canonical
    assert canonical.read_text(encoding="utf-8") == '[{"type":"text"}]'


def test_process_worker_documents_skips_existing_and_counts_failures(tmp_path):
    docs = [tmp_path / "a.pdf", tmp_path / "b.pdf", tmp_path / "c.pdf"]
    for doc in docs:
        doc.write_bytes(b"%PDF-1.4 fake")

    output_dir = tmp_path / "mineru_output"
    existing = output_json_path(output_dir, docs[0])
    existing.parent.mkdir(parents=True)
    existing.write_text("[]", encoding="utf-8")
    existing_marked = output_marked_pdf_path(output_dir, docs[0])
    existing_marked.write_bytes(b"%PDF-1.4 marked")

    def fake_converter(pdf_path: Path, api_url: str, output_dir: Path, timeout: int, parse_method: str = DEFAULT_PARSE_METHOD, extra_form_fields=None):
        assert parse_method == DEFAULT_PARSE_METHOD
        if pdf_path.name == "c.pdf":
            raise RuntimeError("boom")
        out = output_json_path(output_dir, pdf_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("[]", encoding="utf-8")
        return out

    worker = MinerUProcess(
        worker_id=1,
        gpu_index=1,
        port=8011,
        url="http://127.0.0.1:8011",
        process=_FakeProc(),
    )

    result = process_worker_documents(
        worker=worker,
        documents=docs,
        output_dir=output_dir,
        timeout=30,
        force=False,
        converter=fake_converter,
    )

    assert result.total == 3
    assert result.skipped == 1
    assert result.converted == 1
    assert result.failed == 1


def test_process_worker_documents_restarts_and_retries_on_connection_error(tmp_path):
    pdf = tmp_path / "retry_me.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    output_dir = tmp_path / "mineru_output"
    calls = []

    def fake_converter(pdf_path: Path, api_url: str, output_dir: Path, timeout: int, parse_method: str = DEFAULT_PARSE_METHOD, extra_form_fields=None):
        calls.append(api_url)
        if len(calls) == 1:
            raise requests.exceptions.ConnectionError("connection refused")
        out = output_json_path(output_dir, pdf_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("[]", encoding="utf-8")
        return out

    old_worker = MinerUProcess(
        worker_id=0,
        gpu_index=0,
        port=8010,
        url="http://127.0.0.1:8010",
        process=_FakeProc(pid=1),
    )
    new_worker = MinerUProcess(
        worker_id=0,
        gpu_index=0,
        port=8010,
        url="http://127.0.0.1:8010",
        process=_FakeProc(pid=2),
    )
    restarts = []

    def fake_restart(worker: MinerUProcess) -> MinerUProcess:
        restarts.append(worker.pid if hasattr(worker, "pid") else worker.process.pid)
        return new_worker

    result = process_worker_documents(
        worker=old_worker,
        documents=[pdf],
        output_dir=output_dir,
        timeout=30,
        force=False,
        require_marked_pdf=False,
        converter=fake_converter,
        restart_worker=fake_restart,
        max_retries_per_doc=1,
    )

    assert result.total == 1
    assert result.converted == 1
    assert result.failed == 0
    assert len(calls) == 2
    assert restarts == [1]