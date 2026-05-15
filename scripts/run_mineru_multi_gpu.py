#!/usr/bin/env python3
"""Run MinerU PDF conversion across all detected GPUs.

This script is intentionally separate from ``scripts/ingest_mineru_json.py``.
It performs only the MinerU PDF -> JSON conversion step and writes outputs in
the layout consumed by the existing JSON ingestion pipeline::

    mineru_output/<pdf-stem>/vlm/<pdf-stem>_content_list_v2.json

Design constraints:
  - Detect CUDA GPU count at startup with a safe CPU/single-worker fallback.
  - Start one ``mineru-api`` subprocess per detected GPU.
  - Pin each subprocess to its assigned GPU using ``CUDA_VISIBLE_DEVICES``.
  - Assign source PDFs statically by ``document_index % gpu_count``.
  - Process each worker's assigned PDFs sequentially, in bounded memory.
  - Log per-document success/failure; restart costs only failed/current docs.

Example:
    python scripts/run_mineru_multi_gpu.py \
        --pdf-dir ./downloads \
        --output-dir ./mineru_output \
        --base-port 8010
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import requests


log = logging.getLogger(__name__)

DEFAULT_PDF_DIR = "./downloads"
DEFAULT_OUTPUT_DIR = "./mineru_output"
DEFAULT_MINERU_COMMAND = "mineru-api"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_BASE_PORT = 8010
DEFAULT_PARSE_METHOD = "auto"
DEFAULT_MARKED_PDF = True
DEFAULT_STARTUP_TIMEOUT_SECONDS = 180
DEFAULT_REQUEST_TIMEOUT_SECONDS = 900


@dataclass(frozen=True)
class GPUDetection:
    """CUDA detection result used to size MinerU worker pool."""

    gpu_count: int
    cuda_available: bool
    reason: str


@dataclass(frozen=True)
class MinerUProcess:
    """Metadata for one launched MinerU API subprocess."""

    worker_id: int
    gpu_index: int
    port: int
    url: str
    process: subprocess.Popen


@dataclass(frozen=True)
class WorkerResult:
    """Per-worker conversion summary."""

    worker_id: int
    gpu_index: int
    total: int
    converted: int
    skipped: int
    failed: int


def detect_gpus(torch_module: Any = None) -> GPUDetection:
    """Detect available CUDA GPUs with a single-worker CPU fallback.

    Args:
        torch_module: Optional injected torch-like object for tests.

    Returns:
        GPUDetection. ``gpu_count`` is always at least 1.
    """
    try:
        torch = torch_module
        if torch is None:
            import torch as torch  # type: ignore[no-redef]

        cuda = getattr(torch, "cuda", None)
        if cuda is None or not cuda.is_available():
            return GPUDetection(
                gpu_count=1,
                cuda_available=False,
                reason="CUDA unavailable; using single CPU/single-worker fallback",
            )

        count = int(cuda.device_count())
        if count <= 0:
            return GPUDetection(
                gpu_count=1,
                cuda_available=False,
                reason="CUDA reported 0 GPUs; using single CPU/single-worker fallback",
            )

        return GPUDetection(
            gpu_count=count,
            cuda_available=True,
            reason=f"Detected {count} CUDA GPU(s)",
        )
    except Exception as exc:
        return GPUDetection(
            gpu_count=1,
            cuda_available=False,
            reason=f"GPU detection failed ({exc}); using single CPU/single-worker fallback",
        )


def discover_pdfs(pdf_dir: Path, limit: Optional[int] = None) -> List[Path]:
    """Discover source PDFs in stable order."""
    pdfs = sorted(
        p for p in pdf_dir.glob("*.pdf")
        if p.is_file() and not p.name.startswith(".")
    )
    if limit is not None:
        pdfs = pdfs[:limit]
    return pdfs


def assign_documents(documents: Sequence[Path], gpu_count: int) -> List[List[Path]]:
    """Assign documents by static round-robin: document index modulo GPU count."""
    if gpu_count < 1:
        raise ValueError("gpu_count must be >= 1")

    assignments: List[List[Path]] = [[] for _ in range(gpu_count)]
    for doc_index, document in enumerate(documents):
        assignments[doc_index % gpu_count].append(document)
    return assignments


def port_for_gpu(base_port: int, gpu_index: int) -> int:
    """Return the MinerU API port for a GPU index."""
    if gpu_index < 0:
        raise ValueError("gpu_index must be >= 0")
    return base_port + gpu_index


def output_json_path(output_dir: Path, pdf_path: Path) -> Path:
    """Canonical content_list_v2 path consumed by ingest_mineru_json.py."""
    stem = pdf_path.stem
    return output_dir / stem / "vlm" / f"{stem}_content_list_v2.json"


def output_marked_pdf_path(output_dir: Path, pdf_path: Path) -> Path:
    """Canonical marked-layout PDF path for human inspection."""
    stem = pdf_path.stem
    return output_dir / stem / "vlm" / f"{stem}_layout.pdf"


def build_mineru_command(
    mineru_command: str,
    host: str,
    port: int,
    extra_args: Optional[Sequence[str]] = None,
) -> List[str]:
    """Build the mineru-api subprocess command."""
    cmd = [mineru_command, "--host", host, "--port", str(port)]
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def build_worker_env(
    base_env: Mapping[str, str],
    gpu_index: int,
    cuda_available: bool,
    mineru_api_output_root: Optional[Path] = None,
) -> Dict[str, str]:
    """Build environment for a MinerU subprocess.

    If CUDA is available, the subprocess is pinned to exactly one physical GPU.
    With ``CUDA_VISIBLE_DEVICES=<gpu_index>``, MinerU should see that GPU as its
    local ``cuda:0``.  In CPU fallback mode, no CUDA mask is injected.
    """
    env = dict(base_env)
    if cuda_available:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    if mineru_api_output_root is not None:
        worker_output_root = mineru_api_output_root / f"worker_{gpu_index}"
        worker_output_root.mkdir(parents=True, exist_ok=True)
        env["MINERU_API_OUTPUT_ROOT"] = str(worker_output_root)
    return env


def wait_for_mineru_api(url: str, timeout: int) -> None:
    """Wait until a MinerU API process responds on its docs endpoint."""
    deadline = time.time() + timeout
    docs_url = f"{url.rstrip('/')}/docs"
    last_error: Optional[Exception] = None

    while time.time() < deadline:
        try:
            resp = requests.get(docs_url, timeout=2)
            if resp.status_code < 500:
                return
        except Exception as exc:  # service still starting
            last_error = exc
        time.sleep(1)

    raise TimeoutError(
        f"MinerU API at {url} did not become ready within {timeout}s"
        + (f"; last error: {last_error}" if last_error else "")
    )


def start_mineru_processes(
    gpu_count: int,
    detection: GPUDetection,
    host: str,
    connect_host: str,
    base_port: int,
    mineru_command: str,
    startup_timeout: int,
    mineru_api_output_root: Optional[Path] = None,
    extra_args: Optional[Sequence[str]] = None,
    popen_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
    wait_for_ready: Callable[[str, int], None] = wait_for_mineru_api,
) -> List[MinerUProcess]:
    """Start one MinerU API subprocess per worker/GPU."""
    processes: List[MinerUProcess] = []
    try:
        for gpu_index in range(gpu_count):
            worker = start_mineru_process(
                worker_id=gpu_index,
                gpu_index=gpu_index,
                detection=detection,
                host=host,
                connect_host=connect_host,
                base_port=base_port,
                mineru_command=mineru_command,
                startup_timeout=startup_timeout,
                mineru_api_output_root=mineru_api_output_root,
                extra_args=extra_args,
                popen_factory=popen_factory,
                wait_for_ready=wait_for_ready,
            )
            processes.append(worker)
        return processes
    except Exception:
        terminate_mineru_processes(processes)
        raise


def start_mineru_process(
    worker_id: int,
    gpu_index: int,
    detection: GPUDetection,
    host: str,
    connect_host: str,
    base_port: int,
    mineru_command: str,
    startup_timeout: int,
    mineru_api_output_root: Optional[Path] = None,
    extra_args: Optional[Sequence[str]] = None,
    popen_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
    wait_for_ready: Callable[[str, int], None] = wait_for_mineru_api,
) -> MinerUProcess:
    """Start one MinerU API subprocess for a specific worker/GPU."""
    port = port_for_gpu(base_port, gpu_index)
    url = f"http://{connect_host}:{port}"
    cmd = build_mineru_command(mineru_command, host, port, extra_args)
    env = build_worker_env(
        os.environ,
        gpu_index,
        detection.cuda_available,
        mineru_api_output_root=mineru_api_output_root,
    )

    cuda_mask = env.get("CUDA_VISIBLE_DEVICES", "<unset>")
    api_output_root = env.get("MINERU_API_OUTPUT_ROOT", "<unset>")
    log.info(
        "Starting MinerU worker %d | gpu_index=%d | port=%d | CUDA_VISIBLE_DEVICES=%s | MINERU_API_OUTPUT_ROOT=%s | cmd=%s",
        worker_id,
        gpu_index,
        port,
        cuda_mask,
        api_output_root,
        " ".join(cmd),
    )
    proc = popen_factory(
        cmd,
        env=env,
        stdout=None,
        stderr=None,
        start_new_session=True,
    )
    worker = MinerUProcess(
        worker_id=worker_id,
        gpu_index=gpu_index,
        port=port,
        url=url,
        process=proc,
    )
    wait_for_ready(url, startup_timeout)
    log.info("MinerU worker %d ready at %s", worker_id, url)
    return worker


def restart_mineru_process(
    worker: MinerUProcess,
    detection: GPUDetection,
    host: str,
    connect_host: str,
    base_port: int,
    mineru_command: str,
    startup_timeout: int,
    mineru_api_output_root: Optional[Path] = None,
    extra_args: Optional[Sequence[str]] = None,
    popen_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
    wait_for_ready: Callable[[str, int], None] = wait_for_mineru_api,
) -> MinerUProcess:
    """Terminate and restart a dead/unhealthy MinerU worker on the same GPU/port."""
    log.warning(
        "Restarting MinerU worker %d on gpu_index=%d port=%d",
        worker.worker_id,
        worker.gpu_index,
        worker.port,
    )
    terminate_mineru_processes([worker], timeout=10)
    return start_mineru_process(
        worker_id=worker.worker_id,
        gpu_index=worker.gpu_index,
        detection=detection,
        host=host,
        connect_host=connect_host,
        base_port=base_port,
        mineru_command=mineru_command,
        startup_timeout=startup_timeout,
        mineru_api_output_root=mineru_api_output_root,
        extra_args=extra_args,
        popen_factory=popen_factory,
        wait_for_ready=wait_for_ready,
    )


def terminate_mineru_processes(processes: Iterable[MinerUProcess], timeout: int = 20) -> None:
    """Terminate all launched MinerU subprocesses."""
    procs = list(processes)
    for worker in procs:
        proc = worker.process
        if proc.poll() is None:
            log.info("Terminating MinerU worker %d (pid=%s)", worker.worker_id, proc.pid)
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:
                proc.terminate()

    deadline = time.time() + timeout
    for worker in procs:
        proc = worker.process
        remaining = max(0.1, deadline - time.time())
        try:
            proc.wait(timeout=remaining)
        except Exception:
            if proc.poll() is None:
                log.warning("Killing MinerU worker %d (pid=%s)", worker.worker_id, proc.pid)
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    proc.kill()


def parse_api_extra(values: Optional[Sequence[str]]) -> Dict[str, str]:
    """Parse repeated KEY=VALUE form-field options."""
    result: Dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Expected KEY=VALUE for --api-extra, got: {value!r}")
        key, val = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Empty KEY in --api-extra value: {value!r}")
        result[key] = val
    return result


def _extract_content_list(data: Any) -> Optional[Any]:
    """Extract content-list JSON from common MinerU response shapes.

    MinerU API response shapes vary across releases. In the version shown by
    the user's logs, ``/file_parse`` accepts the request but does not write a
    server-side file at our requested path, so the content list must be found in
    the HTTP response. Support direct keys, list-valued results, and dict-valued
    results keyed by filename.
    """
    if isinstance(data, str):
        stripped = data.strip()
        if stripped.startswith("[") or stripped.startswith("{"):
            try:
                return _extract_content_list(json.loads(stripped))
            except Exception:
                return None
        return None

    if isinstance(data, dict):
        for key in (
            "content_list",
            "content_list_v2",
            "content_list_json",
            "*_content_list_v2.json",
        ):
            if key in data:
                value = data[key]
                if isinstance(value, str):
                    parsed = _extract_content_list(value)
                    return parsed if parsed is not None else value
                return value

        # Some MinerU versions return: {"results": {"file.pdf": {...}}}
        results = data.get("results")
        if isinstance(results, dict):
            for value in results.values():
                extracted = _extract_content_list(value)
                if extracted is not None:
                    return extracted

        # Some versions return nested payloads under data/result/output keys.
        for key in ("data", "result", "output"):
            if key in data:
                extracted = _extract_content_list(data[key])
                if extracted is not None:
                    return extracted

        results = data.get("results")
        if isinstance(results, list) and results:
            for item in results:
                extracted = _extract_content_list(item)
                if extracted is not None:
                    return extracted

        # Last resort: recursively inspect nested dict/list values for any
        # content_list key without assuming a particular top-level schema.
        for value in data.values():
            if isinstance(value, (dict, list)):
                extracted = _extract_content_list(value)
                if extracted is not None:
                    return extracted

    if isinstance(data, list):
        # MinerU may return the content list directly as a JSON array.
        if not data:
            return []
        if all(isinstance(item, dict) for item in data):
            return data
        return _extract_content_list(data[0])

    return None


def _find_existing_output(output_dir: Path, pdf_path: Path) -> Optional[Path]:
    """Find a MinerU-written content_list_v2 file for a PDF, if present."""
    expected = output_json_path(output_dir, pdf_path)
    if expected.exists():
        return expected

    stem = pdf_path.stem
    matches = sorted(output_dir.glob(f"**/{stem}_content_list_v2.json"))
    return matches[0] if matches else None


def _ensure_canonical_output(output_dir: Path, pdf_path: Path, existing: Path) -> Path:
    """Copy a MinerU-written output to the ingest-compatible canonical path.

    MinerU 3.x parse methods are ``auto``, ``ocr``, and ``txt``. Depending on
    version/configuration, server-side artifacts may be written under a
    parse-method-specific directory rather than ``vlm``. The downstream
    ``ingest_mineru_json.py`` script discovers ``**/vlm/*_content_list_v2.json``
    or flat files, so normalize any server-written JSON into the existing
    canonical ``vlm`` path.
    """
    canonical = output_json_path(output_dir, pdf_path)
    if existing == canonical:
        return canonical

    canonical.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(existing, canonical)
    log.info("Normalized MinerU output %s -> %s", existing, canonical)
    return canonical


def _iter_content_items(content_list: Any) -> Iterable[Tuple[int, Dict[str, Any]]]:
    """Yield ``(page_index, item)`` pairs from MinerU content_list variants."""
    if not isinstance(content_list, list):
        return

    for page_index, page in enumerate(content_list):
        if isinstance(page, list):
            for item in page:
                if isinstance(item, dict):
                    yield page_index, item
        elif isinstance(page, dict):
            item_page = page.get("page_idx", page.get("page", page_index))
            try:
                item_page_index = int(item_page)
            except Exception:
                item_page_index = page_index
            yield item_page_index, page


def _bbox_bounds_by_page(content_list: Any) -> Dict[int, Tuple[float, float]]:
    """Return max bbox x/y values per page for coordinate scaling."""
    bounds: Dict[int, Tuple[float, float]] = {}
    for page_index, item in _iter_content_items(content_list):
        bbox = item.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        try:
            max_x = float(max(bbox[0], bbox[2]))
            max_y = float(max(bbox[1], bbox[3]))
        except Exception:
            continue
        old_x, old_y = bounds.get(page_index, (0.0, 0.0))
        bounds[page_index] = (max(old_x, max_x), max(old_y, max_y))
    return bounds


def create_marked_pdf_from_content_list(
    source_pdf: Path,
    content_list: Any,
    output_pdf: Path,
) -> Optional[Path]:
    """Create a human-inspection PDF with MinerU bboxes drawn on each page.

    This avoids relying on MinerU private CLI flags. The marked PDF is rendered
    from the original PDF pages plus the bboxes already present in
    ``content_list_v2.json``. It is intended for inspection/debugging, not for
    downstream ingestion.
    """
    items = list(_iter_content_items(content_list))
    if not items:
        log.warning("No content_list items found for marked PDF: %s", source_pdf)
        return None

    try:
        import pypdfium2 as pdfium
        from PIL import Image, ImageDraw, ImageFont
        # Ensure PDF/JPEG save handlers are registered in minimal/lazy Pillow
        # environments. Without this, Image.save(..., "PDF") can raise
        # KeyError('JPEG') even though Pillow is installed.
        import PIL.JpegImagePlugin  # noqa: F401
        import PIL.PdfImagePlugin  # noqa: F401
        Image.init()
    except Exception as exc:
        log.warning("Cannot create marked PDF; missing rendering dependency: %s", exc)
        return None

    type_colors = {
        "title": (255, 0, 0),
        "paragraph": (0, 120, 255),
        "image": (0, 170, 0),
        "table": (180, 0, 180),
        "equation": (255, 140, 0),
    }
    default_color = (255, 215, 0)

    pdf = pdfium.PdfDocument(str(source_pdf))
    bounds = _bbox_bounds_by_page(content_list)
    font = ImageFont.load_default()
    rendered_pages = []
    scale = 2

    try:
        for page_index in range(len(pdf)):
            page = pdf[page_index]
            bitmap = page.render(scale=scale)
            image = bitmap.to_pil().convert("RGB")
            draw = ImageDraw.Draw(image, "RGBA")
            width, height = image.size
            max_x, max_y = bounds.get(page_index, (0.0, 0.0))
            coord_width = max(width / scale, max_x, 1.0)
            coord_height = max(height / scale, max_y, 1.0)
            sx = width / coord_width
            sy = height / coord_height

            for _item_page, item in (pair for pair in items if pair[0] == page_index):
                bbox = item.get("bbox")
                if not (isinstance(bbox, list) and len(bbox) == 4):
                    continue
                try:
                    x0, y0, x1, y1 = [float(v) for v in bbox]
                except Exception:
                    continue
                x0, x1 = sorted((x0 * sx, x1 * sx))
                y0, y1 = sorted((y0 * sy, y1 * sy))
                item_type = str(item.get("type", "unknown"))
                color = type_colors.get(item_type, default_color)
                draw.rectangle(
                    [x0, y0, x1, y1],
                    outline=(*color, 255),
                    width=3,
                )
                label = item_type[:18]
                label_y = max(0, y0 - 12)
                draw.rectangle([x0, label_y, x0 + 7 * len(label) + 4, label_y + 12], fill=(*color, 180))
                draw.text((x0 + 2, label_y), label, fill=(0, 0, 0), font=font)

            rendered_pages.append(image)
    finally:
        pdf.close()

    if not rendered_pages:
        return None

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    try:
        rendered_pages[0].save(
            output_pdf,
            "PDF",
            save_all=True,
            append_images=rendered_pages[1:],
            resolution=144,
        )
    except Exception as exc:
        log.warning("Failed to save marked PDF %s: %s", output_pdf, exc)
        return None
    log.info("Created marked layout PDF %s", output_pdf)
    return output_pdf


def _find_existing_marked_pdf(
    output_dir: Path,
    pdf_path: Path,
    response_data: Any = None,
) -> Optional[Path]:
    """Find MinerU's marked layout PDF for one source document, if present.

    MinerU writes this file as ``<stem>_layout.pdf`` when its internal
    ``f_draw_layout_bbox`` option is enabled. The API writes raw task artifacts
    under ``MINERU_API_OUTPUT_ROOT``; this script sets that to
    ``<output_dir>/_mineru_api/worker_<N>`` for new runs, but also checks the
    default ``./output`` location to recover artifacts from runs started before
    this script controlled the output root.
    """
    stem = pdf_path.stem
    filename = f"{stem}_layout.pdf"
    canonical = output_marked_pdf_path(output_dir, pdf_path)
    if canonical.exists():
        return canonical

    task_id = response_data.get("task_id") if isinstance(response_data, dict) else None
    roots: List[Path] = [output_dir / "_mineru_api", output_dir, Path("output")]
    patterns: List[str] = []
    if task_id:
        patterns.append(f"**/{task_id}/**/{filename}")
    patterns.append(f"**/{filename}")

    candidates: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            candidates.extend(p for p in root.glob(pattern) if p.is_file())

    if not candidates:
        return None

    return max(candidates, key=lambda p: p.stat().st_mtime)


def _copy_marked_pdf_if_available(
    output_dir: Path,
    pdf_path: Path,
    response_data: Any = None,
) -> Optional[Path]:
    """Copy MinerU's marked layout PDF to the canonical output folder."""
    existing = _find_existing_marked_pdf(output_dir, pdf_path, response_data)
    if existing is None:
        log.warning(
            "No marked layout PDF found for %s. Ensure mineru-api was started with f_draw_layout_bbox=true.",
            pdf_path.name,
        )
        return None

    canonical = output_marked_pdf_path(output_dir, pdf_path)
    if existing == canonical:
        return canonical

    canonical.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(existing, canonical)
    log.info("Normalized MinerU marked PDF %s -> %s", existing, canonical)
    return canonical


def ensure_marked_pdf(
    output_dir: Path,
    pdf_path: Path,
    content_list: Any,
    response_data: Any = None,
) -> Optional[Path]:
    """Ensure canonical marked PDF exists, copying MinerU output or rendering locally."""
    try:
        copied = _copy_marked_pdf_if_available(output_dir, pdf_path, response_data)
        if copied is not None:
            return copied
        return create_marked_pdf_from_content_list(
            source_pdf=pdf_path,
            content_list=content_list,
            output_pdf=output_marked_pdf_path(output_dir, pdf_path),
        )
    except Exception as exc:
        log.warning(
            "Marked PDF generation failed for %s, but JSON conversion remains valid: %s",
            pdf_path.name,
            exc,
        )
        return None


def convert_pdf_via_mineru_api(
    pdf_path: Path,
    api_url: str,
    output_dir: Path,
    timeout: int,
    parse_method: str = DEFAULT_PARSE_METHOD,
    extra_form_fields: Optional[Mapping[str, str]] = None,
) -> Path:
    """Convert one PDF through a MinerU API instance.

    The request asks MinerU to produce a content list and also supplies
    ``output_dir`` for MinerU versions that write artifacts server-side.  If the
    content list is returned in the HTTP response, this function writes it to
    the canonical path used by ``ingest_mineru_json.py``.  Otherwise, it verifies
    that MinerU created a matching file under ``output_dir``.
    """
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    url = f"{api_url.rstrip('/')}/file_parse"
    form: Dict[str, str] = {
        "parse_method": parse_method,
        "return_content_list": "true",
        "return_md": "false",
        "output_dir": str(output_dir),
    }
    if extra_form_fields:
        form.update(dict(extra_form_fields))

    with pdf_path.open("rb") as f:
        resp = requests.post(
            url,
            files={"files": (pdf_path.name, f, "application/pdf")},
            data=form,
            timeout=timeout,
        )

    if resp.status_code != 200:
        raise RuntimeError(
            f"MinerU returned HTTP {resp.status_code} for {pdf_path}: {resp.text[:500]}"
        )

    response_excerpt = resp.text[:1000]
    try:
        data = resp.json()
    except Exception:
        data = None

    content_list = _extract_content_list(data)
    if content_list is not None:
        out_path = output_json_path(output_dir, pdf_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(content_list, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        ensure_marked_pdf(output_dir, pdf_path, content_list, data)
        return out_path

    existing = _find_existing_output(output_dir, pdf_path)
    if existing is not None:
        out_path = _ensure_canonical_output(output_dir, pdf_path, existing)
        try:
            existing_content = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            existing_content = None
        ensure_marked_pdf(output_dir, pdf_path, existing_content, data)
        return out_path

    if isinstance(data, dict):
        response_shape = f"top-level keys={list(data.keys())}"
    elif isinstance(data, list):
        response_shape = f"top-level list len={len(data)}"
    else:
        response_shape = f"non-JSON or unrecognized JSON type={type(data).__name__}"

    raise RuntimeError(
        f"MinerU succeeded for {pdf_path}, but no content_list_v2 JSON was returned "
        f"or found under {output_dir}. Response shape: {response_shape}. "
        f"Response excerpt: {response_excerpt}"
    )


def process_worker_documents(
    worker: MinerUProcess,
    documents: Sequence[Path],
    output_dir: Path,
    timeout: int,
    parse_method: str = DEFAULT_PARSE_METHOD,
    force: bool = False,
    require_marked_pdf: bool = DEFAULT_MARKED_PDF,
    extra_form_fields: Optional[Mapping[str, str]] = None,
    converter: Callable[..., Path] = convert_pdf_via_mineru_api,
    restart_worker: Optional[Callable[[MinerUProcess], MinerUProcess]] = None,
    max_retries_per_doc: int = 1,
) -> WorkerResult:
    """Process one worker's assigned documents sequentially."""
    converted = 0
    skipped = 0
    failed = 0
    current_worker = worker

    for local_index, pdf_path in enumerate(documents, start=1):
        out_path = output_json_path(output_dir, pdf_path)
        marked_path = output_marked_pdf_path(output_dir, pdf_path)
        has_required_outputs = out_path.exists() and (
            not require_marked_pdf or marked_path.exists()
        )
        if has_required_outputs and not force:
            skipped += 1
            log.info(
                "[Worker %d] [%d/%d] required output(s) already exist for %s — skipping",
                worker.worker_id,
                local_index,
                len(documents),
                pdf_path.name,
            )
            continue

        start = time.time()
        attempts = 0
        while True:
            if current_worker.process.poll() is not None and restart_worker is not None:
                log.warning(
                    "[Worker %d] MinerU process exited before %s; restarting before request",
                    current_worker.worker_id,
                    pdf_path.name,
                )
                current_worker = restart_worker(current_worker)

            try:
                written = converter(
                    pdf_path=pdf_path,
                    api_url=current_worker.url,
                    output_dir=output_dir,
                    timeout=timeout,
                    parse_method=parse_method,
                    extra_form_fields=extra_form_fields,
                )
                converted += 1
                log.info(
                    "[Worker %d] [%d/%d] converted %s -> %s in %.1fs",
                    current_worker.worker_id,
                    local_index,
                    len(documents),
                    pdf_path.name,
                    written,
                    time.time() - start,
                )
                break
            except requests.exceptions.ConnectionError as exc:
                if restart_worker is not None and attempts < max_retries_per_doc:
                    attempts += 1
                    log.warning(
                        "[Worker %d] [%d/%d] connection failed for %s (attempt %d/%d): %s — restarting MinerU and retrying this document",
                        current_worker.worker_id,
                        local_index,
                        len(documents),
                        pdf_path.name,
                        attempts,
                        max_retries_per_doc,
                        exc,
                    )
                    current_worker = restart_worker(current_worker)
                    continue

                failed += 1
                log.exception(
                    "[Worker %d] [%d/%d] FAILED %s after connection retry budget exhausted: %s",
                    current_worker.worker_id,
                    local_index,
                    len(documents),
                    pdf_path,
                    exc,
                )
                break
            except Exception as exc:
                failed += 1
                log.exception(
                    "[Worker %d] [%d/%d] FAILED %s: %s",
                    current_worker.worker_id,
                    local_index,
                    len(documents),
                    pdf_path,
                    exc,
                )
                break

    return WorkerResult(
        worker_id=worker.worker_id,
        gpu_index=worker.gpu_index,
        total=len(documents),
        converted=converted,
        skipped=skipped,
        failed=failed,
    )


def run_conversion(args: argparse.Namespace) -> List[WorkerResult]:
    """Run the full multi-GPU MinerU conversion orchestration."""
    pdf_dir = Path(args.pdf_dir)
    output_dir = Path(args.output_dir)
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    detection = detect_gpus()
    log.info("%s", detection.reason)
    gpu_count = detection.gpu_count

    documents = discover_pdfs(pdf_dir, args.limit)
    log.info("Discovered %d PDF(s) in %s", len(documents), pdf_dir)
    if not documents:
        return []

    assignments = assign_documents(documents, gpu_count)
    for worker_id, docs in enumerate(assignments):
        log.info(
            "Worker %d assigned %d PDF(s) by document_index %% gpu_count",
            worker_id,
            len(docs),
        )

    extra_form_fields = parse_api_extra(args.api_extra)
    mineru_extra_args = args.mineru_extra_arg or []
    mineru_api_output_root = output_dir / "_mineru_api"

    processes = start_mineru_processes(
        gpu_count=gpu_count,
        detection=detection,
        host=args.host,
        connect_host=args.connect_host,
        base_port=args.base_port,
        mineru_command=args.mineru_command,
        startup_timeout=args.startup_timeout,
        mineru_api_output_root=mineru_api_output_root,
        extra_args=mineru_extra_args,
    )

    try:
        results: List[WorkerResult] = []
        process_lock = threading.Lock()

        def restart_worker(old_worker: MinerUProcess) -> MinerUProcess:
            new_worker = restart_mineru_process(
                worker=old_worker,
                detection=detection,
                host=args.host,
                connect_host=args.connect_host,
                base_port=args.base_port,
                mineru_command=args.mineru_command,
                startup_timeout=args.startup_timeout,
                mineru_api_output_root=mineru_api_output_root,
                extra_args=mineru_extra_args,
            )
            with process_lock:
                for idx, existing in enumerate(processes):
                    if existing.worker_id == old_worker.worker_id:
                        processes[idx] = new_worker
                        break
                else:
                    processes.append(new_worker)
            return new_worker

        with concurrent.futures.ThreadPoolExecutor(max_workers=gpu_count) as executor:
            future_to_worker = {
                executor.submit(
                    process_worker_documents,
                    worker,
                    assignments[worker.worker_id],
                    output_dir,
                    args.request_timeout,
                    args.parse_method,
                    args.force,
                    args.marked_pdf,
                    extra_form_fields,
                    convert_pdf_via_mineru_api,
                    restart_worker,
                ): worker
                for worker in processes
            }

            for future in concurrent.futures.as_completed(future_to_worker):
                worker = future_to_worker[future]
                result = future.result()
                results.append(result)
                log.info(
                    "Worker %d complete: total=%d converted=%d skipped=%d failed=%d",
                    worker.worker_id,
                    result.total,
                    result.converted,
                    result.skipped,
                    result.failed,
                )

        results.sort(key=lambda r: r.worker_id)
        return results
    finally:
        terminate_mineru_processes(processes)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PDFs with one MinerU API subprocess per detected GPU.",
    )
    parser.add_argument("--pdf-dir", default=DEFAULT_PDF_DIR, help="Directory containing source PDFs.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="MinerU output directory.")
    parser.add_argument("--limit", type=int, default=None, help="Max PDFs to process.")
    parser.add_argument("--base-port", type=int, default=DEFAULT_BASE_PORT, help="Port for GPU 0; GPU i uses base-port+i.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host passed to mineru-api --host.")
    parser.add_argument(
        "--connect-host",
        default=DEFAULT_HOST,
        help="Host the parent process uses for HTTP calls. Useful if --host is 0.0.0.0.",
    )
    parser.add_argument("--mineru-command", default=DEFAULT_MINERU_COMMAND, help="MinerU API executable.")
    parser.add_argument(
        "--parse-method",
        default=DEFAULT_PARSE_METHOD,
        choices=("auto", "ocr", "txt"),
        help="MinerU parse_method form field. Defaults to 'auto'.",
    )
    parser.add_argument(
        "--mineru-extra-arg",
        action="append",
        default=[],
        help="Extra argument appended to each mineru-api command. Repeat for multiple args.",
    )
    parser.add_argument(
        "--no-marked-pdf",
        dest="marked_pdf",
        action="store_false",
        default=DEFAULT_MARKED_PDF,
        help="Do not request/copy MinerU's marked layout PDF. By default, JSON plus <stem>_layout.pdf are preserved.",
    )
    parser.add_argument(
        "--api-extra",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra form field sent to /file_parse. Repeat for multiple fields.",
    )
    parser.add_argument(
        "--startup-timeout",
        type=int,
        default=DEFAULT_STARTUP_TIMEOUT_SECONDS,
        help="Seconds to wait for each MinerU API subprocess to become ready.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        help="Seconds to allow each PDF conversion request.",
    )
    parser.add_argument("--force", action="store_true", help="Re-run PDFs even when canonical output JSON exists.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args(argv)

    try:
        results = run_conversion(args)
    except Exception as exc:
        log.exception("MinerU multi-GPU conversion failed: %s", exc)
        return 1

    total = sum(r.total for r in results)
    converted = sum(r.converted for r in results)
    skipped = sum(r.skipped for r in results)
    failed = sum(r.failed for r in results)

    log.info("=" * 60)
    log.info("FINAL SUMMARY")
    log.info("  PDFs total:     %d", total)
    log.info("  PDFs converted: %d", converted)
    log.info("  PDFs skipped:   %d", skipped)
    log.info("  PDFs failed:    %d", failed)
    log.info("=" * 60)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())