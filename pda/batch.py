"""
Title: PDA Batch Orchestration (Batches API)
Description: Submit PDA extractions through the Anthropic Message Batches API
    (50% cheaper, asynchronous) and collect the results into the SQLite store.
    Two resumable phases:
      submit  — build one request per not-yet-done, not-in-flight PDF (reusing
                pda.extract.build_request), chunk under the 256 MB batch cap,
                create the batch(es), and record custom_id↔source_pdf so the
                results can be matched back later.
      collect — for each open batch that has ended, stream results, validate
                each succeeded message into a PdaReport, flatten, and write it
                transactionally; mark errored/expired results failed.
    custom_ids are a deterministic hash of the PDF path, so resubmitting a
    failed PDF reuses its row and the mapping survives process restarts.
Changelog:
    2026-06-05  Initial version.
"""

import hashlib

from pda.db import (
    done_source_pdfs,
    mark_batch_item,
    open_batch_ids,
    pending_source_pdfs,
    record_batch_items,
    source_pdf_for,
    write_report,
)
from pda.extract import MODEL, build_request, report_from_message
from pda.flatten import flatten

# Stay comfortably under the Batches API's 256 MB per-batch limit; base64 PDFs
# plus per-request JSON (schema, system prompt) add overhead beyond the encoded
# bytes we measure, so leave headroom.
DEFAULT_MAX_BYTES = 180 * 1024 * 1024
DEFAULT_MAX_COUNT = 100_000  # Batches API caps a batch at 100k requests


def custom_id_for(source_pdf: str) -> str:
    """Return a deterministic, API-valid custom_id for a PDF path.

    The Batches API requires custom_ids to match ^[a-zA-Z0-9_-]{1,64}$, which a
    raw path (slashes, dots) violates. A hash of the path is stable across runs
    so resubmitting the same PDF reuses its batch_items row.

    Args:
        source_pdf: The PDF path used as the report key.
    Returns:
        A custom_id like "pda-<sha1 hex>".
    """
    return "pda-" + hashlib.sha1(source_pdf.encode("utf-8")).hexdigest()


def build_batch_request(custom_id: str, pdf_bytes: bytes) -> dict:
    """Wrap one extraction request in the Batches API request envelope.

    Args:
        custom_id: The request's custom_id.
        pdf_bytes: Raw PDF bytes.
    Returns:
        {"custom_id": ..., "params": <messages.create kwargs>}.
    """
    return {"custom_id": custom_id, "params": build_request(pdf_bytes)}


def chunk_by_size(items: list, max_bytes: int, max_count: int, size_of) -> list:
    """Greedily group items into chunks bounded by total size and count.

    Args:
        items: The items to group, in order.
        max_bytes: Maximum summed size_of(...) per chunk.
        max_count: Maximum number of items per chunk.
        size_of: Callable returning an item's size in bytes.
    Returns:
        A list of chunks (lists of items). An item larger than max_bytes still
        gets its own chunk rather than being dropped.
    """
    chunks: list = []
    current: list = []
    current_bytes = 0
    for item in items:
        size = size_of(item)
        if current and (current_bytes + size > max_bytes
                        or len(current) >= max_count):
            chunks.append(current)
            current = []
            current_bytes = 0
        current.append(item)
        current_bytes += size
    if current:
        chunks.append(current)
    return chunks


def _read_file(path: str) -> bytes:
    """Read a file's bytes."""
    with open(path, "rb") as handle:
        return handle.read()


def submit(client, conn, pdf_paths: list[str], *,
           max_bytes: int = DEFAULT_MAX_BYTES,
           max_count: int = DEFAULT_MAX_COUNT,
           read_bytes=_read_file) -> list[str]:
    """Submit batches for PDFs that are neither done nor already in flight.

    Args:
        client: An Anthropic client (uses client.messages.batches.create).
        conn: An open SQLite connection from pda.db.connect().
        pdf_paths: Candidate PDF paths.
        max_bytes: Per-batch size budget (encoded PDF bytes).
        max_count: Per-batch request count cap.
        read_bytes: Injection seam for reading a path's bytes (tests).
    Returns:
        The list of created batch ids (empty if nothing to submit).
    """
    skip = done_source_pdfs(conn) | pending_source_pdfs(conn)
    todo = [path for path in pdf_paths if path not in skip]

    sized = []
    for path in todo:
        request = build_batch_request(custom_id_for(path), read_bytes(path))
        encoded = request["params"]["messages"][0]["content"][0]["source"]["data"]
        sized.append((path, request, len(encoded)))

    batch_ids: list[str] = []
    for chunk in chunk_by_size(sized, max_bytes, max_count,
                               size_of=lambda item: item[2]):
        batch = client.messages.batches.create(
            requests=[request for _path, request, _size in chunk])
        record_batch_items(
            conn, batch.id,
            [(request["custom_id"], path) for path, request, _size in chunk])
        batch_ids.append(batch.id)
    return batch_ids


def collect(client, conn, provenance_lookup, *, now, on_result=None
            ) -> tuple[int, int]:
    """Collect results from ended batches into the store.

    For each open batch whose processing has ended, stream results: validate
    and write each succeeded report transactionally, mark errored/expired ones
    failed. Batches still running are left untouched (their items stay pending).

    Args:
        client: An Anthropic client (batches.retrieve / batches.results).
        conn: An open SQLite connection from pda.db.connect().
        provenance_lookup: Callable source_pdf -> provenance dict for flatten.
        now: Callable returning the extracted_at timestamp string.
        on_result: Optional callback (source_pdf, status, detail) for logging.
    Returns:
        (written_count, failed_count).
    """
    def notify(source_pdf: str, status: str, detail: str = "") -> None:
        if on_result is not None:
            on_result(source_pdf, status, detail)

    written = 0
    failed = 0
    for batch_id in open_batch_ids(conn):
        batch = client.messages.batches.retrieve(batch_id)
        if getattr(batch, "processing_status", None) != "ended":
            continue
        for result in client.messages.batches.results(batch_id):
            source_pdf = source_pdf_for(conn, result.custom_id)
            if source_pdf is None:
                continue  # a custom_id we don't recognize — skip
            outcome = result.result.type
            if outcome == "succeeded":
                try:
                    report = report_from_message(result.result.message)
                    meta = {"parser_model": MODEL, "extracted_at": now()}
                    report_row, county_rows = flatten(
                        report, source_pdf, provenance_lookup(source_pdf), meta)
                    write_report(conn, report_row, county_rows)
                    mark_batch_item(conn, result.custom_id, "written")
                    written += 1
                    notify(source_pdf, "written")
                except Exception as error:  # noqa: BLE001 — one bad result won't stop the rest
                    mark_batch_item(conn, result.custom_id, "failed")
                    failed += 1
                    notify(source_pdf, "failed", str(error))
            else:
                detail = getattr(getattr(result.result, "error", None), "type", outcome)
                mark_batch_item(conn, result.custom_id, "failed")
                failed += 1
                notify(source_pdf, "failed", str(detail))
    return written, failed
