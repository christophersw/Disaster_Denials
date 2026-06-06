"""Tests for the Batches API orchestration, using a fake batches client."""

import re
from types import SimpleNamespace

from pda.batch import (
    build_batch_request,
    chunk_by_size,
    collect,
    custom_id_for,
    submit,
)
from pda.db import (
    connect,
    done_source_pdfs,
    open_batch_ids,
    pending_source_pdfs,
    record_batch_items,
    write_report,
)
from pda.extract import TOOL_NAME

CUSTOM_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")  # Batches API constraint


# --- fakes -----------------------------------------------------------------

class _FakeBatches:
    """Stand-in for client.messages.batches (create / retrieve / results)."""

    def __init__(self, results_by_batch=None, status="ended"):
        self.created = []  # (batch_id, requests) for assertions
        self._n = 0
        self._results = results_by_batch or {}
        self._status = status

    def create(self, requests):
        self._n += 1
        batch_id = f"batch_{self._n}"
        self.created.append((batch_id, requests))
        return SimpleNamespace(id=batch_id)

    def retrieve(self, batch_id):
        return SimpleNamespace(id=batch_id, processing_status=self._status)

    def results(self, batch_id):
        return iter(self._results.get(batch_id, []))


def _client(batches):
    return SimpleNamespace(messages=SimpleNamespace(batches=batches))


def _valid_payload():
    """A minimal payload that validates into a PdaReport."""
    fields = (
        "report_outcome", "decision_date", "jurisdiction_name", "state_abbr",
        "requestor_type", "requestor_name", "incident_name", "incident_begin",
        "incident_end", "request_date", "disaster_number", "declaration_type",
        "denial_reason", "original_denial_date", "appeal_date",
        "pa_categories_requested", "ia_residences_total", "ia_destroyed",
        "ia_major", "ia_minor", "ia_affected", "ia_pct_insured",
        "ia_pct_flood_insured", "ia_pct_poverty", "ia_pct_ssi", "ia_pct_snap",
        "ia_pct_ownership", "ia_unemployment", "ia_pct_age_65_plus",
        "ia_pct_age_18_under", "ia_pct_disability", "ia_icc_ratio",
        "ia_pct_low_income", "ia_pct_elderly", "ia_cost_estimate",
        "pa_primary_impact", "pa_cost_estimate", "pa_statewide_per_capita",
        "pa_statewide_per_capita_indicator", "pa_countywide_per_capita_indicator",
        "review_note",
    )
    payload = {field: None for field in fields}
    payload.update({
        "report_outcome": "Declared", "ia_requested": True, "pa_requested": True,
        "hm_requested": False, "counties": [], "needs_review": False,
    })
    return payload


def _succeeded(custom_id):
    message = SimpleNamespace(
        content=[SimpleNamespace(
            type="tool_use", name=TOOL_NAME, input=_valid_payload())],
        stop_reason="end_turn")
    return SimpleNamespace(
        custom_id=custom_id,
        result=SimpleNamespace(type="succeeded", message=message))


def _errored(custom_id):
    return SimpleNamespace(
        custom_id=custom_id,
        result=SimpleNamespace(
            type="errored", error=SimpleNamespace(type="invalid_request")))


def _provenance(_source_pdf):
    return {"report_type": "Denials", "url": "http://x", "posted_date": None}


# --- pure helpers ----------------------------------------------------------

def test_custom_id_is_deterministic_and_valid():
    a = custom_id_for("data/pdfs/Denials/2024/x.pdf")
    assert a == custom_id_for("data/pdfs/Denials/2024/x.pdf")
    assert a != custom_id_for("data/pdfs/Denials/2024/y.pdf")
    assert CUSTOM_ID_RE.match(a)


def test_chunk_by_size_splits_on_bytes_and_count():
    items = [("a", 100), ("b", 100), ("c", 100)]
    by_bytes = chunk_by_size(items, max_bytes=250, max_count=99,
                             size_of=lambda it: it[1])
    assert [len(c) for c in by_bytes] == [2, 1]
    by_count = chunk_by_size(items, max_bytes=10_000, max_count=2,
                             size_of=lambda it: it[1])
    assert [len(c) for c in by_count] == [2, 1]


def test_build_batch_request_wraps_params_with_custom_id():
    req = build_batch_request("pda-abc", b"%PDF fake")
    assert req["custom_id"] == "pda-abc"
    assert req["params"]["model"]  # the extract.build_request payload


# --- submit ----------------------------------------------------------------

def _write_pdf(tmp_path, name, data=b"%PDF-1.4 fake"):
    path = tmp_path / name
    path.write_bytes(data)
    return str(path)


def test_submit_skips_done_and_pending_then_batches_the_rest(tmp_path):
    conn = connect(str(tmp_path / "pda.db"))
    done = _write_pdf(tmp_path, "done.pdf")
    write_report(conn, {"source_pdf": done}, [])  # already extracted
    pending = _write_pdf(tmp_path, "pending.pdf")
    record_batch_items(conn, "old", [(custom_id_for(pending), pending)])
    fresh1 = _write_pdf(tmp_path, "fresh1.pdf")
    fresh2 = _write_pdf(tmp_path, "fresh2.pdf")

    batches = _FakeBatches()
    # max_count=1 forces one batch per fresh PDF.
    ids = submit(_client(batches), conn, [done, pending, fresh1, fresh2],
                 max_count=1)

    assert len(ids) == 2  # only the two fresh PDFs, one per batch
    assert pending_source_pdfs(conn) == {pending, fresh1, fresh2}
    submitted = {pair[0]
                 for _bid, reqs in batches.created for pair in [(r["custom_id"],) for r in reqs]}
    assert custom_id_for(fresh1) in submitted
    assert custom_id_for(done) not in submitted


# --- collect ---------------------------------------------------------------

def test_collect_writes_succeeded_and_marks_errored(tmp_path):
    conn = connect(str(tmp_path / "pda.db"))
    good, bad = "good.pdf", "bad.pdf"
    record_batch_items(conn, "batch_x",
                       [(custom_id_for(good), good), (custom_id_for(bad), bad)])

    batches = _FakeBatches(results_by_batch={"batch_x": [
        _succeeded(custom_id_for(good)),
        _errored(custom_id_for(bad)),
    ]})
    ok, failed = collect(_client(batches), conn, _provenance,
                         now=lambda: "2026-06-05T00:00:00Z")

    assert (ok, failed) == (1, 1)
    assert done_source_pdfs(conn) == {good}          # the report row landed
    assert pending_source_pdfs(conn) == set()        # both resolved
    assert open_batch_ids(conn) == []                # batch closed


def test_collect_leaves_unfinished_batches_untouched(tmp_path):
    conn = connect(str(tmp_path / "pda.db"))
    record_batch_items(conn, "batch_x", [(custom_id_for("a.pdf"), "a.pdf")])
    batches = _FakeBatches(
        results_by_batch={"batch_x": [_succeeded(custom_id_for("a.pdf"))]},
        status="in_progress")

    ok, failed = collect(_client(batches), conn, _provenance,
                         now=lambda: "2026-06-05T00:00:00Z")

    assert (ok, failed) == (0, 0)
    assert pending_source_pdfs(conn) == {"a.pdf"}     # still in flight
    assert open_batch_ids(conn) == ["batch_x"]
