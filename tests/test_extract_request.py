"""Tests that the request is shaped correctly, without calling the API."""

import base64

import pytest

from pda.extract import MODEL, build_request, extract_report


class _Block:
    """Minimal stand-in for a response content block."""

    def __init__(self, block_type: str, text: str | None = None):
        self.type = block_type
        self.text = text


class _Response:
    """Minimal stand-in for a Messages API response."""

    def __init__(self, content: list, stop_reason: str = "end_turn"):
        self.content = content
        self.stop_reason = stop_reason


class _FakeClient:
    """A client whose messages.create returns a preset response (no network)."""

    def __init__(self, response: _Response):
        self._response = response

        class _Messages:
            def create(_self, **_kwargs):
                return response

        self.messages = _Messages()


def test_build_request_has_pdf_document_and_schema():
    req = build_request(b"%PDF-1.4 fake bytes")
    assert req["model"] == MODEL
    assert req["thinking"] == {"type": "adaptive"}
    assert req["output_config"]["effort"] == "high"
    assert req["output_config"]["format"]["type"] == "json_schema"

    content = req["messages"][0]["content"]
    doc = next(b for b in content if b["type"] == "document")
    assert doc["source"]["media_type"] == "application/pdf"
    # data is base64 of the input bytes
    assert base64.b64decode(doc["source"]["data"]) == b"%PDF-1.4 fake bytes"


def test_system_prompt_is_cacheable():
    req = build_request(b"x")
    assert req["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_extract_report_raises_clear_error_when_no_text_block():
    """A response with no text block should raise a clear ValueError, not StopIteration."""
    response = _Response(content=[_Block("thinking")], stop_reason="refusal")
    with pytest.raises(ValueError, match="No text block"):
        extract_report(_FakeClient(response), b"%PDF fake")
