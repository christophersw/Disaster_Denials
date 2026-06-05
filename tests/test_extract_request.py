"""Tests that the request is shaped correctly, without calling the API."""

import base64

from pda.extract import MODEL, build_request


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
