"""Structural tests for the extraction JSON Schema."""

from pda.schema import PdaReport, json_schema


def test_schema_forbids_extra_properties_on_all_objects():
    """Structured outputs require additionalProperties:false on every object."""
    schema = json_schema()

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False, node
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(schema)


def test_report_has_counties_list_and_core_fields():
    fields = PdaReport.model_fields
    for name in ("report_outcome", "decision_date", "ia_requested",
                 "pa_requested", "counties", "needs_review"):
        assert name in fields


def test_legacy_and_modern_ia_fields_coexist():
    fields = PdaReport.model_fields
    for name in ("ia_pct_poverty", "ia_pct_ssi", "ia_pct_low_income", "ia_pct_elderly"):
        assert name in fields
