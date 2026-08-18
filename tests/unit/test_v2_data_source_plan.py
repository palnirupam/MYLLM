from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from scripts.validate_v2_data_sources import load_plan, validate_plan


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "configs" / "dhruva_v2_data_sources.yaml"


def test_v2_source_catalog_is_immutable_and_balanced():
    payload = load_plan(PLAN)
    assert validate_plan(payload, require_complete=False) == []


def test_v2_source_catalog_does_not_admit_blocked_code_slice():
    payload = load_plan(PLAN)
    failures = validate_plan(payload, require_complete=True)
    assert any("python_edu_pending_license_filter" in failure for failure in failures)
