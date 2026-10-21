"""Behavioral tests for the LLM API visor's local usage-insights tracker."""

import sys


import pytest

sys.path.insert(0, "/a0")

from usr.plugins.a0_llm_api_visor.helpers import usage_tracker as ut


@pytest.fixture(autouse=True)
def _isolated_history(tmp_path, monkeypatch):
    monkeypatch.setattr(ut, "DATA_DIR", tmp_path)
    monkeypatch.setattr(ut, "HISTORY_FILE", tmp_path / "history.json")
    yield


def _snapshot(requests=5, tokens_in=1000, tokens_out=500, credits=1.5, epoch=1):
    return {
        "usage": {
            "user_usage": {
                "epoch": epoch,
                "requests": requests,
                "tokens_input": tokens_in,
                "tokens_output": tokens_out,
                "credits": credits,
            },
            "usageByModel": [],
        }
    }


def test_first_snapshot_creates_today_with_values():
    ut.record_snapshot(_snapshot())
    day = ut.load_history()["days"][max(ut.load_history()["days"])]
    assert day["requests"] == 5
    assert day["tokens_input"] == 1000
    assert day["tokens_output"] == 500
    assert day["credits"] == 1.5


def test_cumulative_max_never_regresses():
    ut.record_snapshot(_snapshot(requests=10, credits=2.0))
    ut.record_snapshot(_snapshot(requests=7, credits=1.0))  # older/stale cumulative view
    day = list(ut.load_history()["days"].values())[0]
    assert day["requests"] == 10
    assert day["credits"] == 2.0


def test_growth_accumulates():
    ut.record_snapshot(_snapshot(requests=10))
    ut.record_snapshot(_snapshot(requests=25))
    day = list(ut.load_history()["days"].values())[0]
    assert day["requests"] == 25


def test_invalid_payloads_are_noops():
    ut.record_snapshot(None)
    ut.record_snapshot({"usage": None})
    ut.record_snapshot({"usage": {"user_usage": "junk"}})
    ut.record_snapshot(_snapshot(requests=0, tokens_in=0, tokens_out=0))  # all-zero: skipped
    assert ut.load_history()["days"] == {}


def test_corrupt_history_file_resets_to_empty(tmp_path):
    ut.HISTORY_FILE.write_text("{not json")
    assert ut.load_history()["days"] == {}
    assert ut.load_history()["version"] == ut.SCHEMA_VERSION


def test_clear_history_removes_file():
    ut.record_snapshot(_snapshot())
    assert ut.HISTORY_FILE.exists()
    ut.clear_history()
    assert not ut.HISTORY_FILE.exists()
