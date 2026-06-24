"""Deterministic tests for orchestration/preset.py (ADR-0006 b4.2)."""

from __future__ import annotations

import pytest

from lutgen.orchestration.preset import load_preset, save_preset


def test_round_trip(tmp_path):
    p = tmp_path / "look.json"
    save_preset(p, ["a.png", "b.png"], 0.8, title="Teal")
    data = load_preset(p)
    assert data["refs"] == ["a.png", "b.png"]
    assert data["strength"] == 0.8
    assert data["title"] == "Teal"


def test_defaults_filled(tmp_path):
    p = tmp_path / "min.json"
    p.write_text('{"refs": ["x.png"]}', encoding="utf-8")
    data = load_preset(p)
    assert data["strength"] == 1.0 and data["title"] is None


def test_invalid_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text('{"strength": 1.0}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_preset(p)
