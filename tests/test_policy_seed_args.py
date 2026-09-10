from __future__ import annotations

import sys

from mario_rl.evaluate import parse_args as parse_evaluate_args
from mario_rl.play import parse_args as parse_play_args


def test_policy_seed_defaults_to_none(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["mario-evaluate"])
    assert parse_evaluate_args().policy_seed is None

    monkeypatch.setattr(sys, "argv", ["mario-play"])
    assert parse_play_args().policy_seed is None


def test_policy_seed_can_be_overridden(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["mario-evaluate", "--policy-seed", "20260910"],
    )
    assert parse_evaluate_args().policy_seed == 20260910

    monkeypatch.setattr(
        sys,
        "argv",
        ["mario-play", "--policy-seed", "20260910"],
    )
    assert parse_play_args().policy_seed == 20260910
