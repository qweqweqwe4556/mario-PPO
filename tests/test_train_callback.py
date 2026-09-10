from __future__ import annotations

import sys
from types import SimpleNamespace

from torch import nn

from mario_rl.train import evaluation_score, parse_args, set_features_frozen


def test_flag_score_ignores_progress_when_success_count_is_equal():
    near_finish = evaluation_score("flag", 0, 10, 2994.0, 2994)
    early_failure = evaluation_score("flag", 0, 0, 500.0, 500)

    assert near_finish == early_failure == 0.0


def test_flag_score_always_prefers_more_completions():
    one_completion = evaluation_score("flag", 1, 0, 500.0, 500)
    no_completion = evaluation_score("flag", 0, 10, 2994.0, 2994)

    assert one_completion > no_completion


def test_legacy_progress_metrics_remain_available():
    assert evaluation_score("mean", 0, 0, 1200.0, 1800) == 1200.0
    assert evaluation_score("max", 0, 0, 1200.0, 1800) == 1800.0
    assert evaluation_score("success", 0, 2, 1200.0, 1800) == 21200.0


def test_n_epochs_defaults_to_existing_ppo_value(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["mario-train"])

    assert parse_args().n_epochs == 10


def test_n_epochs_can_be_overridden(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["mario-train", "--n-epochs", "3"])

    assert parse_args().n_epochs == 3


def test_freeze_features_defaults_to_false(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["mario-train"])

    assert parse_args().freeze_features is False


def test_freeze_features_can_be_enabled(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["mario-train", "--freeze-features"])

    assert parse_args().freeze_features is True


def test_set_features_frozen_disables_gradients():
    extractor = nn.Sequential(nn.Linear(4, 2), nn.ReLU())
    model = SimpleNamespace(
        policy=SimpleNamespace(features_extractor=extractor),
    )

    set_features_frozen(model, True)

    assert all(not parameter.requires_grad for parameter in extractor.parameters())


def test_set_features_frozen_can_restore_gradients():
    extractor = nn.Sequential(nn.Linear(4, 2), nn.ReLU())
    model = SimpleNamespace(
        policy=SimpleNamespace(features_extractor=extractor),
    )
    set_features_frozen(model, True)

    set_features_frozen(model, False)

    assert all(parameter.requires_grad for parameter in extractor.parameters())
