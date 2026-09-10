from __future__ import annotations

import json

import pytest

from mario_rl.utils import load_frontier_actions


def test_load_frontier_actions_from_object(tmp_path):
    path = tmp_path / "frontier.json"
    path.write_text(json.dumps({"actions": [0, 3, 4]}), encoding="utf-8")

    assert load_frontier_actions(path) == (0, 3, 4)


def test_load_frontier_actions_rejects_empty_list(tmp_path):
    path = tmp_path / "frontier.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="No actions found"):
        load_frontier_actions(path)
