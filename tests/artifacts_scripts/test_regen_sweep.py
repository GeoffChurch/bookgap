"""regen_sweep's cell plan and run identity: what makes a run resumable in place."""
from __future__ import annotations

import json

from omegaconf import OmegaConf

from artifacts_scripts.paper.regen_sweep import (
    RADIUS_ROOT, TOPIC_SET_ROOTS, cells, config_for, is_scored, run_dir, run_identity,
)
from bookgap.llm.generators.run_state import RunStep


def _yaml(cell) -> str:
    return OmegaConf.to_yaml(OmegaConf.structured(config_for(cell)))


def test_random20_keeps_the_headline_on_all_topics_and_the_rest_on_twenty():
    plan = cells("gist", topics="random20")
    table = [c for c in plan if c.root == TOPIC_SET_ROOTS["random20"]]
    assert [c.slug for c in table if c.all_topics] == ["gist_a10_b5_dropk10_lpp_t77_r0.75"]
    assert all(c.root == RADIUS_ROOT and c.all_topics for c in plan if c.radius_fraction != 0.75)
    assert len(plan) == 15 and len(cells("gemma", topics="random20")) == 12


def test_all_puts_every_table_cell_on_all_topics_in_its_own_directory():
    plan = cells("gemma", topics="all")
    assert all(c.all_topics and c.root == TOPIC_SET_ROOTS["all"] for c in plan)
    headline = next(c for c in cells("gist", topics="all") if c.slug == "gist_a10_b5_dropk10_lpp_t77_r0.75")
    twenty = next(c for c in cells("gist", topics="random20") if c.slug == headline.slug)
    # The same configuration in both sets, so the same run identity.
    assert run_identity(_yaml(headline), "d") == run_identity(_yaml(twenty), "d")


def test_identity_follows_results_not_the_context_window():
    cell = cells("gist", topics="random20")[0]
    base = _yaml(cell)
    wider = OmegaConf.to_yaml(OmegaConf.merge(OmegaConf.create(base), {"llm": {"ollama_num_ctx": 65536}}))
    hotter = OmegaConf.to_yaml(OmegaConf.merge(OmegaConf.create(base), {"llm": {"temperature": 0.9}}))
    assert run_identity(base, "d") == run_identity(wider, "d")
    assert run_identity(base, "d") != run_identity(hotter, "d")
    assert run_identity(base, "d") != run_identity(base, "other-weights")


def test_a_cell_is_done_only_once_scored(tmp_path):
    cell = cells("gist", topics="random20")[0]
    out = run_dir(cell, _yaml(cell), "d").relative_to(cell.root)
    out = tmp_path / out
    assert not is_scored(out)
    out.mkdir()
    (out / "run_state.json").write_text(json.dumps({"step": RunStep.COMPLETED.name}))
    assert not is_scored(out)
    (out / "run_state.json").write_text(json.dumps({"step": RunStep.SCORED.name}))
    assert is_scored(out)
