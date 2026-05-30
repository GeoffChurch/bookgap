# scripts/capture_gap_parity_golden.py
"""One-shot: snapshot current gap output for a 2-topic subset (pre-refactor golden)."""
import json
from pathlib import Path

from hydra import compose, initialize_config_module

from bookgap.configs.config_store import register_configs
from bookgap.gaps import compute_gaps_for_config

# Two AI-subfield topics with small, fully-cached clouds. Adjust if these are slow.
GOLDEN_TOPICS = ("T13629", "T12761")


def main() -> None:
    register_configs()
    with initialize_config_module(version_base=None, config_module="bookgap.configs"):
        cfg = compose(
            config_name="standard",
            overrides=[
                # Pin the cloud to the cached AI-subfield (1702) production cloud
                # so the capture runs offline against on-disk embeddings and
                # reproduces the production gap output byte-for-byte.
                "cloud.filter_pair.pre.domain=null",
                "cloud.filter_pair.pre.subfield='1702'",
                "cloud.filter_pair.pre.work_types=[book,article]",
                "cloud.denoiser=LPP",
                "cloud.drop_k=10",
                "gap.radius_fraction=0.75",
                f"gap.topic_subset=[{','.join(GOLDEN_TOPICS)}]",
            ],
        )
    gaps = compute_gaps_for_config(cfg)
    golden = {}
    for g in gaps:
        key = f"{g.topic_id}:{g.gap_idx}"
        golden[key] = {
            "in_gap_article_ids": sorted(
                g.articles.work_ids()[int(i)] for i in g.gap.get_dists_idxs().idxs
            ),
            "gap_size": int(g.gap_size),
            "gap_size_percentile": round(float(g.gap_size_percentile), 9),
        }
    out = Path("tests/fixtures/gap_parity_golden.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(golden, indent=2, sort_keys=True))
    print(f"Wrote {out} with {len(golden)} (topic,gap) entries")


if __name__ == "__main__":
    main()
