import pytest
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from bookgap.configs.config_store import register_configs
from bookgap.subcloud import SubCloud
from bookgap.util import Filter

# Cloud overrides that pin the cached AI-subfield (1702) production cloud, so tests
# run offline against on-disk embeddings. MUST match scripts/capture_gap_parity_golden.py.
_CLOUD_OVERRIDES = [
    "cloud.filter_pair.pre.domain=null",
    "cloud.filter_pair.pre.subfield='1702'",
    "cloud.filter_pair.pre.work_types=[book,article]",
    "cloud.denoiser=LPP",
    "cloud.drop_k=10",
    "gap.radius_fraction=0.75",
]


@pytest.fixture(scope="session")
def _ai_cloud():
    register_configs()
    with initialize_config_module(version_base=None, config_module="bookgap.configs"):
        cfg = compose(config_name="standard", overrides=_CLOUD_OVERRIDES)
    return instantiate(cfg.cloud)


@pytest.fixture
def tiny_subcloud(_ai_cloud):
    return SubCloud(parent=_ai_cloud, post_filter=Filter(topic="T13629", work_types=("article",)))


@pytest.fixture
def gist_embedder_cfg(_ai_cloud):
    return _ai_cloud.embedder


@pytest.fixture
def gemma_embedder_cfg(gist_embedder_cfg):
    import dataclasses
    return dataclasses.replace(gist_embedder_cfg, model_name="google/embeddinggemma-300m")
