import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Optional

from nextbook.cloud import WorkDenoisers
from nextbook.index import IndexType
from nextbook.util import create_unicode_safe_filename, setup_logging


@dataclass(frozen=True)
class Args:
    drop_k: int
    log_level: str
    max_works: Optional[int]
    duplicate_expectation: float
    radius_fraction: float
    denoiser: Optional[str]
    index_type: IndexType
    hnsw_m: int
    hnsw_efc: int
    hnsw_efs: int
def parse_args(log_prefix: Optional[str] = None) -> Args:
    parser = argparse.ArgumentParser(description="Fetch, deduplicate, embed, and analyze OpenAlex works.")
    parser.add_argument("--drop_k", type=int, default=0, help="Number of top components to drop from embeddings")
    # Arguments for Config fields
    parser.add_argument("--log_level", type=str, default=logging._levelToName[logging.DEBUG], choices=sorted(logging._nameToLevel.keys(), key=lambda x: logging._nameToLevel[x]))
    parser.add_argument("--max_works", type=int, default=None, help="Maximum number of works to fetch")
    parser.add_argument("--duplicate_expectation", type=float, default=.05, help="Value in [0, 1] for expected fraction of duplicates (removed by semantic distance percentile)")
    parser.add_argument("--radius_fraction", type=float, default=0.75, help="Radius fraction for ball coverage")
    parser.add_argument("--denoiser", type=str, choices=[d.name for d in WorkDenoisers], default=None, help=f"Denoiser method.")
    
    # Arguments for IndexInfo fields (and its subtypes)
    parser.add_argument("--index_type", type=IndexType._member_map_.__getitem__, choices=list(IndexType), default=IndexType.HNSW, help=f"Faiss index type. Choices: {', '.join(map(str, IndexType))}")
    # HNSW specific arguments
    parser.add_argument("--hnsw_m", type=int, default=16, help="HNSW M parameter")
    parser.add_argument("--hnsw_efc", type=int, default=8192, help="HNSW efConstruction parameter")
    parser.add_argument("--hnsw_efs", type=int, default=1024, help="HNSW efSearch parameter")
    # parser.add_argument("--index_metric", type=IndexMetric._member_map_.__getitem__, choices=list(IndexMetric), default=IndexMetric.L2.name, help=f"Faiss index metric (used in HNSW). Choices: {', '.join(IndexMetric._member_names_)}")
    # parser.add_argument("--normalize_embeddings", action="store_true", default=False, help="Whether to normalize embeddings")

    parsed_args = parser.parse_args()


    def get_explicit_args(parser: argparse.ArgumentParser, args: argparse.Namespace):
        # Set all arguments to have unique default
        sentinel = object()
        parser.set_defaults(**{arg: sentinel for arg in vars(args)})
        # Re-parse the arguments to get explicit values
        explicit_args = vars(parser.parse_args())
        ret = [(k, v) for k, v in explicit_args.items() if v is not sentinel]
        ret = [(k, v) for k, v in ret if k not in {"log_level"}]
        return [(k, v) for k, v in explicit_args.items() if v is not sentinel]
    cmdline_args = get_explicit_args(parser, parsed_args)

    args = Args(**vars(parsed_args))
    log_name_parts = []
    if log_prefix:
        log_name_parts.append(log_prefix)
    log_name_parts.append(create_unicode_safe_filename("_".join(f"{k}:{v}" for k, v in sorted(cmdline_args))))
    setup_logging(log_name="_".join(log_name_parts),
                  log_level=args.log_level)
    return args