import argparse
from dataclasses import dataclass
import logging
from typing import Any, Optional

from nextbook.cloud import WorkDenoisers
from nextbook.index import IndexType
from nextbook.deduplicators import Deduplicator, FractionalDeduplicator, SqDistDeduplicator


@dataclass(frozen=True)
class Args:
    drop_k: int
    # log_level: str
    max_works: Optional[int]
    deduplicator: Optional[Deduplicator]
    radius_fraction: float
    denoiser: Optional[str]
    index_type: IndexType
    hnsw_m: int
    hnsw_efc: int
    hnsw_efs: int

def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drop_k", type=int, default=0, help="Number of top components to drop from embeddings")
    # parser.add_argument("--log_level", type=str, default=logging._levelToName[logging.DEBUG], choices=sorted(logging._nameToLevel.keys(), key=lambda x: logging._nameToLevel[x]))
    parser.add_argument("--max_works", type=int, default=None, help="Maximum number of works to fetch")
    parser.add_argument("--duplicate_fraction", type=float, default=0.02, help="Value in [0, 1] for fraction of duplicates (removed by semantic distance percentile). This is more computationally expensive than min_sqdist.")
    parser.add_argument("--min_sqdist", type=float, default=None, help="Minimum squared distance for deduplication using range search")
    parser.add_argument("--radius_fraction", type=float, default=0.75, help="Radius fraction for ball coverage")
    parser.add_argument("--denoiser", type=str, choices=[d.name for d in WorkDenoisers], default=None, help=f"Denoiser method.")
    parser.add_argument("--index_type", type=IndexType._member_map_.__getitem__, choices=list(IndexType), default=IndexType.HNSW, help=f"Faiss index type. Choices: {', '.join(map(str, IndexType))}")
    # HNSW specific arguments
    parser.add_argument("--hnsw_m", type=int, default=16, help="HNSW M parameter")
    parser.add_argument("--hnsw_efc", type=int, default=512, help="HNSW efConstruction parameter")
    parser.add_argument("--hnsw_efs", type=int, default=1024, help="HNSW efSearch parameter")
    return parser

def _process_deduplication_args(parsed_args: dict[str, Any]):
    duplicate_fraction = parsed_args["duplicate_fraction"]
    min_sqdist = parsed_args["min_sqdist"]
    if duplicate_fraction is not None and min_sqdist is not None:
       raise ValueError("Cannot specify both duplicate_fraction and min_sqdist")
    if min_sqdist is not None:
        deduplicator = SqDistDeduplicator(min_sqdist=min_sqdist)
    elif duplicate_fraction is not None:
        deduplicator = FractionalDeduplicator(duplicate_fraction=duplicate_fraction)
    else:
        deduplicator = None
    del parsed_args["duplicate_fraction"]
    del parsed_args["min_sqdist"]
    parsed_args["deduplicator"] = deduplicator

def get_explicit_args(args: Optional[list[str]] = None):
    parser = _get_parser()
    # Set all arguments to have unique default
    sentinel = object()
    parser.set_defaults(**{arg: sentinel for arg in vars(parser.parse_args(args))})
    # Re-parse the arguments to get explicit values
    explicit_args = vars(parser.parse_args(args))
    return [(k, v) for k, v in explicit_args.items() if v is not sentinel]

def parse_args(args: Optional[list[str]] = None) -> Args:
    parser = _get_parser()
    parsed_args = parser.parse_args(args)
    parsed_dict = vars(parsed_args)
    _process_deduplication_args(parsed_dict)
    return Args(**parsed_dict)
