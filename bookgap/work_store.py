from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence, cast
from datetime import datetime, date
import os
import shutil

import numpy as np
from numpy.typing import NDArray
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

WORKS_STORE_VERSION = "1"
ROW_GROUP_SIZE = 8192

_METADATA_VERSION_KEY = b"bookgap:works_store_version"
_METADATA_FINGERPRINT_KEY = b"bookgap:works_store_fingerprint"
_METADATA_EMBED_DIM_KEY = b"bookgap:works_store_embedding_dim"


def _metadata_from_parquet(path: Path) -> dict[bytes, bytes]:
    """Read metadata from Parquet dataset (handles both single file and partitioned)."""
    if path.is_dir():
        # For partitioned datasets, read metadata from first parquet file
        parquet_files = list(path.rglob("*.parquet"))
        if not parquet_files:
            raise ValueError(f"No parquet files found in dataset directory: {path}")
        metadata = pq.read_metadata(parquet_files[0]).metadata
    else:
        metadata = pq.read_metadata(path).metadata
    return metadata or {}


def _fixed_size_list_from_embeddings(embeddings: NDArray[np.floating]) -> pa.FixedSizeListArray:
    emb = np.asarray(embeddings, dtype=np.float32)
    if emb.ndim != 2:
        raise ValueError(f"Embeddings must be 2D, got shape {emb.shape}")
    flat = pa.array(emb.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, list_size=emb.shape[1])


def _indices_filter(indices: Optional[Sequence[int]]) -> Optional[pc.Expression]:
    if indices is None:
        return None
    if len(indices) == 0:
        return pc.field("row_id") == -1
    return pc.field("row_id").isin(pa.array(indices, type=pa.int64()))


def _to_pylist(arr: pa.Array) -> List:
    return arr.to_pylist()


def _safe_timestamp(values: Sequence[object]) -> pa.Array:
    """Coerce inputs to date32; tolerant of datetime/date/ISO strings; preserves nulls."""
    def _coerce(v: object, idx: int) -> Optional[date]:
        if v is None:
            return None
        if isinstance(v, date):
            return v
        if isinstance(v, datetime):
            return v.date()
        try:
            parsed = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
            return parsed.date()
        except Exception as e:
            raise ValueError(f"Invalid timestamp at index {idx}: {v!r}") from e
    coerced = [_coerce(v, i) for i, v in enumerate(values)]
    return pa.array(coerced, type=pa.date32())


def _sanitize_references(values: Sequence[object]) -> List[List[str]]:
    cleaned: List[List[str]] = []
    for i, v in enumerate(values):
        if v is None:
            cleaned.append([])
        elif isinstance(v, list):
            cleaned.append([str(x) for x in v])
        else:
            raise TypeError(f"Invalid referenced_works at index {i}: expected list or None, got {type(v).__name__}")
    return cleaned


@dataclass(frozen=True)
class WorksStore:
    dataset_path: Path
    fingerprint: str
    version: str
    length: int
    embedding_dim: int
    dataset: ds.Dataset = field(init=False, repr=False, compare=False)
    schema: pa.Schema = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        dataset = ds.dataset(self.dataset_path, format="parquet")
        object.__setattr__(self, "dataset", dataset)
        object.__setattr__(self, "schema", dataset.schema)

    def __getstate__(self) -> dict[str, object]:
        return {
            "dataset_path": str(self.dataset_path),
            "fingerprint": self.fingerprint,
            "version": self.version,
            "length": self.length,
            "embedding_dim": self.embedding_dim,
        }

    def __setstate__(self, state: dict[str, object]) -> None:  # pragma: no cover - defensive
        path = Path(str(state["dataset_path"]))
        object.__setattr__(self, "dataset_path", path)
        object.__setattr__(self, "fingerprint", str(state["fingerprint"]))
        object.__setattr__(self, "version", str(state["version"]))
        object.__setattr__(self, "length", int(cast(Any, state["length"])))
        object.__setattr__(self, "embedding_dim", int(cast(Any, state["embedding_dim"])))
        dataset = ds.dataset(path, format="parquet")
        object.__setattr__(self, "dataset", dataset)
        object.__setattr__(self, "schema", dataset.schema)

    @classmethod
    def load(cls, dataset_path: Path, *, expected_fingerprint: str) -> "WorksStore":
        metadata = _metadata_from_parquet(dataset_path)
        version = metadata.get(_METADATA_VERSION_KEY, b"").decode()
        fingerprint = metadata.get(_METADATA_FINGERPRINT_KEY, b"").decode()
        emb_dim_raw = metadata.get(_METADATA_EMBED_DIM_KEY, b"0")
        embedding_dim = int(emb_dim_raw.decode()) if isinstance(emb_dim_raw, (bytes, bytearray)) else int(emb_dim_raw)

        if version != WORKS_STORE_VERSION:
            raise ValueError(f"WorksStore version mismatch: found '{version}', expected '{WORKS_STORE_VERSION}'")
        if fingerprint != expected_fingerprint:
            raise ValueError(f"WorksStore fingerprint mismatch: found '{fingerprint}', expected '{expected_fingerprint}'")

        # For partitioned datasets, count rows across all files
        dataset = ds.dataset(dataset_path, format="parquet")
        num_rows = dataset.count_rows()
        return cls(
            dataset_path=dataset_path,
            fingerprint=fingerprint,
            version=version,
            length=num_rows,
            embedding_dim=embedding_dim,
        )

    @classmethod
    def write(
        cls,
        dataset_path: Path,
        *,
        fingerprint: str,
        works: list[dict[str, object]],
        embeddings: NDArray[np.floating],
    ) -> "WorksStore":
        dataset_path.parent.mkdir(parents=True, exist_ok=True)

        emb_array = _fixed_size_list_from_embeddings(embeddings)
        types = [w.get("type") or "UNKNOWN" for w in works]
        primary_topic_id: list[Optional[str]] = []
        primary_subfield_id: list[Optional[str]] = []
        for w in works:
            pt = w.get("primary_topic")
            if isinstance(pt, dict):
                primary_topic_id.append(cast(Optional[str], pt.get("id")))
                subfield = pt.get("subfield") if isinstance(pt.get("subfield"), dict) else None
                primary_subfield_id.append(cast(Optional[str], subfield.get("id")) if subfield else None)
            else:
                primary_topic_id.append(None)
                primary_subfield_id.append(None)

        publication_dates = _safe_timestamp([w.get("publication_date") for w in works])
        created_dates = _safe_timestamp([w.get("created_date") for w in works])
        updated_dates = _safe_timestamp([w.get("updated_date") for w in works])
        referenced_lists = _sanitize_references([w.get("referenced_works") for w in works])
        table = pa.table(
            {
                "row_id": pa.array(range(len(works)), type=pa.int64()),
                "id": pa.array([w.get("id") for w in works], type=pa.string()),
                "display_name": pa.array([w.get("display_name", "") for w in works], type=pa.string()),
                "abstract": pa.array([w.get("abstract", "") for w in works], type=pa.string()),
                "type": pa.array(types, type=pa.string()),
                "publication_date": publication_dates,
                "created_date": created_dates,
                "updated_date": updated_dates,
                "language": pa.array([w.get("language") for w in works], type=pa.string()),
                "topic_label": pa.array([w.get("topic") for w in works], type=pa.string()),
                "primary_topic_id": pa.array(primary_topic_id, type=pa.string()),
                "primary_subfield_id": pa.array(primary_subfield_id, type=pa.string()),
                "referenced_works": pa.array(referenced_lists, type=pa.list_(pa.string())),
                "embeddings": emb_array,
            }
        )

        metadata = {
            **(table.schema.metadata or {}),
            _METADATA_VERSION_KEY: WORKS_STORE_VERSION.encode(),
            _METADATA_FINGERPRINT_KEY: fingerprint.encode(),
            _METADATA_EMBED_DIM_KEY: str(embeddings.shape[1]).encode(),
        }
        table = table.replace_schema_metadata(metadata)

        # Write with partitioning by work type for efficient filtering
        # For partitioned datasets, dataset_path is the root directory
        tmp_path = dataset_path.parent / (dataset_path.name + ".tmp")
        try:
            ds.write_dataset(
                table,
                tmp_path,
                format="parquet",
                file_options=ds.ParquetFileFormat().make_write_options(
                    compression="zstd",
                    use_dictionary=True,
                ),
                existing_data_behavior="overwrite_or_ignore",
                max_rows_per_file=ROW_GROUP_SIZE,
                max_rows_per_group=ROW_GROUP_SIZE,
            )
            # Atomic replace: move temp dir to final location
            if dataset_path.exists():
                shutil.rmtree(dataset_path)
            os.rename(tmp_path, dataset_path)
        except Exception:
            if tmp_path.exists():
                shutil.rmtree(tmp_path)
            raise
        return cls.load(dataset_path, expected_fingerprint=fingerprint)

    def _table(self, *, columns: Optional[Sequence[str]] = None, indices: Optional[Sequence[int]] = None) -> pa.Table:
        filt = _indices_filter(indices)
        return self.dataset.to_table(columns=columns, filter=filt)

    def to_list(self, column: str, *, indices: Optional[Sequence[int]] = None) -> List:
        if indices is not None and len(indices) == 0:
            return []
        table = self._table(columns=[column], indices=indices)
        return _to_pylist(table.column(0))

    def embeddings(self, *, indices: Optional[Sequence[int]] = None) -> NDArray[np.float32]:
        if indices is not None and len(indices) == 0:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        table = self._table(columns=["embeddings"], indices=indices)
        arr = table.column(0).combine_chunks().values
        emb = np.asarray(arr.to_numpy(zero_copy_only=False), dtype=np.float32)
        if emb.size == 0:
            return emb.reshape(0, self.embedding_dim)
        return emb.reshape(-1, self.embedding_dim)

    def to_dicts(self, *, indices: Optional[Sequence[int]] = None) -> List[dict[str, object]]:
        if indices is not None and len(indices) == 0:
            return []
        columns = [
            "id",
            "display_name",
            "abstract",
            "type",
            "publication_date",
            "created_date",
            "updated_date",
            "language",
            "topic_label",
            "primary_topic_id",
            "primary_subfield_id",
            "referenced_works",
        ]
        table = self._table(columns=columns, indices=indices)
        ids = _to_pylist(table.column("id"))
        display_names = _to_pylist(table.column("display_name"))
        abstracts = _to_pylist(table.column("abstract"))
        types = _to_pylist(table.column("type"))
        publication_dates = _to_pylist(table.column("publication_date"))
        created_dates = _to_pylist(table.column("created_date"))
        updated_dates = _to_pylist(table.column("updated_date"))
        languages = _to_pylist(table.column("language"))
        topic_labels = _to_pylist(table.column("topic_label"))
        primary_topic_ids = _to_pylist(table.column("primary_topic_id"))
        primary_subfield_ids = _to_pylist(table.column("primary_subfield_id"))
        references = _to_pylist(table.column("referenced_works"))

        works: List[dict[str, object]] = []
        for i in range(len(ids)):
            pub = publication_dates[i]
            if isinstance(pub, datetime):
                pub = pub.isoformat()
            created = created_dates[i]
            if isinstance(created, datetime):
                created = created.isoformat()
            updated = updated_dates[i]
            if isinstance(updated, datetime):
                updated = updated.isoformat()
            works.append(
                {
                    "id": ids[i],
                    "display_name": display_names[i] or "",
                    "abstract": abstracts[i] or "",
                    "type": types[i] or "UNKNOWN",
                    "publication_date": pub,
                    "created_date": created,
                    "updated_date": updated,
                    "language": languages[i],
                    "topic": topic_labels[i],
                    "primary_topic": {
                        "id": primary_topic_ids[i],
                        "subfield": {"id": primary_subfield_ids[i]},
                    },
                    "referenced_works": references[i] or [],
                }
            )
        return works
