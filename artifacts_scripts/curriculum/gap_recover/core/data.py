"""Generic gap-recover data contract loader (substrate-agnostic)."""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import scipy.sparse as sp


class Data:
    def __init__(self, data_dir: str | os.PathLike) -> None:
        d = np.load(Path(data_dir) / "data.npz", allow_pickle=True)
        self.emb = d["emb"].astype(np.float32)
        self.is_F = d["is_F"]; self.is_B = d["is_B"]
        self.kinds = d["kinds"]; self.mods = d["mods"]
        self.names = np.array([str(n) for n in d["names"]])
        self.structural_users = d["structural_users"].astype(np.int64)
        self.operational_users = d["operational_users"].astype(np.int64)
        self.concept_pref = d["concept_pref"].astype(bool)
        self.N = self.names.shape[0]
        self.sub = sp.load_npz(Path(data_dir) / "subgraph.npz").tocsr()
        self.subT = self.sub.T.tocsr()
        self.F_idx = np.where(self.is_F)[0]; self.B_idx = np.where(self.is_B)[0]
        self.subT_B = self.subT[self.B_idx, :].tocsr()
        self.Zf = self.emb[self.F_idx]; self.Zb = self.emb[self.B_idx]
        self.f_pos = -np.ones(self.N, dtype=np.int64); self.f_pos[self.F_idx] = np.arange(self.F_idx.size)
        self.b_pos = -np.ones(self.N, dtype=np.int64); self.b_pos[self.B_idx] = np.arange(self.B_idx.size)
        self.ns = np.array([n.rsplit(".", 1)[0] if "." in n else "" for n in self.names], dtype=object)

    def load_births(self, data_dir):
        b = np.load(Path(data_dir) / "births.npz", allow_pickle=True)
        self.birth = b["birth"].astype(np.int64)
        self.tags = [str(t) for t in b["tags"]]
        self.SENTINEL = len(self.tags)
        return self
