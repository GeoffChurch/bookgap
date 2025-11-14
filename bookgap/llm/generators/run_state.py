"""Run state tracking utilities for resumable batch workflows."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional


class RunStep(str, Enum):
    """Enumerated stages of the batch workflow lifecycle."""

    INITIALIZED = "initialized"
    READY_TO_SUBMIT = "ready_to_submit"
    SUBMITTED = "submitted"
    COMPLETED = "completed"


@dataclass
class RunState:
    """Serializable record of workflow progress for resumption."""

    step: RunStep = RunStep.INITIALIZED
    batch_job_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    last_updated: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step.value,
            "batch_job_id": self.batch_job_id,
            "details": self.details,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunState":
        step_value = data.get("step", RunStep.INITIALIZED.value)
        step = RunStep(step_value)
        state = cls(
            step=step,
            batch_job_id=data.get("batch_job_id"),
            details=data.get("details", {}),
            last_updated=data.get("last_updated", datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )
        return state


class RunStateManager:
    """Utility wrapper for reading and persisting run-state metadata."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.state_path = output_dir / "run_state.json"
        if self.state_path.exists():
            self.state = self._load()
        else:
            self.state = RunState()
            self._persist()

    def _load(self) -> RunState:
        with open(self.state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return RunState.from_dict(data)

    def _persist(self) -> None:
        self.state.last_updated = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(self.state.to_dict(), f, indent=2, sort_keys=True)

    def refresh(self) -> None:
        """Reload state from disk (useful after external modifications)."""
        if self.state_path.exists():
            self.state = self._load()

    def mark_ready_to_submit(
        self,
        *,
        gap_count: int,
        context_file_count: int,
        request_count: int,
        embedder_model: str,
        denoiser: str,
        drop_k: str,
    ) -> None:
        self.state.details.update(
            {
                "gap_count": gap_count,
                "context_file_count": context_file_count,
                "request_count": request_count,
                "embedder_model": embedder_model,
                "denoiser": denoiser,
                "drop_k": drop_k,
            }
        )
        if self.state.step in {RunStep.INITIALIZED, RunStep.READY_TO_SUBMIT}:
            self.state.step = RunStep.READY_TO_SUBMIT
            self.state.batch_job_id = None
        self._persist()

    def mark_submitted(self, batch_job_id: str, *, request_count: int) -> None:
        self.state.step = RunStep.SUBMITTED
        self.state.batch_job_id = batch_job_id
        self.state.details["request_count"] = request_count
        self._persist()

    def mark_completed(self, batch_job_id: str, *, proposal_count: int) -> None:
        self.state.step = RunStep.COMPLETED
        self.state.batch_job_id = batch_job_id
        self.state.details["proposal_count"] = proposal_count
        self._persist()

    def mark_ready_for_resubmission(self, *, status: str) -> None:
        details = dict(self.state.details)
        details["last_failure_status"] = status
        self.state.step = RunStep.READY_TO_SUBMIT
        self.state.batch_job_id = None
        self.state.details = details
        self._persist()

    def require_batch_job_id(self) -> str:
        batch_job_id = self.state.batch_job_id
        if not batch_job_id:
            raise ValueError("run_state.json is missing a recorded batch_job_id")
        return batch_job_id

    def reset(self) -> None:
        """Reset the run state to initial for fresh runs."""
        self.state = RunState()
        self._persist()