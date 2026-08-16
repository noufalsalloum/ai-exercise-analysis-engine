"""Fail-open Boundary V1 integration for REHAB24 Ex5 Lunge."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import torch

from models.lunge_rep_boundary import LungeRepBoundaryModel
from preprocessing.h36m_coordinate_normalizer import H36MCoordinateNormalizer
from preprocessing.landmark_selector import LandmarkSelector
from training.squat_rep_boundary_v2 import v2_segments


class LungeBoundaryExperimentalOrchestrator:
    SCOPE="REHAB24 Ex5 Lunge — Boundary V1 only"
    PROJECT_ROOT=Path(__file__).resolve().parents[1]

    def __init__(self, checkpoint: Path, device: torch.device|None=None) -> None:
        self.device=device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        payload=torch.load(checkpoint,map_location=self.device,weights_only=True)
        self.model=LungeRepBoundaryModel().to(self.device); self.model.load_state_dict(payload["model_state_dict"],strict=True); self.model.eval()
        self.postprocessing=dict(payload["postprocessing"]); self.selector=LandmarkSelector({"landmarks":{"selected_landmarks":[]}}); self.normalizer=H36MCoordinateNormalizer()
        self._frames=[]; self._results=[]; self._lock=Lock(); self._executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix="lunge-boundary"); self._future:Future|None=None; self._last_requested=0; self._closed=False

    @classmethod
    def from_default_config(cls):
        return cls(cls.PROJECT_ROOT/"checkpoints/lunge_ai_v1/boundary/best.pt")

    def record_frame(self, landmarks, frame_index:int, timestamp:float) -> None:
        if landmarks is None:
            value=self._frames[-1].copy() if self._frames else np.zeros((33,4),np.float32); value[:,3]=0
        else:
            value=np.asarray(landmarks,np.float32)
            if value.shape!=(33,4) or not np.isfinite(value).all(): raise ValueError("Lunge landmarks must be finite (33,4)")
        with self._lock: self._frames.append(value.copy())

    @torch.no_grad()
    def _detect(self, frames:np.ndarray, stable_only:bool):
        if len(frames)<24:return []
        h36m=self.selector.to_h36m_17(frames); values,_=self.normalizer.normalize(h36m); tensor=torch.from_numpy(values).unsqueeze(0).to(self.device); output=self.model(tensor)
        active=torch.sigmoid(output["active_logits"]).squeeze(0).cpu().numpy(); boundary=torch.sigmoid(output["boundary_logits"]).squeeze(0).cpu().numpy(); segments=v2_segments(active,boundary,self.postprocessing)
        if stable_only: segments=[value for value in segments if value[1]<=len(frames)-10]
        return [{"rep_index":i,"start_frame":start+1,"end_frame":end+1,"assessment":None,"confidence":None,"experimental":True} for i,(start,end) in enumerate(segments,1)]

    def _collect(self):
        if self._future is not None and self._future.done(): self._results=self._future.result(); self._future=None

    def request_live_analysis(self):
        self._collect()
        with self._lock:
            n=len(self._frames)
            if self._future is not None or n<32 or n-self._last_requested<15:return
            snapshot=np.asarray(self._frames,np.float32); self._last_requested=n
        self._future=self._executor.submit(self._detect,snapshot,True)

    def live_status(self):
        self._collect(); last=self._results[-1] if self._results else None
        return {"exercise":"lunge","available":True,"experimental":True,"scope":self.SCOPE,"assessment_kind":"learned_detection_only","ai_detected_reps":len(self._results),"ai_detected_count":len(self._results),"last_ai_rep":last,"per_rep_results":list(self._results),"analyzing":self._future is not None,"performance_score":None,"assessment_coverage":None,"model_status":{"detection":"Boundary V1 — Development","correctness":"Not Enabled — locked-Test generalization failed","errors":"Not Available","score":"Not Available"}}

    def finalize(self):
        if self._future is not None:self._future.result();self._collect()
        with self._lock: frames=np.asarray(self._frames,np.float32)
        reps=self._detect(frames,False); result=self.live_status(); result.update({"ai_detected_reps":len(reps),"ai_detected_count":len(reps),"per_rep_results":reps,"last_ai_rep":reps[-1] if reps else None,"analyzing":False}); return result

    def close(self):
        if self._closed:return
        self._closed=True
        if self._future is not None:self._future.cancel()
        self._executor.shutdown(wait=True,cancel_futures=True)
