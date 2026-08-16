from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HAS_RUNTIME = all(importlib.util.find_spec(name) is not None for name in ("torch", "mediapipe", "cv2"))


@unittest.skipUnless(
    HAS_RUNTIME and os.environ.get("RUN_SLOW_TESTS") == "1",
    "Slow video test requires RUN_SLOW_TESTS=1 plus torch, MediaPipe, and OpenCV.",
)
class FullPipelineSlowTests(unittest.TestCase):
    """Optional real-video smoke test; excluded from normal unit runs."""

    def test_plank_video_limited_frames_and_cache(self) -> None:
        import numpy as np
        import torch

        from experts.registry import ExpertRegistry
        from backbone.motionbert import MotionBERT
        from preprocessing.preprocessor import PreprocessorController

        video = ROOT / "datasets" / "raw" / "plank" / "plank_0.mp4"
        pose_model = Path(r"C:\MediaPipe\pose_landmarker_full.task")
        checkpoint = ROOT / "models" / "latest_epoch.bin"
        cache = ROOT / "datasets" / "pose_cache" / "plank_0_90frames.npz"
        self.assertTrue(video.is_file())
        self.assertTrue(pose_model.is_file())
        self.assertTrue(checkpoint.is_file())

        controller = PreprocessorController(
            ROOT / "configs" / "plank.json",
            pose_model_path=str(pose_model),
        )
        windows = controller.process_video(video, max_frames=90, cache_path=cache)
        inputs = torch.from_numpy(
            np.stack([window["motionbert_input"] for window in windows]).astype(np.float32)
        )
        backbone = MotionBERT(checkpoint_path=str(checkpoint)).eval()
        registry = ExpertRegistry(dropout=0.0).eval()
        with torch.no_grad():
            features = backbone(inputs)
            output = registry(features, "plank")
        self.assertEqual(features.shape[-2:], (17, 512))
        self.assertEqual(output["global_embedding"].shape, (len(windows), 1024))
        self.assertEqual(output["temporal_embedding"].shape[:2], inputs.shape[:2])
        self.assertTrue(torch.isfinite(output["global_embedding"]).all())


if __name__ == "__main__":
    unittest.main()
