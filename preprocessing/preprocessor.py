from __future__ import annotations

import json
from pathlib import Path

import numpy as np

if __package__:
    from .landmark_selector import LandmarkSelector
    from .interpolation import Interpolator
    from .smoothing import Smoother
    from .normalizer import Normalizer
    from .feature_extractor import FeatureExtractor
    from .sequence_builder import SequenceBuilder
    from .pose_cache import PoseCache
else:
    from landmark_selector import LandmarkSelector
    from interpolation import Interpolator
    from smoothing import Smoother
    from normalizer import Normalizer
    from feature_extractor import FeatureExtractor
    from sequence_builder import SequenceBuilder
    from pose_cache import PoseCache


class PreprocessorController:
    """
    Phase 2:
    - Input: processed .npy from PoseExtractor (frames, 33, 4)
    - Output: sequence windows containing:
        * motionbert_input (17, 3)
        * landmarks (selected landmarks)
        * features (angles, velocity, rom, trajectory)
    """

    def __init__(
        self,
        spec_path: str,
        window_size: int = 30,
        step_size: int = 5,
        pose_model_path: str | None = None,
    ):
        self.spec_path = Path(spec_path)

        with self.spec_path.open("r", encoding="utf-8") as f:
            self.spec = json.load(f)

        if pose_model_path:
            if __package__:
                from .pose_extractor import PoseExtractor
            else:
                from pose_extractor import PoseExtractor
            self.pose_extractor = PoseExtractor(pose_model_path)
        else:
            self.pose_extractor = None
        self.last_video_fps: float | None = None
        self.selector = LandmarkSelector(self.spec)
        self.interpolator = Interpolator()
        self.smoother = Smoother(window_size=5)
        self.normalizer = Normalizer(self.spec)
        self.feature_extractor = FeatureExtractor(self.spec)
        self.builder = SequenceBuilder(window_size=window_size, step_size=step_size)

    def process_video(
        self,
        video_path: str | Path,
        max_frames: int | None = None,
        cache_path: str | Path | None = None,
        use_cache: bool = True,
    ) -> list:
        if self.pose_extractor is None:
            raise ValueError("pose_model_path was not provided in __init__.")

        video = Path(video_path)
        cache = Path(cache_path) if cache_path is not None else None
        if use_cache and cache is not None and cache.is_file():
            raw_landmarks_33, cache_metadata = PoseCache.load(cache)
            cached_fps = cache_metadata.get("fps")
            self.last_video_fps = float(cached_fps) if cached_fps is not None else None
            if max_frames is not None:
                raw_landmarks_33 = raw_landmarks_33[:max_frames]
        else:
            raw_landmarks_33 = self.pose_extractor.extract_from_video(
                video,
                max_frames=max_frames,
            )
            self.last_video_fps = self.pose_extractor.last_fps
            if cache is not None:
                PoseCache.save(
                    cache,
                    raw_landmarks_33,
                    metadata={
                        "video_path": str(video),
                        "max_frames": max_frames,
                        "fps": self.last_video_fps,
                    },
                )
        return self.process_landmarks(raw_landmarks_33)


    def process_landmarks(self, raw_landmarks_33: np.ndarray) -> list:

        if raw_landmarks_33.ndim != 3 or raw_landmarks_33.shape[1:] != (33, 4):
            raise ValueError(
                "Expected raw_landmarks_33 shape: (frames, 33, 4)"
            )
        if len(raw_landmarks_33) == 0 or not np.isfinite(raw_landmarks_33).all():
            raise ValueError("MediaPipe landmarks must be non-empty and finite.")

        # ======================================================
        # Step 1: preprocess ALL 33 landmarks
        # ======================================================

        interpolated_33 = self.interpolator.interpolate_sequence(
            raw_landmarks_33
        )

        smoothed_33 = self.smoother.smooth_sequence(
            interpolated_33
        )

        normalized_33 = self.normalizer.normalize_coordinates(
            smoothed_33
        )

        # ======================================================
        # Step 2: MotionBERT branch
        # (needs all 33 landmarks)
        # ======================================================

        motionbert_input = self.selector.to_h36m_17(
            normalized_33
        )

        # ======================================================
        # Step 3: Feature branch
        # (selected landmarks only)
        # ======================================================

        selected_landmarks = self.selector.select_landmarks(
            normalized_33
        )

        # ======================================================
        # Step 4: Build aligned windows
        # ======================================================

        windows = self.builder.build_sequence(
            motionbert_input=motionbert_input,
            selected_landmarks=selected_landmarks,
        )

        # ======================================================
        # Step 5: Extract features
        # ======================================================

        enriched_windows = []

        for window in windows:

            features = self.feature_extractor.extract_features(
                window["landmarks"]
            )

            enriched_windows.append(
                {
                    "motionbert_input": window["motionbert_input"],
                    "landmarks": window["landmarks"],
                    "features": features,
                    "window_start": window["window_start"],
                    "window_end": window["window_end"],
                    "exercise_id": self.spec["exercise"]["id"],
                    "exercise_name": self.spec["exercise"]["name"],
                }
            )

        return enriched_windows

    def process_npy_file(self, input_path: str | Path, output_path: str | Path | None = None) -> list:
        input_path = Path(input_path)
        raw_landmarks_33 = np.load(input_path, allow_pickle=False)
        windows = self.process_landmarks(raw_landmarks_33)

        if output_path is not None:
            self.save_windows(output_path, windows, source_name=input_path.stem)

        return windows

    def save_windows(self, output_path: str | Path, windows: list, source_name: str | None = None) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = np.array(windows, dtype=object) if windows else np.empty((0,), dtype=object)

        np.savez_compressed(
            output_path,
            windows=payload,
            source_name=source_name or output_path.stem,
            exercise_id=self.spec["exercise"]["id"],
            exercise_name=self.spec["exercise"]["name"],
            schema_version=self.spec.get("metadata", {}).get("schema_version", "unknown"),
            window_size=self.builder.window_size,
            step_size=self.builder.step_size,
        )

    def process_folder(self, input_folder: str | Path, output_folder: str | Path) -> None:
        input_folder = Path(input_folder)
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        for npy_file in sorted(input_folder.glob("*.npy")):
            out_file = output_folder / f"{npy_file.stem}.npz"
            if out_file.exists():
                continue

            windows = self.process_npy_file(npy_file)
            self.save_windows(out_file, windows, source_name=npy_file.stem)

    def process_dataset(self, processed_root: str | Path, sequences_root: str | Path) -> None:
        processed_root = Path(processed_root)
        sequences_root = Path(sequences_root)

        for exercise_folder in sorted(processed_root.iterdir()):
            if not exercise_folder.is_dir():
                continue

            self.process_folder(exercise_folder, sequences_root / exercise_folder.name)
