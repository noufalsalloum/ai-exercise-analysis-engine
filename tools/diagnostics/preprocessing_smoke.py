from __future__ import annotations

import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# ==========================================================
# Make project root importable
# ==========================================================
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing.pose_extractor import PoseExtractor
from preprocessing.landmark_selector import LandmarkSelector
from preprocessing.interpolation import Interpolator
from preprocessing.smoothing import Smoother
from preprocessing.normalizer import Normalizer
from preprocessing.feature_extractor import FeatureExtractor
from preprocessing.sequence_builder import SequenceBuilder
from preprocessing.preprocessor import PreprocessorController


# ==========================================================
# Helpers
# ==========================================================
def section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def require(condition: bool, message: str) -> None:
    assert condition, message


def choose_sample_video(raw_exercise_dir: Path) -> Path:
    videos = sorted(raw_exercise_dir.glob("*.mp4"))
    require(len(videos) > 0, f"No .mp4 files found in:\n{raw_exercise_dir}")
    return videos[0]


# ==========================================================
# Paths
# ==========================================================
TARGET_EXERCISE = "plank"

PROJECT_ROOT = ROOT
CONFIG_PATH = PROJECT_ROOT / "configs" / f"{TARGET_EXERCISE}.json"
MODEL_PATH = Path(r"C:\MediaPipe\pose_landmarker_full.task")

RAW_EXERCISE_DIR = PROJECT_ROOT / "datasets" / "raw" / TARGET_EXERCISE

# Work inside a temp folder so we do not touch your real outputs
# ==========================================================
# Main test
# ==========================================================
def main() -> None:
    start_time = time.perf_counter()

    require(CONFIG_PATH.exists(), f"Config not found:\n{CONFIG_PATH}")
    require(MODEL_PATH.exists(), f"MediaPipe model not found:\n{MODEL_PATH}")
    require(RAW_EXERCISE_DIR.exists(), f"Raw folder not found:\n{RAW_EXERCISE_DIR}")

    sample_video = choose_sample_video(RAW_EXERCISE_DIR)

    print("Sample video:", sample_video.name)
    print("Config path  :", CONFIG_PATH)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        copied_raw_root = tmp / "raw_copy"
        copied_processed_root = tmp / "processed"
        copied_sequences_root = tmp / "sequences"

        copied_raw_exercise_dir = copied_raw_root / TARGET_EXERCISE
        copied_raw_exercise_dir.mkdir(parents=True, exist_ok=True)

        # copy one sample video so we can test dataset-level pose extraction
        copied_sample_video = copied_raw_exercise_dir / sample_video.name
        shutil.copy2(sample_video, copied_sample_video)

        # Controller
        controller = PreprocessorController(
            spec_path=str(CONFIG_PATH),
            window_size=30,
            step_size=10,
            pose_model_path=str(MODEL_PATH),
        )

        # ======================================================
        # 1) PoseExtractor - direct extraction
        # ======================================================
        section("1) PoseExtractor - direct extraction")

        extractor = controller.pose_extractor
        require(extractor is not None, "PoseExtractor was not initialized.")

        raw_landmarks = extractor.extract_from_video(str(sample_video))

        print("Type :", type(raw_landmarks))
        print("Shape :", raw_landmarks.shape)
        print("Dtype :", raw_landmarks.dtype)

        require(isinstance(raw_landmarks, np.ndarray), "PoseExtractor output is not numpy.ndarray")
        require(raw_landmarks.ndim == 3, "PoseExtractor output must be 3D")
        require(raw_landmarks.shape[1] == 33, "PoseExtractor must return 33 landmarks")
        require(raw_landmarks.shape[2] == 4, "PoseExtractor must return (x, y, z, visibility)")
        require(raw_landmarks.dtype == np.float32, "PoseExtractor output dtype must be float32")
        require(np.isfinite(raw_landmarks).all(), "PoseExtractor output contains NaN/Inf")

        # ======================================================
        # 2) PoseExtractor - dataset processing
        # ======================================================
        section("2) PoseExtractor - dataset processing")

        extractor.process_dataset(
            raw_root=str(copied_raw_root),
            processed_root=str(copied_processed_root),
        )

        processed_file = copied_processed_root / TARGET_EXERCISE / f"{sample_video.stem}.npy"
        require(processed_file.exists(), f"Processed .npy not found:\n{processed_file}")

        saved_processed = np.load(processed_file)
        print("Saved processed shape:", saved_processed.shape)
        print("Saved processed dtype:", saved_processed.dtype)

        require(saved_processed.shape == raw_landmarks.shape, "Saved processed file shape mismatch")
        require(saved_processed.dtype == np.float32, "Saved processed file dtype must be float32")

        # ======================================================
        # 3) LandmarkSelector
        # ======================================================
        section("3) LandmarkSelector")

        selector = controller.selector
        selected_names = selector.load_selected_landmarks()

        print("Selected landmarks count:", len(selected_names))
        print("Selected landmarks list  :", selected_names)

        selected = selector.select_landmarks(raw_landmarks)
        print("Selected shape:", selected.shape)

        require(selected.shape[0] == raw_landmarks.shape[0], "Selected frames count mismatch")
        require(selected.shape[1] == len(selected_names), "Selected landmarks count mismatch")

        first_name = selected_names[0]
        one_landmark = selector.extract_landmark(raw_landmarks, first_name)
        print(f"Extract landmark '{first_name}' shape:", one_landmark.shape)

        require(one_landmark.shape[0] == raw_landmarks.shape[0], "extract_landmark frames mismatch")

        h36m = selector.to_h36m_17(raw_landmarks)
        print("H36M shape:", h36m.shape)

        require(h36m.shape[0] == raw_landmarks.shape[0], "H36M frames count mismatch")
        require(h36m.shape[1] == 17, "H36M must have 17 joints")
        require(h36m.shape[2] == 3, "H36M must have 3 coordinates")

        # ======================================================
        # 4) Interpolator
        # ======================================================
        section("4) Interpolator")

        interpolator = controller.interpolator

        interp_sample = selected[: min(60, len(selected))].copy()

        # simulate a missing segment
        if interp_sample.shape[0] >= 8 and interp_sample.shape[1] > 0 and interp_sample.shape[2] >= 4:
            interp_sample[2:6, 0, :3] = 0.0
            interp_sample[2:6, 0, 3] = 0.0

        missing_before = interpolator.find_missing(interp_sample)
        interpolated = interpolator.interpolate_sequence(interp_sample)

        print("Missing mask shape:", missing_before.shape)
        print("Interpolated shape :", interpolated.shape)

        require(interpolated.shape == interp_sample.shape, "Interpolation changed shape")
        require(np.isfinite(interpolated).all(), "Interpolation output contains NaN/Inf")
        require(not np.isnan(interpolated[:, :, :3]).any(), "Interpolation output still has NaNs in coordinates")

        # ======================================================
        # 5) Smoother
        # ======================================================
        section("5) Smoother")

        smoother = controller.smoother

        smoothed_ma = smoother.moving_average(interpolated)
        smoothed_kalman = smoother.kalman_filter(interpolated)
        smoothed_auto = smoother.smooth_sequence(interpolated, method="moving_average")
        smoothed_auto_k = smoother.smooth_sequence(interpolated, method="kalman")

        print("Moving average shape:", smoothed_ma.shape)
        print("Kalman placeholder shape:", smoothed_kalman.shape)

        require(smoothed_ma.shape == interpolated.shape, "moving_average changed shape")
        require(smoothed_kalman.shape == interpolated.shape, "kalman_filter changed shape")
        require(smoothed_auto.shape == interpolated.shape, "smooth_sequence(moving_average) changed shape")
        require(smoothed_auto_k.shape == interpolated.shape, "smooth_sequence(kalman) changed shape")

        # ======================================================
        # 6) Normalizer
        # ======================================================
        section("6) Normalizer")

        normalizer = controller.normalizer

        single_frame = smoothed_ma[0]
        centered = normalizer.center_pose(single_frame)
        scaled = normalizer.scale_pose(centered)
        normalized = normalizer.normalize_coordinates(smoothed_ma)

        print("Centered frame shape :", centered.shape)
        print("Scaled frame shape   :", scaled.shape)
        print("Normalized sequence  :", normalized.shape)

        require(centered.shape == single_frame.shape, "center_pose changed frame shape")
        require(scaled.shape == single_frame.shape, "scale_pose changed frame shape")
        require(normalized.shape == smoothed_ma.shape, "normalize_coordinates changed sequence shape")
        require(np.isfinite(normalized).all(), "Normalization output contains NaN/Inf")

        # center should make mean close to zero for xyz
        coords_mean = normalized[:, :, :3].mean(axis=1)
        require(
            np.allclose(coords_mean, 0.0, atol=1e-4),
            "Normalized coordinates are not centered around zero"
        )

        # ======================================================
        # 7) FeatureExtractor
        # ======================================================
        section("7) FeatureExtractor")

        feature_extractor = controller.feature_extractor

        # Individual functions
        frame_angles = feature_extractor.calculate_angles(normalized[0])
        velocity = feature_extractor.calculate_velocity(normalized)
        rom = feature_extractor.calculate_rom([feature_extractor.calculate_angles(normalized[i]) for i in range(normalized.shape[0])])
        trajectory = feature_extractor.calculate_trajectory(normalized)

        print("Angles dict keys:", list(frame_angles.keys()))
        print("Velocity shape   :", velocity.shape)
        print("ROM keys         :", list(rom.keys()))
        print("Trajectory shape :", trajectory.shape)

        require(isinstance(frame_angles, dict), "calculate_angles must return dict")
        require(velocity.shape == (normalized.shape[0], normalized.shape[1], 3), "calculate_velocity shape mismatch")
        require(isinstance(rom, dict), "calculate_rom must return dict")
        require(trajectory.shape == (normalized.shape[0], normalized.shape[1], 2), "calculate_trajectory shape mismatch")

        # Full extraction
        features = feature_extractor.extract_features(normalized)
        print("Feature keys:", list(features.keys()))

        expected_feature_keys = {
            "landmarks",
            "angles",
            "velocity",
            "angular_velocity",
            "rom",
            "trajectory",
        }
        require(expected_feature_keys.issubset(features.keys()), "FeatureExtractor missing expected keys")
        require(features["landmarks"].shape == normalized.shape, "features['landmarks'] shape mismatch")
        require(len(features["angles"]) == normalized.shape[0], "angles list length mismatch")
        require(features["velocity"].shape == (normalized.shape[0], normalized.shape[1], 3), "features['velocity'] shape mismatch")
        require(features["angular_velocity"].shape == (normalized.shape[0], normalized.shape[1], 3), "features['angular_velocity'] shape mismatch")
        require(features["trajectory"].shape == (normalized.shape[0], normalized.shape[1], 2), "features['trajectory'] shape mismatch")
        require(isinstance(features["rom"], dict), "features['rom'] must be dict")

        # ======================================================
        # 8) SequenceBuilder
        # ======================================================
        section("8) SequenceBuilder")

        builder = controller.builder

        padded_h36m = builder.pad_sequence(h36m, 30)
        windows = builder.build_sequence(
            motionbert_input=h36m,
            selected_landmarks=selected,
        )

        print("Padded H36M shape:", padded_h36m.shape)
        print("Number of windows :", len(windows))

        require(padded_h36m.shape == (30, 17, 3), "pad_sequence output shape mismatch")
        require(len(windows) > 0, "SequenceBuilder returned no windows")

        first_window = windows[0]
        print("First window keys:", list(first_window.keys()))

        require("motionbert_input" in first_window, "Window missing motionbert_input")
        require("landmarks" in first_window, "Window missing landmarks")
        require("window_start" in first_window, "Window missing window_start")
        require("window_end" in first_window, "Window missing window_end")

        require(first_window["motionbert_input"].shape[1:] == (17, 3), "motionbert_input window shape mismatch")
        require(first_window["landmarks"].shape[1:] == selected.shape[1:], "landmarks window shape mismatch")

        # ======================================================
        # 9) PreprocessorController - end-to-end from raw landmarks
        # ======================================================
        section("9) PreprocessorController - end-to-end from raw landmarks")

        enriched_windows = controller.process_landmarks(raw_landmarks)
        print("Enriched windows count:", len(enriched_windows))

        require(len(enriched_windows) > 0, "process_landmarks returned no windows")

        e0 = enriched_windows[0]
        print("First enriched window keys:", list(e0.keys()))

        required_enriched_keys = {
            "motionbert_input",
            "landmarks",
            "features",
            "window_start",
            "window_end",
            "exercise_id",
            "exercise_name",
        }
        require(required_enriched_keys.issubset(e0.keys()), "Enriched window missing required keys")
        require(e0["motionbert_input"].shape[1:] == (17, 3), "Enriched motionbert_input shape mismatch")
        require(e0["landmarks"].shape[1:] == selected.shape[1:], "Enriched landmarks shape mismatch")
        require(isinstance(e0["features"], dict), "Enriched features must be dict")

        # Also test process_video (full raw video -> pose -> preprocessing)
        video_windows = controller.process_video(str(sample_video))
        print("process_video windows count:", len(video_windows))

        require(len(video_windows) > 0, "process_video returned no windows")

        # ======================================================
        # 10) PreprocessorController - file pipeline
        # ======================================================
        section("10) PreprocessorController - file pipeline")

        direct_sequences_dir = tmp / "direct_sequences"
        direct_sequence_file = direct_sequences_dir / TARGET_EXERCISE / f"{sample_video.stem}.npz"

        controller.process_npy_file(
            input_path=processed_file,
            output_path=direct_sequence_file,
        )

        require(direct_sequence_file.exists(), f"Direct sequence file not found:\n{direct_sequence_file}")

        with np.load(direct_sequence_file, allow_pickle=True) as loaded_direct:
            direct_windows = loaded_direct["windows"]

        print("Direct npz windows count:", len(direct_windows))

        require(len(direct_windows) > 0, "Direct npz contains no windows")

        direct_first = direct_windows[0]
        print("Direct first window keys:", list(direct_first.keys()))

        require("motionbert_input" in direct_first, "Direct npz window missing motionbert_input")
        require("landmarks" in direct_first, "Direct npz window missing landmarks")
        require("features" in direct_first, "Direct npz window missing features")

        # ======================================================
        # 11) PreprocessorController - dataset pipeline
        # ======================================================
        section("11) PreprocessorController - dataset pipeline")

        controller.process_dataset(
            processed_root=copied_processed_root,
            sequences_root=copied_sequences_root,
        )

        dataset_sequence_file = copied_sequences_root / TARGET_EXERCISE / f"{sample_video.stem}.npz"
        require(dataset_sequence_file.exists(), f"Dataset sequence file not found:\n{dataset_sequence_file}")

        with np.load(dataset_sequence_file, allow_pickle=True) as loaded_dataset:
            dataset_windows = loaded_dataset["windows"]

        print("Dataset npz windows count:", len(dataset_windows))

        require(len(dataset_windows) > 0, "Dataset npz contains no windows")

        dataset_first = dataset_windows[0]
        print("Dataset first window keys:", list(dataset_first.keys()))

        require("motionbert_input" in dataset_first, "Dataset npz window missing motionbert_input")
        require("landmarks" in dataset_first, "Dataset npz window missing landmarks")
        require("features" in dataset_first, "Dataset npz window missing features")

        # ======================================================
        # Final summary
        # ======================================================
        elapsed = time.perf_counter() - start_time

        section("FINAL SUMMARY")
        print("✅ All preprocessing tests passed")
        print(f"Exercise tested       : {TARGET_EXERCISE}")
        print(f"Sample video          : {sample_video.name}")
        print(f"Raw landmarks shape   : {raw_landmarks.shape}")
        print(f"Selected landmarks    : {selected.shape}")
        print(f"H36M shape            : {h36m.shape}")
        print(f"Windows (direct)      : {len(direct_windows)}")
        print(f"Windows (dataset)     : {len(dataset_windows)}")
        print(f"Execution time (sec)  : {elapsed:.2f}")


if __name__ == "__main__":
    main()
