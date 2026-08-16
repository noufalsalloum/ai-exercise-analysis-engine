from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class PoseExtractor:
    """
    Extracts MediaPipe BlazePose landmarks from video files
    and saves them as .npy arrays with shape:

        (num_frames, 33, 4)

    where:

        x
        y
        z
        visibility
    """

    VIDEO_EXTENSIONS = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".wmv",
    }

    def __init__(self, model_path: str):

        self.model_path = str(model_path)
        self.last_fps: float | None = None

        base_options = python.BaseOptions(
            model_asset_path=self.model_path
        )

        self.options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            output_segmentation_masks=False,
        )

    # =====================================================
    # Extract landmarks from one video
    # =====================================================

    def extract_from_video(
        self,
        video_path: str | Path,
        max_frames: int | None = None,
    ) -> np.ndarray:

        if max_frames is not None and max_frames <= 0:
            raise ValueError("max_frames must be positive when provided.")

        video_path = str(video_path)

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise FileNotFoundError(
                f"Cannot open video:\n{video_path}"
            )

        landmarker = vision.PoseLandmarker.create_from_options(
            self.options
        )

        try:

            fps = cap.get(cv2.CAP_PROP_FPS)

            if fps <= 0:
                fps = 30.0
            self.last_fps = float(fps)

            frames_landmarks: List[np.ndarray] = []

            frame_idx = 0

            last_valid = None

            while True:

                if max_frames is not None and frame_idx >= max_frames:
                    break

                ret, frame = cap.read()

                if not ret:
                    break

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                timestamp_ms = int(
                    (frame_idx / fps) * 1000
                )

                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms
                )

                if result.pose_landmarks:

                    landmarks = result.pose_landmarks[0]

                    frame_landmarks = np.array(
                        [
                            [
                                lm.x,
                                lm.y,
                                lm.z,
                                lm.visibility
                            ]
                            for lm in landmarks
                        ],
                        dtype=np.float32
                    )

                    last_valid = frame_landmarks.copy()

                else:

                    if last_valid is not None:

                        frame_landmarks = last_valid.copy()

                    else:

                        frame_landmarks = np.zeros(
                            (33, 4),
                            dtype=np.float32
                        )

                frames_landmarks.append(frame_landmarks)

                frame_idx += 1

        finally:

            cap.release()
            landmarker.close()

        if len(frames_landmarks) == 0:

            raise ValueError(
                f"No frames extracted from:\n{video_path}"
            )

        return np.stack(
            frames_landmarks,
            axis=0
        )

    # =====================================================
    # Process one video
    # =====================================================

    def process_video_file(
        self,
        video_path,
        output_path
    ) -> None:

        print(f"\nProcessing: {Path(video_path).name}")

        landmarks = self.extract_from_video(video_path)

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        np.save(
            output_path,
            landmarks
        )

        print(f"Saved -> {output_path.name}")
        print(f"Shape -> {landmarks.shape}")

    # =====================================================
    # Process one exercise folder
    # =====================================================

    def process_folder(
        self,
        input_folder,
        output_folder
    ) -> None:

        input_folder = Path(input_folder)

        output_folder = Path(output_folder)

        output_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        for video in sorted(input_folder.iterdir()):

            if (
                video.is_file()
                and
                video.suffix.lower() in self.VIDEO_EXTENSIONS
            ):

                output_file = (
                    output_folder /
                    f"{video.stem}.npy"
                )

                self.process_video_file(
                    video,
                    output_file
                )

    # =====================================================
    # Process entire dataset
    # =====================================================

    def process_dataset(
        self,
        raw_root,
        processed_root
    ) -> None:

        raw_root = Path(raw_root)

        processed_root = Path(processed_root)

        print("=" * 70)
        print("Processing Dataset")
        print("=" * 70)

        for exercise_folder in sorted(raw_root.iterdir()):

            if not exercise_folder.is_dir():
                continue

            print("\n" + "=" * 70)
            print(f"Exercise : {exercise_folder.name}")
            print("=" * 70)

            output_folder = (
                processed_root /
                exercise_folder.name
            )

            self.process_folder(
                exercise_folder,
                output_folder
            )

        print("\n" + "=" * 70)
        print("Finished Successfully")
        print("=" * 70)


# =====================================================
# Test
# =====================================================

if __name__ == "__main__":

    extractor = PoseExtractor(

        model_path=r"C:\MediaPipe\pose_landmarker_full.task"

    )

    extractor.process_dataset(

        raw_root=r"C:\Users\JoudA\OneDrive\سطح المكتب\COOP - JOUD ALSHEHRI\code\ai_engine\datasets\raw",

        processed_root=r"C:\Users\JoudA\OneDrive\سطح المكتب\COOP - JOUD ALSHEHRI\code\ai_engine\datasets\processed",

    )
