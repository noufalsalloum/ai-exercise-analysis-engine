from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime
from typing import Any

from inference.predict import ExercisePredictor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "latest_epoch.bin"
DEFAULT_CONFIGS_DIR = PROJECT_ROOT / "configs"
DEFAULT_PROTOTYPES_DIR = PROJECT_ROOT / "prototype_store"
DEFAULT_POSE_MODEL = Path(r"C:\MediaPipe\pose_landmarker_full.task")

EXERCISE_CHOICES = {
    "1": ("pushup", "Push-up"),
    "2": ("squat", "Squat"),
    "3": ("plank", "Plank"),
    "4": ("pullup", "Pull-up"),
    "5": ("lunge", "Lunge"),
}

RAW_EXERCISE_FOLDERS = {
    "pushup": "push-up",
    "squat": "squat",
    "plank": "plank",
    "pullup": "pull Up",
    "lunge": "lunge",
}


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for end-to-end video inference."""

    parser = argparse.ArgumentParser(
        description="Run the AI Exercise Quality Assessment pipeline on a video."
    )
    parser.add_argument(
        "video_path",
        type=Path,
        nargs="?",
        help="Path to the input video. Omit it to open interactive mode.",
    )
    parser.add_argument(
        "exercise_id",
        nargs="?",
        choices=("plank", "squat", "pushup", "pullup", "lunge"),
        help="Exercise expert to use.",
    )
    parser.add_argument(
        "--motionbert-checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to the audited MotionBERT-Lite checkpoint.",
    )
    parser.add_argument(
        "--pose-model",
        type=Path,
        default=DEFAULT_POSE_MODEL,
        help="Path to MediaPipe pose_landmarker_full.task.",
    )
    parser.add_argument(
        "--aqa-checkpoint",
        type=Path,
        default=None,
        help="Optional trained full-AQA checkpoint.",
    )
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=DEFAULT_CONFIGS_DIR,
        help="Directory containing exercise JSON configs.",
    )
    parser.add_argument(
        "--prototypes-dir",
        type=Path,
        default=DEFAULT_PROTOTYPES_DIR,
        help="Directory containing <exercise_id>.npz prototypes.",
    )
    parser.add_argument(
        "--pose-cache",
        type=Path,
        default=None,
        help="Optional NPZ pose cache to read or create.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional positive frame limit for faster diagnostics.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="MotionBERT window batch size.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or a device such as cuda:0.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the resulting JSON file.",
    )
    return parser


def validate_paths(args: argparse.Namespace) -> None:
    """Fail early with actionable messages for required local assets."""

    required_files = {
        "video": args.video_path,
        "MotionBERT checkpoint": args.motionbert_checkpoint,
        "MediaPipe pose model": args.pose_model,
        "exercise config": args.configs_dir / f"{args.exercise_id}.json",
    }
    if args.aqa_checkpoint is not None:
        required_files["AQA checkpoint"] = args.aqa_checkpoint

    missing = [f"{name}: {path}" for name, path in required_files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files:\n- " + "\n- ".join(missing))
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")


def choose_exercise() -> str:
    """Ask for an exercise, defaulting to Push-up for an empty answer."""

    print("\nاختر التمرين:")
    for number, (_, display_name) in EXERCISE_CHOICES.items():
        default = " (الافتراضي)" if number == "1" else ""
        print(f"  {number}) {display_name}{default}")

    while True:
        choice = input("\nرقم التمرين [1]: ").strip() or "1"
        if choice in EXERCISE_CHOICES:
            return EXERCISE_CHOICES[choice][0]
        print("اختيار غير صحيح. أدخل رقمًا من 1 إلى 5.")


def choose_input_mode() -> str:
    """Choose between a live camera capture and an uploaded video."""

    print("\nاختر مصدر المقطع:")
    print("  1) فتح الكاميرا")
    print("  2) اختيار مقطع من الجهاز (الافتراضي)")
    while True:
        choice = input("\nاختيارك [2]: ").strip() or "2"
        if choice in {"1", "camera"}:
            return "camera"
        if choice in {"2", "video", "upload"}:
            return "video"
        print("اختيار غير صحيح. أدخل 1 أو 2.")


def choose_video_file(exercise_id: str) -> Path:
    """Open a native file picker, with a terminal fallback if unavailable."""

    initial_directory = (
        PROJECT_ROOT
        / "datasets"
        / "raw"
        / RAW_EXERCISE_FOLDERS[exercise_id]
    )
    selected = ""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askopenfilename(
            title="اختر مقطع التمرين",
            initialdir=str(initial_directory if initial_directory.is_dir() else PROJECT_ROOT),
            filetypes=(
                ("Video files", "*.mp4 *.mov *.avi *.mkv *.webm *.m4v"),
                ("All files", "*.*"),
            ),
        )
        root.destroy()
    except (ImportError, RuntimeError):
        selected = input("ألصق المسار الكامل للمقطع: ").strip().strip('"')
    except tk.TclError:
        selected = input("ألصق المسار الكامل للمقطع: ").strip().strip('"')

    if not selected:
        raise SystemExit("لم يتم اختيار أي مقطع.")
    return Path(selected).expanduser().resolve()


def capture_camera_video(exercise_id: str, camera_index: int = 0) -> Path:
    """Record a camera clip until Q/ESC, then return its saved path."""

    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "فتح الكاميرا يحتاج OpenCV. ثبّت opencv-python في بيئة VS Code."
        ) from exc

    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(
            "تعذر فتح الكاميرا. تحقق من صلاحية الكاميرا ومن إغلاق البرامج الأخرى لها."
        )

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 1.0 or fps > 120.0:
        fps = 30.0

    captured_dir = PROJECT_ROOT / "datasets" / "captured" / exercise_id
    captured_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = captured_dir / f"{exercise_id}_{timestamp}.mp4"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("تعذر إنشاء ملف تسجيل الكاميرا.")

    frame_count = 0
    print("\nتم فتح الكاميرا. نفّذ التمرين، ثم اضغط Q أو ESC لإيقاف التسجيل.")
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            writer.write(frame)
            frame_count += 1
            cv2.putText(
                frame,
                "Recording - press Q or ESC to analyze",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
            cv2.imshow("AI Exercise Capture", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        capture.release()
        writer.release()
        cv2.destroyAllWindows()

    if frame_count == 0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError("لم تُلتقط أي صورة من الكاميرا.")
    print(f"تم حفظ التسجيل: {output_path}")
    return output_path


def build_interactive_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Create normal CLI arguments from the three-step VS Code workflow."""

    args = parser.parse_args([])
    print("=" * 66)
    print("AI Exercise Quality Assessment")
    print("=" * 66)
    args.exercise_id = choose_exercise()
    mode = choose_input_mode()
    args.video_path = (
        capture_camera_video(args.exercise_id)
        if mode == "camera"
        else choose_video_file(args.exercise_id)
    )

    cache_key = hashlib.sha1(str(args.video_path).encode("utf-8")).hexdigest()[:10]
    args.pose_cache = (
        PROJECT_ROOT
        / "datasets"
        / "pose_cache"
        / f"{args.exercise_id}_{args.video_path.stem}_{cache_key}.npz"
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output = (
        PROJECT_ROOT
        / "results"
        / f"{args.exercise_id}_{args.video_path.stem}_{timestamp}.json"
    )
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Construct the persistent model once and run one input video."""

    validate_paths(args)
    predictor = ExercisePredictor(
        motionbert_checkpoint=args.motionbert_checkpoint,
        pose_model_path=args.pose_model,
        configs_dir=args.configs_dir,
        aqa_checkpoint=args.aqa_checkpoint,
        prototypes_dir=args.prototypes_dir,
        device=args.device,
        batch_size=args.batch_size,
    )
    return predictor.predict_video(
        video_path=args.video_path,
        exercise_id=args.exercise_id,
        max_frames=args.max_frames,
        pose_cache_path=args.pose_cache,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.video_path is None and args.exercise_id is None:
        args = build_interactive_args(parser)
    elif args.video_path is None or args.exercise_id is None:
        parser.error("video_path and exercise_id must be provided together.")

    print("\nجاري تشغيل MediaPipe وMotionBERT والـExercise Expert...")
    print("قد تستغرق المعالجة وقتًا على CPU. سيُستخدم Pose Cache تلقائيًا لاحقًا.\n")
    result = run(args)
    output_json = json.dumps(result, ensure_ascii=False, indent=2)
    print(output_json)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_json + "\n", encoding="utf-8")
        print(f"\nSaved result to: {args.output}")


if __name__ == "__main__":
    main()
