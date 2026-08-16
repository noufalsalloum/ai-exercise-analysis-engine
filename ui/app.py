from __future__ import annotations

import argparse
import json
import queue
import time
import zlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageTk

from audio_feedback.service import AudioFeedbackService

from application.exercise_registry import ExerciseDefinition, ExerciseRegistry, PROJECT_ROOT
from application.presentation_contract import (
    ALLOWED_PROCESSING_STATES,
    COMPLETED,
    PROCESSING,
    READY,
    WAITING_REPETITION,
)
from application.runtime_router import FamilyRuntimeRouter
from application.workers import AnalysisWorker
from input_sources.frame_sources import CameraFrameSource, VideoFrameSource
from input_sources.pose_stream import PoseStreamProcessor

from . import strings
from .squat_ai_presentation import (
    present_live_squat_ai,
    present_squat_dashboard,
)
from .family_parity_presentation import present_live_family_status, present_family_dashboard, present_shared_dashboard
from .theme import COLORS, FONTS, configure_ttk_theme


DEFAULT_POSE_MODEL = Path(r"C:\MediaPipe\pose_landmarker_full.task")
VIDEO_FILE_TYPES = (
    ("Video files", "*.mp4 *.mov *.avi *.mkv *.wmv *.m4v"),
    ("All files", "*.*"),
)


class ExerciseAnalysisApp:
    """Responsive Tkinter application for uploaded and continuous live analysis."""

    def __init__(
        self,
        root: Any,
        registry: ExerciseRegistry | None = None,
        pose_model_path: str | Path = DEFAULT_POSE_MODEL,
        camera_index: int = 0,
    ) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.registry = registry or ExerciseRegistry()
        self.pose_model_path = Path(pose_model_path)
        self.camera_index = int(camera_index)
        self.selected_exercise: ExerciseDefinition | None = None
        self.selected_view: Any = None
        self.worker: AnalysisWorker | None = None
        self.events: queue.Queue[dict] = queue.Queue(maxsize=4)
        self._photo: ImageTk.PhotoImage | None = None
        self._last_frame: np.ndarray | None = None
        self._image_item: int | None = None
        self._poll_id: str | None = None
        self._resize_id: str | None = None
        self._terminal_id: str | None = None
        self._closing = False
        self._last_result: dict[str, Any] | None = None
        self._last_worker_stats: dict[str, Any] | None = None
        self._dashboard_ai_view: dict[str, Any] | None = None
        self._ai_rep_text_widget: Any | None = None
        self._last_live_overlay_snapshot: dict[str, str] = {}
        self._active_input_mode: str | None = None
        self.current_screen = "initializing"
        self.automated_smoke = False
        self._smoke_error: str | None = None
        self._reset_render_diagnostics()

        self.root.title(strings.APP_TITLE)
        self.root.geometry("1280x800")
        self.root.minsize(900, 600)
        self.root.resizable(True, True)
        self.root.configure(bg=COLORS["main_background"])
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        configure_ttk_theme(self.ttk.Style(self.root), self.root)
        self.container = self.ttk.Frame(self.root, style="App.TFrame", padding=20)
        self.container.grid(row=0, column=0, sticky="nsew")
        self.show_exercise_selection()

    def _reset_render_diagnostics(self) -> None:
        self._displayed_frame_indices: list[int] = []
        self._frame_checksums: set[int] = set()
        self._photo_reference_kept = False
        self._placeholder_hidden_after_frame = False
        self._countdown_history: list[str] = []
        self._maximum_session_time = 0.0
        self._maximum_progress = 0.0
        self._smoke_error = None

    def _reset_session_presentation_state(self) -> None:
        """Clear UI-only session data so results cannot leak across sessions."""

        self._last_result = None
        self._last_worker_stats = None
        self._dashboard_ai_view = None
        self._ai_rep_text_widget = None
        self._last_live_overlay_snapshot = {}

    def _cancel_after(self, attribute: str) -> None:
        callback_id = getattr(self, attribute)
        if callback_id is None:
            return
        try:
            self.root.after_cancel(callback_id)
        except self.tk.TclError:
            pass
        setattr(self, attribute, None)

    def _clear(self, stop_worker: bool = False) -> None:
        self._cancel_after("_poll_id")
        self._cancel_after("_resize_id")
        self._cancel_after("_terminal_id")
        if stop_worker:
            self._stop_worker(wait=True)
        for child in self.container.winfo_children():
            child.destroy()
        for index in range(4):
            self.container.grid_rowconfigure(index, weight=0)
            self.container.grid_columnconfigure(index, weight=0)
        self.container.grid_columnconfigure(0, weight=1)
        self._photo = None
        self._last_frame = None
        self._image_item = None

    def _header(self, title: str, subtitle: str) -> None:
        header = self.ttk.Frame(self.container, style="App.TFrame")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        header.grid_columnconfigure(0, weight=1)
        style = "AppTitle.TLabel" if title == strings.APP_TITLE else "Heading.TLabel"
        self.ttk.Label(header, text=title, style=style).grid(row=0, column=0, sticky="w")
        self.ttk.Label(
            header,
            text=subtitle,
            style="Subtitle.TLabel",
            wraplength=1050,
        ).grid(row=1, column=0, sticky="w", pady=(3, 0))

    def show_exercise_selection(self) -> None:
        self._clear(stop_worker=True)
        self._reset_session_presentation_state()
        self.current_screen = "families"
        self.selected_exercise = None
        self._header(strings.APP_TITLE, strings.APP_SUBTITLE)
        self.container.grid_rowconfigure(1, weight=1)
        grid = self.ttk.Frame(self.container, style="App.TFrame")
        grid.grid(row=1, column=0, sticky="nsew")
        for column in range(3):
            grid.grid_columnconfigure(column, weight=1, uniform="family")
        for row in range(2):
            grid.grid_rowconfigure(row, weight=1, uniform="family")
        for index, exercise in enumerate(self.registry.main_families()):
            card = self.ttk.Frame(grid, style="Card.TFrame", padding=18)
            card.grid(
                row=index // 3,
                column=index % 3,
                padx=8,
                pady=8,
                sticky="nsew",
            )
            card.grid_columnconfigure(0, weight=1)
            self.ttk.Label(
                card,
                text=exercise.display_name,
                style="CardTitle.TLabel",
            ).grid(row=0, column=0, sticky="w")
            status = strings.STATUS_LABELS[exercise.status]
            badge_color = {
                "ready": COLORS["success"],
                "partial": COLORS["warning"],
                "development": COLORS["muted_text"],
            }[exercise.status]
            self.tk.Label(
                card,
                text=status,
                bg=badge_color,
                fg=COLORS["main_background"],
                font=FONTS["badge"],
                padx=9,
                pady=3,
            ).grid(row=1, column=0, sticky="w", pady=(9, 10))
            capabilities = [
                name.replace("_", " ").title()
                for name, available in exercise.capabilities.to_dict().items()
                if available
            ]
            text = ", ".join(capabilities) if capabilities else "No validated analysis yet"
            self.ttk.Label(
                card,
                text=text,
                style="CardText.TLabel",
                wraplength=310,
            ).grid(row=2, column=0, sticky="nw")
            button = self.ttk.Button(
                card,
                text="Select Family" if exercise.can_analyze else "In Development",
                style="Primary.TButton" if exercise.can_analyze else "Secondary.TButton",
                command=lambda item=exercise: self.select_exercise(item),
            )
            button.grid(row=3, column=0, sticky="sw", pady=(18, 0))
            if not exercise.can_analyze:
                button.configure(state="disabled")

    def select_exercise(self, exercise: ExerciseDefinition) -> None:
        if not exercise.can_analyze:
            self._show_development(exercise)
            return
        self.selected_exercise = exercise
        self.show_input_selection()

    def _show_development(self, exercise: ExerciseDefinition) -> None:
        from tkinter import messagebox

        messagebox.showinfo(
            exercise.display_name,
            "This exercise family is still in development.",
            parent=self.root,
        )

    def show_input_selection(self) -> None:
        exercise = self.selected_exercise
        if exercise is None:
            self.show_exercise_selection()
            return
        self._clear(stop_worker=True)
        self._reset_session_presentation_state()
        self.current_screen = "input_selection"
        self._header(
            exercise.display_name,
            "Choose an input source and confirm the camera view",
        )
        self.container.grid_rowconfigure(1, weight=1)
        panel = self.ttk.Frame(self.container, style="Card.TFrame", padding=24)
        panel.grid(row=1, column=0, sticky="new")
        panel.grid_columnconfigure(0, weight=1)
        self.ttk.Label(
            panel,
            text=f"Recommended view: {exercise.recommended_camera_view.title()}",
            style="CardTitle.TLabel",
        ).grid(row=0, column=0, sticky="w")
        self.ttk.Label(
            panel,
            text=exercise.positioning_instruction,
            style="CardText.TLabel",
            wraplength=980,
        ).grid(row=1, column=0, sticky="w", pady=(8, 14))
        next_row = 2
        if exercise.family_id == "pushup":
            self.ttk.Label(panel, text="Push-up variation", style="CardText.TLabel").grid(
                row=next_row, column=0, sticky="w"
            )
            variation_values = ("Standard / Floor Push-up", "Table / Incline Push-up")
            selected_variation = self.tk.StringVar(
                value=variation_values[1] if exercise.variation_id == "table_incline" else variation_values[0]
            )
            variation_box = self.ttk.Combobox(
                panel, textvariable=selected_variation, values=variation_values,
                state="readonly", width=28,
            )
            variation_box.grid(row=next_row + 1, column=0, sticky="w", pady=(4, 12))
            variation_box.bind(
                "<<ComboboxSelected>>",
                lambda _event: self._select_pushup_variation(selected_variation.get()),
            )
            next_row += 2
        self.ttk.Label(panel, text="Camera view", style="CardText.TLabel").grid(
            row=next_row,
            column=0,
            sticky="w",
        )
        self.selected_view = self.tk.StringVar(value=exercise.recommended_camera_view)
        self.ttk.Combobox(
            panel,
            textvariable=self.selected_view,
            values=exercise.allowed_camera_views,
            state="readonly",
            width=22,
        ).grid(row=next_row + 1, column=0, sticky="w", pady=(4, 18))
        actions = self.ttk.Frame(panel, style="Card.TFrame")
        actions.grid(row=next_row + 2, column=0, sticky="ew")
        actions.grid_columnconfigure(3, weight=1)
        self.ttk.Button(
            actions,
            text="Upload Video",
            style="Primary.TButton",
            command=self.choose_video,
        ).grid(row=0, column=0, padx=(0, 10))
        self.ttk.Button(
            actions,
            text="Real-Time Camera",
            style="Primary.TButton",
            command=self.start_realtime,
        ).grid(row=0, column=1)
        self.ttk.Button(
            actions,
            text="Back",
            style="Secondary.TButton",
            command=self.show_exercise_selection,
        ).grid(row=0, column=4, sticky="e")
        limitations = "\n".join(f"• {item}" for item in exercise.limitations)
        self.ttk.Label(
            self.container,
            text="Current limitations\n" + limitations,
            style="Body.TLabel",
            wraplength=1080,
        ).grid(row=2, column=0, sticky="w", pady=(18, 0))

    def _select_pushup_variation(self, display_name: str) -> None:
        exercise_id = "table_incline_pushup" if display_name.startswith("Table") else "pushup"
        selected = self.registry.get(exercise_id)
        if self.selected_exercise is None or self.selected_exercise.exercise_id != selected.exercise_id:
            self.selected_exercise = selected
            self.show_input_selection()

    def choose_video(self) -> None:
        from tkinter import filedialog

        selected = filedialog.askopenfilename(
            parent=self.root,
            title="Select an exercise video",
            initialdir=str(PROJECT_ROOT / "datasets"),
            filetypes=VIDEO_FILE_TYPES,
        )
        if selected:
            self.start_analysis("video", Path(selected))

    def start_realtime(self) -> None:
        self.start_analysis("realtime", None)

    def start_analysis(self, input_mode: str, video_path: Path | None) -> None:
        from tkinter import messagebox

        exercise = self.selected_exercise
        if exercise is None:
            return
        try:
            self.registry.require_runnable(exercise.exercise_id, input_mode)
            if not self.pose_model_path.is_file():
                raise FileNotFoundError(
                    f"MediaPipe model is missing: {self.pose_model_path}\n"
                    "Pass --pose-model with the correct pose_landmarker_full.task path."
                )
        except Exception as exc:
            messagebox.showerror("Cannot start analysis", str(exc), parent=self.root)
            return
        camera_view = str(self.selected_view.get())
        self._clear(stop_worker=True)
        self._reset_session_presentation_state()
        self.current_screen = "analysis"
        self._active_input_mode = input_mode
        self._reset_render_diagnostics()
        mode_label = "Uploaded Video" if input_mode == "video" else "Real-Time Camera"
        self._header(
            f"{exercise.display_name} — {mode_label}",
            strings.PREPARATION_GUIDANCE
            if input_mode == "realtime"
            else "Video playback follows the source FPS while analysis runs in the background",
        )
        self.container.grid_rowconfigure(1, weight=1)
        video_container = self.tk.Frame(
            self.container,
            bg=COLORS["video_background"],
            highlightbackground=COLORS["dark_turquoise"],
            highlightthickness=1,
        )
        video_container.grid(row=1, column=0, sticky="nsew")
        video_container.grid_rowconfigure(0, weight=1)
        video_container.grid_columnconfigure(0, weight=1)
        self.video_canvas = self.tk.Canvas(
            video_container,
            bg=COLORS["video_background"],
            highlightthickness=0,
        )
        self.video_canvas.grid(row=0, column=0, sticky="nsew")
        self._image_item = self.video_canvas.create_image(0, 0, anchor="center")
        self._placeholder_item = self.video_canvas.create_text(
            0,
            0,
            text="Opening video stream…",
            fill=COLORS["secondary_text"],
            font=FONTS["normal"],
        )
        self.video_canvas.bind("<Configure>", self._on_video_resize)

        official_live_label = "Hold Time" if exercise.family_id == "plank" else "Official Reps"
        overlay_values = {
            "Exercise": exercise.display_name,
            "Camera View": camera_view.title(),
            official_live_label: "0 s" if exercise.family_id == "plank" else "0",
            "Current Phase": READY,
            "Current Elbow Angle": strings.NOT_AVAILABLE,
            "Selected Side" if exercise.family_id == "pushup" else "Vertical Body Motion": strings.NOT_AVAILABLE,
            "Signal Confidence": strings.NOT_AVAILABLE,
            "Session Time": "0.0 s",
            "Processing State": READY,
        }
        pushup_learned_assessment = (
            exercise.family_id == "pushup"
            and exercise.assessment_checkpoint is not None
        )
        live_learned_ai = exercise.family_id == "squat" or pushup_learned_assessment
        if live_learned_ai:
            overlay_values.update(
                {
                    "AI Assessment": "AI ANALYSIS — EXPERIMENTAL",
                    "AI Detected Reps": "0",
                    "AI Processing": WAITING_REPETITION,
                    "Latest Rep": WAITING_REPETITION,
                    "Confidence": strings.NOT_AVAILABLE,
                    "Score": strings.NOT_AVAILABLE,
                    "Error": strings.NOT_AVAILABLE,
                }
            )
        else:
            overlay_values.update(
                {
                    "AI Assessment": strings.NOT_AVAILABLE,
                    "AI Detected Reps": strings.NOT_AVAILABLE,
                    "AI Processing": strings.NOT_AVAILABLE,
                    "Latest Rep": strings.NOT_AVAILABLE,
                    "Confidence": strings.NOT_AVAILABLE,
                    "Score": strings.NOT_AVAILABLE,
                    "Error": strings.NOT_AVAILABLE,
                }
            )
        if exercise.family_id == "pushup":
            overlay_values["Coverage"] = strings.NOT_AVAILABLE
        self.overlay_vars = {
            name: self.tk.StringVar(value=value)
            for name, value in overlay_values.items()
        }
        overlay = self.tk.Frame(
            video_container,
            bg=COLORS["elevated_panel"],
            highlightbackground=COLORS["cyan"],
            highlightthickness=1,
            padx=11,
            pady=9,
        )
        overlay.place(x=14, y=14, anchor="nw")
        for row, (name, variable) in enumerate(self.overlay_vars.items()):
            self.tk.Label(
                overlay,
                text=f"{name}:",
                bg=COLORS["elevated_panel"],
                fg=COLORS["secondary_text"],
                font=FONTS["overlay"],
                anchor="w",
            ).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=1)
            self.tk.Label(
                overlay,
                textvariable=variable,
                bg=COLORS["elevated_panel"],
                fg=COLORS["main_text"],
                font=FONTS["overlay_value"],
                anchor="w",
            ).grid(row=row, column=1, sticky="w", pady=1)

        self.countdown_var = self.tk.StringVar(value="")
        self.countdown_label = self.tk.Label(
            video_container,
            textvariable=self.countdown_var,
            bg=COLORS["video_background"],
            fg=COLORS["light_gold"],
            font=FONTS["countdown"],
            padx=18,
            pady=8,
        )

        footer = self.ttk.Frame(self.container, style="App.TFrame")
        footer.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        footer.grid_columnconfigure(3, weight=1)
        self.status_var = self.tk.StringVar(value="Starting")
        self.frame_var = self.tk.StringVar(value="Frame 0")
        self.progress_var = self.tk.DoubleVar(value=0.0)
        self.ttk.Button(
            footer,
            text="Back",
            style="Secondary.TButton",
            command=self._back_from_analysis,
        ).grid(row=0, column=0, padx=(0, 8))
        self.ttk.Button(
            footer,
            text="Stop Session",
            style="Primary.TButton",
            command=self.stop_session,
        ).grid(row=0, column=1, padx=(0, 14))
        self.ttk.Label(footer, textvariable=self.status_var, style="Body.TLabel").grid(
            row=0,
            column=2,
            sticky="w",
        )
        self.ttk.Label(footer, textvariable=self.frame_var, style="Body.TLabel").grid(
            row=0,
            column=3,
            sticky="e",
            padx=12,
        )
        self.progress = self.ttk.Progressbar(
            footer,
            variable=self.progress_var,
            maximum=100,
            length=250,
            style="App.Horizontal.TProgressbar",
        )
        self.progress.grid(row=0, column=4, sticky="e")

        # Uploaded playback must transport every analyzed frame. Live capture is
        # intentionally bounded so old camera frames are replaced instead of
        # creating visible latency.
        self.events = queue.Queue(maxsize=0 if input_mode == "video" else 4)
        source_factory = (
            (lambda: VideoFrameSource(video_path))
            if input_mode == "video" and video_path is not None
            else (lambda: CameraFrameSource(self.camera_index))
        )
        self.worker = AnalysisWorker(
            exercise=exercise,
            input_mode=input_mode,
            camera_view=camera_view,
            source_factory=source_factory,
            pose_factory=lambda: PoseStreamProcessor(self.pose_model_path),
            events=self.events,
            video_path=video_path,
            audio_feedback_factory=AudioFeedbackService.from_default_config,
        )
        self.worker.start()
        self._schedule_poll()

    def _back_from_analysis(self) -> None:
        self._stop_worker(wait=True)
        self.show_input_selection()

    def _schedule_poll(self) -> None:
        if not self._closing and self.container.winfo_exists():
            self._poll_id = self.root.after(20, self._poll_events)

    def _poll_events(self) -> None:
        from tkinter import messagebox

        self._poll_id = None
        terminal_event = False
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            event_type = event.get("type")
            if event_type == "frame":
                self._display_frame(event["frame"])
                self._displayed_frame_indices.append(int(event.get("frame_index", 0)))
                self._update_live_metrics(event.get("metrics", {}))
                progress = event.get("progress")
                if progress is not None:
                    self._maximum_progress = max(self._maximum_progress, float(progress))
                    self.progress_var.set(float(progress) * 100.0)
            elif event_type == "started":
                backend = event.get("backend")
                camera_index = event.get("camera_index")
                if event.get("input_mode") == "realtime":
                    self.status_var.set(f"Live camera {camera_index} via {backend}")
                else:
                    self.status_var.set(PROCESSING)
            elif event_type == "status":
                self.status_var.set(PROCESSING)
                self.overlay_vars["Processing State"].set(PROCESSING)
            elif event_type == "complete":
                self._last_result = event["result"]
                self._last_worker_stats = event.get("worker_stats")
                if (
                    self.selected_exercise
                    and self.selected_exercise.family_id == "pushup"
                ):
                    self._apply_family_live_ai(
                        self._last_result.get("experimental_ai"),
                        completed=True,
                    )
                self._save_session_result(self._last_result)
                self.status_var.set(COMPLETED)
                self.root.update_idletasks()
                self._terminal_id = self.root.after(180, self._show_completed_dashboard)
                terminal_event = True
                break
            elif event_type == "error":
                self._stop_worker(wait=True)
                self._smoke_error = str(event.get("message", "Unknown error"))
                if self.automated_smoke:
                    self.status_var.set(self._smoke_error)
                else:
                    messagebox.showerror(
                        "Analysis error",
                        self._smoke_error,
                        parent=self.root,
                    )
                    self.show_input_selection()
                terminal_event = True
                break
        if not terminal_event and self.worker is not None:
            self._schedule_poll()

    def _show_completed_dashboard(self) -> None:
        self._terminal_id = None
        if not self._closing and self._last_result is not None:
            self.show_dashboard(self._last_result)

    def _update_live_metrics(self, metrics: dict[str, Any]) -> None:
        angle = metrics.get("current_angle")
        confidence = metrics.get("confidence")
        vertical = metrics.get("vertical_body_motion")
        plank_live = bool(self.selected_exercise and self.selected_exercise.family_id == "plank")
        official_key = "Hold Time" if plank_live else "Official Reps"
        official_value = int(metrics.get("repetitions", 0))
        self.overlay_vars[official_key].set(f"{official_value} s" if plank_live else str(official_value))
        self.overlay_vars["Current Phase"].set(str(metrics.get("phase", READY)))
        self.overlay_vars["Current Elbow Angle"].set(
            f"{float(angle):.1f}°" if angle is not None else strings.NOT_AVAILABLE
        )
        family_specific = (
            "Selected Side"
            if self.selected_exercise and self.selected_exercise.family_id == "pushup"
            else "Vertical Body Motion"
        )
        if family_specific == "Selected Side":
            side = metrics.get("selected_side")
            value = str(side).title() if side else strings.NOT_AVAILABLE
        else:
            value = f"{float(vertical):+.3f}" if vertical is not None else strings.NOT_AVAILABLE
        self.overlay_vars[family_specific].set(value)
        self.overlay_vars["Signal Confidence"].set(
            f"{float(confidence):.2f}" if confidence is not None else strings.NOT_AVAILABLE
        )
        self.overlay_vars["Session Time"].set(
            f"{float(metrics.get('session_time', 0.0)):.1f} s"
        )
        self._maximum_session_time = max(
            self._maximum_session_time,
            float(metrics.get("session_time", 0.0)),
        )
        status = str(metrics.get("status", PROCESSING))
        if status not in ALLOWED_PROCESSING_STATES:
            status = PROCESSING
        self.overlay_vars["Processing State"].set(status)
        self.status_var.set(status)
        self.frame_var.set(f"Frame {int(metrics.get('frame_index', 0))}")
        experimental_ai = metrics.get("experimental_ai")
        if self.selected_exercise and self.selected_exercise.family_id == "squat":
            ai_view = present_live_squat_ai(
                experimental_ai if isinstance(experimental_ai, dict) else None
            )
            self.overlay_vars["AI Assessment"].set(ai_view["heading"])
            self.overlay_vars["AI Detected Reps"].set(ai_view["detected"])
            self.overlay_vars["AI Processing"].set(ai_view["processing"])
            self.overlay_vars["Latest Rep"].set(ai_view["last_rep"])
            self.overlay_vars["Confidence"].set(ai_view["confidence"])
            self.overlay_vars["Error"].set(ai_view["warning"])
        elif self.selected_exercise:
            self._apply_family_live_ai(experimental_ai)
        self._last_live_overlay_snapshot = {
            name: str(variable.get()) for name, variable in self.overlay_vars.items()
        }
        countdown = metrics.get("countdown") or ""
        self.countdown_var.set(str(countdown))
        if countdown and (not self._countdown_history or self._countdown_history[-1] != str(countdown)):
            self._countdown_history.append(str(countdown))
        if countdown:
            self.countdown_label.place(relx=0.5, rely=0.5, anchor="center")
        else:
            self.countdown_label.place_forget()

    def _apply_family_live_ai(
        self,
        experimental_ai: Any,
        *,
        completed: bool = False,
    ) -> None:
        """Bind one normalized family presentation snapshot to the live fields."""

        if self.selected_exercise is None:
            return
        family_view = present_live_family_status(
            self.selected_exercise.family_id,
            experimental_ai if isinstance(experimental_ai, dict) else None,
            variation_id=self.selected_exercise.variation_id,
            completed=completed,
        )
        self.overlay_vars["AI Assessment"].set(family_view["heading"])
        self.overlay_vars["AI Detected Reps"].set(family_view["detected"])
        self.overlay_vars["AI Processing"].set(family_view["processing"])
        self.overlay_vars["Latest Rep"].set(family_view["last"])
        self.overlay_vars["Confidence"].set(family_view["confidence"])
        self.overlay_vars["Score"].set(family_view["score"])
        self.overlay_vars["Error"].set(family_view["errors"])
        if "Coverage" in self.overlay_vars:
            self.overlay_vars["Coverage"].set(family_view["coverage"])
        self._last_live_overlay_snapshot = {
            name: str(variable.get()) for name, variable in self.overlay_vars.items()
        }

    @staticmethod
    def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
        """Convert an OpenCV BGR frame to the RGB contract expected by Pillow."""

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected a BGR frame with shape (H, W, 3).")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    @staticmethod
    def fitted_size(
        source_width: int,
        source_height: int,
        target_width: int,
        target_height: int,
    ) -> tuple[int, int]:
        """Return a non-distorting letterboxed size within the target area."""

        if min(source_width, source_height, target_width, target_height) <= 0:
            raise ValueError("Image and target dimensions must be positive.")
        scale = min(target_width / source_width, target_height / source_height)
        return max(1, round(source_width * scale)), max(1, round(source_height * scale))

    def _display_frame(self, frame: np.ndarray) -> None:
        self._last_frame = np.asarray(frame).copy()
        self._frame_checksums.add(zlib.adler32(self._last_frame.tobytes()))
        self._render_last_frame()
        self.video_canvas.itemconfigure(self._placeholder_item, state="hidden")
        self._placeholder_hidden_after_frame = (
            self.video_canvas.itemcget(self._placeholder_item, "state") == "hidden"
        )

    def _render_last_frame(self) -> None:
        if self._last_frame is None or not hasattr(self, "video_canvas"):
            return
        if not self.video_canvas.winfo_exists():
            return
        width = max(1, self.video_canvas.winfo_width())
        height = max(1, self.video_canvas.winfo_height())
        source_height, source_width = self._last_frame.shape[:2]
        fitted = self.fitted_size(source_width, source_height, width, height)
        rgb = self.bgr_to_rgb(self._last_frame)
        # Bilinear resampling is visually smooth for moving video and avoids
        # turning 1080p/4K frame presentation into a Tk main-thread bottleneck.
        image = Image.fromarray(rgb).resize(fitted, Image.Resampling.BILINEAR)
        self._photo = ImageTk.PhotoImage(image=image, master=self.root)
        self.video_canvas.coords(self._image_item, width // 2, height // 2)
        self.video_canvas.itemconfigure(self._image_item, image=self._photo)
        self.video_canvas.image = self._photo
        self._photo_reference_kept = self.video_canvas.image is self._photo

    def _on_video_resize(self, event: Any) -> None:
        if hasattr(self, "_placeholder_item"):
            self.video_canvas.coords(self._placeholder_item, event.width // 2, event.height // 2)
        self._cancel_after("_resize_id")
        self._resize_id = self.root.after(80, self._finish_resize)

    def _finish_resize(self) -> None:
        self._resize_id = None
        self._render_last_frame()

    def stop_session(self) -> None:
        if self.worker is not None:
            self.status_var.set("Stopping safely…")
            self.worker.stop()

    def _stop_worker(self, wait: bool) -> None:
        worker = self.worker
        self.worker = None
        if worker is not None:
            worker.stop()
            if wait:
                worker.join(timeout=5.0)

    def _save_session_result(self, result: dict[str, Any]) -> Path:
        output_dir = PROJECT_ROOT / "results" / "application_sessions"
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"{result['exercise_id']}_{result['session_id']}.json"
        output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return output

    def show_dashboard(self, result: dict[str, Any]) -> None:
        self._clear(stop_worker=True)
        self.current_screen = "dashboard"
        self._header(
            "Session Dashboard",
            f"{result['exercise_id'].replace('_', ' ').title()} — "
            f"{result['camera_view'].replace('_', ' ').title()} view",
        )
        self._render_shared_dashboard(present_shared_dashboard(result), row=1)
        return

        # Legacy family-specific dashboard code is intentionally bypassed.
        # It remains below temporarily so old result fixtures stay readable
        # while every active family uses the shared renderer above.
        summary = result["summary"]
        experimental_ai = result.get("experimental_ai")
        performance_score = (
            experimental_ai.get("performance_score")
            if isinstance(experimental_ai, dict)
            else None
        )
        all_metrics = (
            ("Total Repetitions", summary.get("total_repetitions")),
            ("Session Duration", f"{result['duration_seconds']:.1f} s"),
            ("Average Rep Duration", self._format_seconds(summary.get("average_repetition_duration"))),
            ("Last Rep Duration", self._format_seconds(summary.get("last_repetition_duration"))),
            ("Completed Cycles", summary.get("completed_cycles")),
            ("Incomplete Cycles", summary.get("incomplete_cycles")),
            ("Camera View", result.get("camera_view", strings.NOT_AVAILABLE).replace("_", " ").title()),
            ("Correct Repetitions", summary.get("correct_repetitions")),
            ("Incorrect Repetitions", summary.get("incorrect_repetitions")),
            ("Average Quality Score", summary.get("average_score")),
            ("Best Score", summary.get("best_score")),
            ("Lowest Score", summary.get("lowest_score")),
            ("Pass / Fail", None),
            ("Most Frequent Error", summary.get("most_frequent_error")),
            ("Detailed Errors", None),
        )
        if result.get("family_id") == "squat":
            metrics = all_metrics[:7] + (
                (
                    "Performance Score",
                    None
                    if performance_score is None
                    else f"{float(performance_score):.0f} / 100",
                ),
            )
        else:
            metrics = all_metrics
        self.container.grid_rowconfigure(1, weight=1)
        grid = self.ttk.Frame(self.container, style="App.TFrame")
        grid.grid(row=1, column=0, sticky="nsew")
        for column in range(4):
            grid.grid_columnconfigure(column, weight=1, uniform="metric")
        metric_rows = max(1, (len(metrics) + 3) // 4)
        for row in range(metric_rows):
            grid.grid_rowconfigure(row, weight=1, uniform="metric")
        for index, (name, value) in enumerate(metrics):
            card = self.ttk.Frame(grid, style="Metric.TFrame", padding=12)
            card.grid(row=index // 4, column=index % 4, padx=5, pady=5, sticky="nsew")
            self.ttk.Label(card, text=name, style="MetricName.TLabel").grid(
                row=0,
                column=0,
                sticky="w",
            )
            self.ttk.Label(
                card,
                text=self._display_value(value),
                style="MetricValue.TLabel",
                wraplength=230,
            ).grid(row=1, column=0, sticky="w", pady=(5, 0))
        phase_summary = summary.get("phase_activity") or {}
        phase_text = (
            ", ".join(f"{name}: {count} frames" for name, count in phase_summary.items())
            or strings.NOT_AVAILABLE
        )
        self.ttk.Label(
            self.container,
            text=f"Phase activity: {phase_text}",
            style="Body.TLabel",
            wraplength=1100,
        ).grid(row=2, column=0, sticky="w", pady=(10, 5))
        action_row = 3
        if result.get("family_id") == "squat":
            self._render_squat_ai_dashboard(
                result,
                experimental_ai if isinstance(experimental_ai, dict) else None,
                row=3,
            )
            action_row = 4
        elif result.get("family_id") in {"pushup", "pullup", "plank", "lunge"} and isinstance(experimental_ai, dict):
            self._render_family_ai_dashboard(experimental_ai, row=3)
            action_row = 4
        actions = self.ttk.Frame(self.container, style="App.TFrame")
        actions.grid(row=action_row, column=0, sticky="ew", pady=(8, 0))
        actions.grid_columnconfigure(1, weight=1)
        self.ttk.Button(
            actions,
            text="New Session",
            style="Primary.TButton",
            command=self.show_input_selection,
        ).grid(row=0, column=0)
        self.ttk.Button(
            actions,
            text="Back to Families",
            style="Secondary.TButton",
            command=self.show_exercise_selection,
        ).grid(row=0, column=2)

    def _render_shared_dashboard(self, view: dict[str, Any], *, row: int) -> None:
        """Render the same section hierarchy for every exercise family."""
        self._dashboard_ai_view = view
        self.container.grid_rowconfigure(row, weight=1)
        body = self.ttk.Frame(self.container, style="App.TFrame")
        body.grid(row=row, column=0, sticky="nsew")
        body.grid_columnconfigure(0, weight=1, uniform="dashboard")
        body.grid_columnconfigure(1, weight=1, uniform="dashboard")

        def section(parent: Any, title: str, fields: list[tuple[str, Any]], column: int) -> None:
            panel = self.ttk.Frame(parent, style="Metric.TFrame", padding=10)
            panel.grid(row=0, column=column, sticky="nsew", padx=5, pady=5)
            self.ttk.Label(panel, text=title, style="CardTitle.TLabel").grid(
                row=0, column=0, columnspan=2, sticky="w", pady=(0, 4)
            )
            for index, (name, value) in enumerate(fields, 1):
                self.ttk.Label(panel, text=name, style="MetricName.TLabel").grid(
                    row=index, column=0, sticky="w", padx=(0, 10), pady=1
                )
                self.ttk.Label(
                    panel,
                    text=self._display_value(value),
                    style="PanelText.TLabel",
                    wraplength=360,
                ).grid(row=index, column=1, sticky="w", pady=1)

        section(body, "SESSION SUMMARY", list(view["session_fields"]), 0)
        section(body, "ANALYSIS", list(view["analysis_fields"]), 1)

        body.grid_rowconfigure(1, weight=1)
        results_panel = self.ttk.Frame(body, style="Metric.TFrame", padding=10)
        results_panel.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        results_panel.grid_columnconfigure(0, weight=1)
        results_panel.grid_rowconfigure(1, weight=1)
        self.ttk.Label(results_panel, text="LATEST / PER-UNIT RESULTS", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        unit_box = self.tk.Text(
            results_panel,
            height=5,
            wrap="word",
            bg=COLORS["secondary_background"],
            fg=COLORS["main_text"],
            relief="flat",
            font=FONTS["small"],
            padx=8,
            pady=6,
        )
        scrollbar = self.ttk.Scrollbar(results_panel, orient="vertical", command=unit_box.yview)
        unit_box.configure(yscrollcommand=scrollbar.set)
        unit_box.grid(row=1, column=0, sticky="nsew", pady=(4, 0))
        scrollbar.grid(row=1, column=1, sticky="ns", pady=(4, 0))
        rows = list(view.get("rows") or [])
        unit_box.insert("1.0", "\n".join(rows) if rows else "No completed repetitions")
        unit_box.configure(state="disabled")
        self._ai_rep_text_widget = unit_box

        footer_panel = self.ttk.Frame(body, style="Metric.TFrame", padding=10)
        footer_panel.grid(row=2, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        form_fields = list(view.get("form_fields") or [])
        self.ttk.Label(footer_panel, text="FORM / ERROR", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        form_text = "    ".join(
            f"{name}: {self._display_value(value)}" for name, value in form_fields
        )
        self.ttk.Label(footer_panel, text=form_text, style="PanelText.TLabel").grid(
            row=1, column=0, sticky="w", pady=(3, 0)
        )
        models = view.get("models") or {}
        model_text = "    ".join(
            f"{str(name).replace('_', ' ').title()}: {value}" for name, value in models.items()
        ) or strings.NOT_AVAILABLE
        self.ttk.Label(footer_panel, text="MODEL STATUS", style="CardTitle.TLabel").grid(
            row=0, column=1, sticky="w", padx=(30, 0)
        )
        self.ttk.Label(
            footer_panel,
            text=model_text,
            style="MetricName.TLabel",
            wraplength=640,
        ).grid(row=1, column=1, sticky="w", padx=(30, 0), pady=(3, 0))

        actions = self.ttk.Frame(self.container, style="App.TFrame")
        actions.grid(row=row + 1, column=0, sticky="ew", pady=(8, 0))
        actions.grid_columnconfigure(1, weight=1)
        self.ttk.Button(
            actions, text="New Session", style="Primary.TButton", command=self.show_input_selection
        ).grid(row=0, column=0)
        self.ttk.Button(
            actions, text="Back to Families", style="Secondary.TButton", command=self.show_exercise_selection
        ).grid(row=0, column=2)

    def _render_squat_ai_dashboard(
        self,
        result: dict[str, Any],
        experimental_ai: dict[str, Any] | None,
        *,
        row: int,
    ) -> None:
        """Render the existing Squat AI aggregate without changing its decisions."""

        summary = result.get("summary") or {}
        view = present_squat_dashboard(
            "squat",
            summary.get("total_repetitions"),
            experimental_ai,
        )
        if view is None:
            return
        self._dashboard_ai_view = view
        self.container.grid_rowconfigure(row, weight=2)
        panel = self.ttk.Frame(self.container, style="Metric.TFrame", padding=10)
        panel.grid(row=row, column=0, sticky="nsew", pady=(5, 4))
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(3, weight=1)

        self.ttk.Label(
            panel,
            text=view["heading"],
            style="CardTitle.TLabel",
        ).grid(row=0, column=0, sticky="w")
        self.ttk.Label(
            panel,
            text=view["warning"],
            style="MetricName.TLabel",
        ).grid(row=0, column=1, sticky="e")

        detected = self._display_value(view["ai_detected_reps"])
        difference = view["difference"]
        difference_text = strings.NOT_AVAILABLE if difference is None else f"{difference:+d}"
        pass_rate = view["pass_rate"] or strings.NOT_AVAILABLE
        counts = (
            f"Official Reps: {self._display_value(view['official_count'])}    "
            f"AI Detected Reps: {detected}    Difference: {difference_text}    "
            f"Correct Reps: {view['correct_reps']}    "
            f"Incorrect Reps: {view['incorrect_reps']}    "
            f"Needs Review: {view['needs_review_reps']}    Pass Rate: {pass_rate}"
        )
        self.ttk.Label(
            panel,
            text=counts,
            style="PanelText.TLabel",
            wraplength=1100,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(5, 3))

        errors = view["error_counts"]
        if view["no_form_issues"]:
            issue_text = "FORM ISSUES: No AI form issues reported"
        else:
            issue_text = (
                "FORM ISSUES    "
                f"Bad Back: {errors['bad_back']}    "
                f"Bad Heel: {errors['bad_heel']}    "
                f"Form Issue: {errors['form_issue']}"
            )
        self.ttk.Label(
            panel,
            text=issue_text,
            style="PanelText.TLabel",
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 4))

        list_frame = self.tk.Frame(panel, bg=COLORS["card_background"])
        list_frame.grid(row=3, column=0, columnspan=2, sticky="nsew")
        list_frame.grid_rowconfigure(0, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)
        rep_text = self.tk.Text(
            list_frame,
            height=7,
            wrap="word",
            bg=COLORS["secondary_background"],
            fg=COLORS["main_text"],
            insertbackground=COLORS["cyan"],
            selectbackground=COLORS["dark_turquoise"],
            relief="flat",
            font=FONTS["small"],
            padx=8,
            pady=6,
        )
        scrollbar = self.ttk.Scrollbar(list_frame, orient="vertical", command=rep_text.yview)
        rep_text.configure(yscrollcommand=scrollbar.set)
        rep_text.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        self._ai_rep_text_widget = rep_text
        rows = view["per_rep_results"]
        if not rows:
            message = (
                strings.NOT_AVAILABLE
                if not view["available"]
                else "No completed AI repetitions were detected."
            )
            rep_text.insert("end", message)
        else:
            for rep in rows:
                line = rep["summary"]
                confidence = rep["correctness_confidence"]
                error_confidence = rep["error_confidence"]
                details = []
                if confidence:
                    details.append(f"Correctness confidence: {confidence}")
                if error_confidence:
                    details.append(f"Error confidence: {error_confidence}")
                if rep["pass_fail"] == "NEEDS_REVIEW" and rep["needs_review_reason"]:
                    details.append(str(rep["needs_review_reason"]))
                rep_text.insert("end", line + "\n")
                if details:
                    rep_text.insert("end", "    " + " | ".join(details) + "\n")
        rep_text.configure(state="disabled")

        models = view["models"]
        coverage = view["assessment_coverage"]
        coverage_text = (
            strings.NOT_AVAILABLE
            if coverage is None
            else f"{view['assessed_reps']} / {view['ai_detected_reps']} reps ({float(coverage) * 100.0:.0f}%)"
        )
        score_text = (
            f"{float(view['score']):.0f} / 100"
            if view["score"] is not None
            else strings.NOT_AVAILABLE
        )
        model_text = (
            f"Performance Score: {score_text}    Assessment Coverage: {coverage_text}\n"
            "AI MODELS    "
            f"Rep Detection: {models['rep_detection']}    "
            f"Correctness: {models['correctness']}    "
            f"Error Detection: {models['error_detection']}    "
            "Score: Experimental AI-derived performance score"
        )
        self.ttk.Label(
            panel,
            text=model_text,
            style="MetricName.TLabel",
            wraplength=1100,
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(5, 0))

    def _render_family_ai_dashboard(self, ai: dict[str, Any], *, row: int) -> None:
        """Show learned Push-up or rule-event parity without fabricating outputs."""
        family_id = self.selected_exercise.family_id if self.selected_exercise else str(ai.get("exercise") or "")
        view = present_family_dashboard(family_id, ai)
        panel = self.ttk.Frame(self.container, style="Metric.TFrame", padding=10)
        panel.grid(row=row, column=0, sticky="nsew", pady=(5, 4))
        panel.grid_columnconfigure(0, weight=1)
        title = "AI ANALYSIS — EXPERIMENTAL" if view["learned"] else ("AI DETECTION — EXPERIMENTAL" if view["detection_only"] else "RULE-BASED SESSION ASSESSMENT")
        self.ttk.Label(panel, text=title, style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")
        self.ttk.Label(panel, text=str(view["scope"]), style="MetricName.TLabel").grid(row=0, column=1, sticky="e")
        if view["learned"]:
            score = strings.NOT_AVAILABLE if view["score"] is None else f"{float(view['score']):.0f} / 100"
            coverage = strings.NOT_AVAILABLE if view["coverage"] is None else f"{100.0 * float(view['coverage']):.0f}%"
            summary = f"AI Detected: {view['ai_detected']}    PASS: {view['pass_count']}    FAIL: {view['fail_count']}    Performance Score: {score}    Coverage: {coverage}"
        elif view["detection_only"]:
            summary=f"AI Detected Reps: {view['ai_detected']}    Correctness: {strings.NOT_AVAILABLE}    Performance Score: {strings.NOT_AVAILABLE}"
        else:
            summary = str(view["reason"] or "Completed rule-based repetition events; learned assessment is not available.")
        self.ttk.Label(panel, text=summary, style="PanelText.TLabel", wraplength=1100).grid(row=1, column=0, columnspan=2, sticky="w", pady=(5, 3))
        unit_text = "\n".join(view["rows"]) if view["rows"] else "No completed repetitions"
        unit_box = self.tk.Text(panel, height=min(6, max(2, len(view["rows"]))), wrap="none", bg=COLORS["card_background"], fg=COLORS["main_text"], relief="flat", font=FONTS["normal"])
        unit_box.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(4, 4))
        unit_box.insert("1.0", unit_text)
        unit_box.configure(state="disabled")
        self._ai_rep_text_widget = unit_box
        models = view["models"]
        model_text = "    ".join(f"{key.replace('_', ' ').title()}: {value}" for key, value in models.items())
        self.ttk.Label(panel, text=model_text or strings.NOT_AVAILABLE, style="MetricName.TLabel", wraplength=1100).grid(row=3, column=0, columnspan=2, sticky="w")

    @staticmethod
    def _display_value(value: Any) -> str:
        return strings.NOT_AVAILABLE if value is None else str(value)

    @staticmethod
    def _format_seconds(value: float | None) -> str | None:
        return f"{value:.2f} s" if value is not None else None

    def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        self._cancel_after("_poll_id")
        self._cancel_after("_resize_id")
        self._cancel_after("_terminal_id")
        self._stop_worker(wait=True)
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        self.root.quit()
        self.root.destroy()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the unified AI Exercise Analysis desktop app."
    )
    parser.add_argument("--pose-model", type=Path, default=DEFAULT_POSE_MODEL)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--headless-smoke", action="store_true")
    parser.add_argument("--ui-startup-smoke", action="store_true")
    parser.add_argument("--ui-resize-smoke", action="store_true")
    parser.add_argument("--ui-video-smoke", type=Path, default=None)
    parser.add_argument("--ui-camera-smoke-seconds", type=float, default=None)
    parser.add_argument("--ui-smoke-timeout", type=float, default=90.0)
    parser.add_argument("--headless-video", type=Path, default=None)
    parser.add_argument("--exercise", choices=("pushup", "table_incline_pushup", "pullup", "squat", "plank", "lunge"), default="pushup")
    parser.add_argument("--camera-view", choices=("front", "side", "half_profile"), default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    return parser


def run_headless_smoke() -> dict[str, Any]:
    registry = ExerciseRegistry()
    router = FamilyRuntimeRouter(registry)
    runtimes = {
        exercise_id: type(router.create(exercise_id, "video")).__name__
        for exercise_id in ("pushup", "pullup", "squat", "plank", "lunge")
    }
    return {
        "application": "ok",
        "registry_exercise_count": len(registry.all()),
        "visible_families": [item.exercise_id for item in registry.main_families()],
        "runnable_families": [item.exercise_id for item in registry.main_families() if item.can_analyze],
        "runtimes": runtimes,
        "pushup_variations_share_family_expert": (
            registry.get("pushup").family_expert_class
            is registry.get("wall_pushup").family_expert_class
            is registry.get("seal_pushup").family_expert_class
        ),
    }


def run_headless_video(
    video_path: Path,
    pose_model_path: Path,
    exercise_id: str,
    camera_view: str | None = None,
    max_frames: int | None = None,
) -> dict[str, Any]:
    registry = ExerciseRegistry()
    exercise = registry.require_runnable(exercise_id, "video")
    selected_view = camera_view or exercise.recommended_camera_view
    events: queue.Queue[dict] = queue.Queue(maxsize=12)
    worker = AnalysisWorker(
        exercise=exercise,
        input_mode="video",
        camera_view=selected_view,
        source_factory=lambda: VideoFrameSource(video_path, max_frames=max_frames),
        pose_factory=lambda: PoseStreamProcessor(pose_model_path),
        events=events,
        video_path=video_path,
        preserve_video_timing=False,
    )
    worker.run_sync()
    terminal = [
        event
        for event in list(events.queue)
        if event.get("type") in {"complete", "error"}
    ]
    if not terminal:
        raise RuntimeError("Headless video worker produced no terminal event.")
    event = terminal[-1]
    if event["type"] == "error":
        raise RuntimeError(str(event.get("message")))
    return {"result": event["result"], "worker_stats": event.get("worker_stats", {})}


def _ui_diagnostics(app: ExerciseAnalysisApp) -> dict[str, Any]:
    rep_text: str | None = None
    widget = app._ai_rep_text_widget
    if widget is not None:
        try:
            rep_text = str(widget.get("1.0", "end-1c"))
        except app.tk.TclError:
            rep_text = None
    return {
        "screen": app.current_screen,
        "displayed_frame_count": len(app._displayed_frame_indices),
        "first_displayed_frame": (
            app._displayed_frame_indices[0] if app._displayed_frame_indices else None
        ),
        "last_displayed_frame": (
            app._displayed_frame_indices[-1] if app._displayed_frame_indices else None
        ),
        "unique_rendered_frames": len(app._frame_checksums),
        "photo_reference_kept": app._photo_reference_kept,
        "placeholder_hidden_after_frame": app._placeholder_hidden_after_frame,
        "countdown_history": list(app._countdown_history),
        "maximum_session_time": app._maximum_session_time,
        "maximum_progress": app._maximum_progress,
        "worker_stats": app._last_worker_stats,
        "squat_ai_dashboard": app._dashboard_ai_view,
        "squat_ai_rep_text": rep_text,
        "live_overlay": dict(app._last_live_overlay_snapshot),
        "error": app._smoke_error,
    }


def run_ui_stream_smoke(
    root: Any,
    app: ExerciseAnalysisApp,
    exercise_id: str,
    camera_view: str | None,
    *,
    video_path: Path | None = None,
    camera_seconds: float | None = None,
    timeout_seconds: float = 90.0,
) -> dict[str, Any]:
    """Exercise the real Tk event loop with an uploaded file or live camera."""

    app.automated_smoke = True
    exercise = app.registry.require_runnable(
        exercise_id,
        "video" if video_path is not None else "realtime",
    )
    app.selected_exercise = exercise
    app.show_input_selection()
    app.selected_view.set(camera_view or exercise.recommended_camera_view)
    started = time.monotonic()
    stop_requested = False
    report: dict[str, Any] = {}

    def finish(reason: str) -> None:
        if report:
            return
        report.update(_ui_diagnostics(app))
        report["finish_reason"] = reason
        report["elapsed_wall_seconds"] = time.monotonic() - started
        report["dashboard_shown"] = app.current_screen == "dashboard"
        report["result_repetitions"] = (
            app._last_result.get("summary", {}).get("total_repetitions")
            if app._last_result is not None
            else None
        )
        app.close()

    def monitor() -> None:
        nonlocal stop_requested
        elapsed = time.monotonic() - started
        if app._smoke_error is not None:
            finish("error")
            return
        if app.current_screen == "dashboard":
            finish("dashboard")
            return
        if camera_seconds is not None and elapsed >= camera_seconds and not stop_requested:
            stop_requested = True
            app.stop_session()
        if elapsed >= timeout_seconds:
            finish("timeout")
            return
        root.after(50, monitor)

    if video_path is not None:
        app.start_analysis("video", video_path)
    else:
        app.start_analysis("realtime", None)
    root.after(50, monitor)
    root.mainloop()
    return report


def main() -> None:
    args = build_parser().parse_args()
    if args.headless_smoke:
        print(json.dumps(run_headless_smoke(), indent=2))
        return
    if args.headless_video is not None:
        result = run_headless_video(
            args.headless_video,
            args.pose_model,
            args.exercise,
            args.camera_view,
            args.max_frames,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    import tkinter as tk

    root = tk.Tk()
    app = ExerciseAnalysisApp(
        root,
        pose_model_path=args.pose_model,
        camera_index=args.camera_index,
    )
    if args.ui_video_smoke is not None or args.ui_camera_smoke_seconds is not None:
        report = run_ui_stream_smoke(
            root,
            app,
            args.exercise,
            args.camera_view,
            video_path=args.ui_video_smoke,
            camera_seconds=args.ui_camera_smoke_seconds,
            timeout_seconds=args.ui_smoke_timeout,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return
    if args.ui_startup_smoke or args.ui_resize_smoke:
        root.update_idletasks()
        report: dict[str, Any] = {
            "ui_startup": "ok",
            "visible_family_count": len(app.registry.main_families()),
            "resizable": tuple(bool(value) for value in root.resizable()),
            "minimum_size": tuple(root.minsize()),
        }
        if args.ui_resize_smoke:
            sizes = []
            for geometry in ("900x600", "1280x800", "1500x900"):
                root.geometry(geometry)
                root.update_idletasks()
                sizes.append((root.winfo_width(), root.winfo_height()))
            maximize_ok = False
            restore_ok = False
            try:
                root.state("zoomed")
                root.update_idletasks()
                maximize_ok = root.state() == "zoomed"
                root.state("normal")
                root.update_idletasks()
                restore_ok = root.state() == "normal"
            except tk.TclError:
                pass
            report.update(
                {
                    "resize_sizes": sizes,
                    "maximize_ok": maximize_ok,
                    "restore_ok": restore_ok,
                }
            )
        app.close()
        print(json.dumps(report, indent=2))
        return
    root.mainloop()


if __name__ == "__main__":
    main()
