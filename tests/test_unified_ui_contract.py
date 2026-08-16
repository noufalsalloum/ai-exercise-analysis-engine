from __future__ import annotations

import unittest

import numpy as np

from application.exercise_registry import ExerciseRegistry
from ui.app import ExerciseAnalysisApp
from ui.theme import COLORS, FONTS


class UnifiedUIContractTests(unittest.TestCase):
    def test_opencv_bgr_is_explicitly_converted_to_rgb(self) -> None:
        bgr = np.asarray([[[0, 0, 255], [0, 255, 0], [255, 0, 0]]], dtype=np.uint8)
        rgb = ExerciseAnalysisApp.bgr_to_rgb(bgr)
        np.testing.assert_array_equal(
            rgb,
            np.asarray([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8),
        )

    def test_aspect_ratio_is_preserved_with_letterboxing(self) -> None:
        self.assertEqual(ExerciseAnalysisApp.fitted_size(1920, 1080, 1000, 600), (1000, 562))
        width, height = ExerciseAnalysisApp.fitted_size(720, 1280, 1000, 600)
        self.assertLessEqual(width, 1000)
        self.assertLessEqual(height, 600)
        self.assertAlmostEqual(width / height, 720 / 1280, places=2)

    def test_main_screen_contract_and_theme_are_centralized(self) -> None:
        registry = ExerciseRegistry()
        self.assertEqual(
            [item.exercise_id for item in registry.main_families()],
            ["pushup", "pullup", "squat", "lunge", "plank"],
        )
        self.assertEqual(COLORS["main_background"], "#020817")
        self.assertEqual(COLORS["cyan"], "#16D9E8")
        self.assertEqual(COLORS["gold"], "#D8B36A")
        self.assertLessEqual(FONTS["overlay"][1], 13)
        self.assertLessEqual(FONTS["button"][1], 14)

    def test_tk_canvas_keeps_photo_reference_and_hides_placeholder(self) -> None:
        try:
            import tkinter as tk

            root = tk.Tk()
        except Exception as exc:  # pragma: no cover - platform display availability
            self.skipTest(f"Tk display unavailable: {exc}")
        app: ExerciseAnalysisApp | None = None
        try:
            root.withdraw()
            app = ExerciseAnalysisApp(root)
            app._clear(stop_worker=True)
            app.video_canvas = tk.Canvas(app.container, width=400, height=240)
            app.video_canvas.grid(row=0, column=0, sticky="nsew")
            app._image_item = app.video_canvas.create_image(0, 0, anchor="center")
            app._placeholder_item = app.video_canvas.create_text(10, 10, text="placeholder")
            root.update_idletasks()
            frame = np.zeros((120, 200, 3), dtype=np.uint8)
            frame[:, :, 2] = 255
            app._display_frame(frame)
            root.update_idletasks()
            self.assertIsNotNone(app._photo)
            self.assertIs(app.video_canvas.image, app._photo)
            self.assertNotEqual(app.video_canvas.itemcget(app._image_item, "image"), "")
            self.assertEqual(app.video_canvas.itemcget(app._placeholder_item, "state"), "hidden")
            self.assertTrue(app._photo_reference_kept)
            self.assertTrue(app._placeholder_hidden_after_frame)
            self.assertEqual(tuple(bool(value) for value in root.resizable()), (True, True))
        finally:
            if app is not None:
                app.close()
            else:
                root.destroy()


if __name__ == "__main__":
    unittest.main()
