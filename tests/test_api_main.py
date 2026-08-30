"""Endpoint-level test for the duration guard in api/main.py's /analyze:
an over-long video must be rejected with a clean 400 before run_analysis()
(the expensive part) is ever called. See api/config.py's
MAX_VIDEO_DURATION_SECONDS comment for how the cap was derived.

Sets API_SHARED_SECRET before importing api.main, since api/config.py
fails fast at import time without it (by design — see that module).
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

os.environ.setdefault("API_SHARED_SECRET", "test-secret-for-unit-tests")

from fastapi.testclient import TestClient  # noqa: E402

import api.main as api_main  # noqa: E402
from api.config import PROJECT_ROOT  # noqa: E402


class AnalyzeDurationGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(api_main.app)
        self.headers = {"X-API-Key": "test-secret-for-unit-tests"}
        self.form = {"exercise_id": "squat", "video_url": "https://example.com/video.mp4"}

    def test_over_duration_video_is_rejected_before_run_analysis_is_called(self) -> None:
        fake_path = Path("fake-downloaded-video.mp4")
        with patch.object(api_main, "_download_video", new=AsyncMock(return_value=fake_path)), patch.object(
            api_main.config, "POSE_MODEL_PATH"
        ) as mock_pose_path, patch(
            "api.main.probe_video_duration_seconds", return_value=api_main.config.MAX_VIDEO_DURATION_SECONDS + 5.0
        ), patch(
            "api.main.run_analysis"
        ) as mock_run_analysis, patch.object(
            Path, "unlink"
        ):
            mock_pose_path.is_file.return_value = True
            response = self.client.post("/analyze", data=self.form, headers=self.headers)

        self.assertEqual(response.status_code, 400)
        body = response.json()
        self.assertEqual(body["error"]["code"], "video_invalid")
        mock_run_analysis.assert_not_called()

    def test_under_duration_video_proceeds_to_run_analysis(self) -> None:
        fake_path = Path("fake-downloaded-video.mp4")
        fake_result = {
            "session_id": "s1",
            "exercise_id": "squat",
            "family_id": "squat",
            "summary": {"total_repetitions": 4},
        }
        with patch.object(api_main, "_download_video", new=AsyncMock(return_value=fake_path)), patch.object(
            api_main.config, "POSE_MODEL_PATH"
        ) as mock_pose_path, patch(
            "api.main.probe_video_duration_seconds", return_value=api_main.config.MAX_VIDEO_DURATION_SECONDS - 5.0
        ), patch(
            "api.main.run_analysis", return_value={"result": fake_result, "pose_coverage_rate": 1.0}
        ) as mock_run_analysis, patch.object(
            Path, "unlink"
        ):
            mock_pose_path.is_file.return_value = True
            response = self.client.post("/analyze", data=self.form, headers=self.headers)

        self.assertEqual(response.status_code, 200)
        mock_run_analysis.assert_called_once()


class PingSquatRuntimeDiagnosticTests(unittest.TestCase):
    """Guards against the exact class of bug found 2026-08-31: a fix to
    inference/squat_runtime.py's SquatRepConfig *dataclass default* silently
    had zero effect in production, because
    application/runtime_router.py's FamilyRuntimeRouter.create() - the code
    path api/pipeline.py's AnalysisWorker actually uses for every /analyze
    video - loads configs/squat.json fresh on every call and overrides every
    field via SquatRepConfig.from_dict(...). /ping's squat_runtime block
    must report the value a real request would actually use (i.e. what's in
    configs/squat.json), not the dataclass default, or this exact failure
    mode becomes invisible again."""

    def setUp(self) -> None:
        self.client = TestClient(api_main.app)

    def test_ping_reports_the_real_configs_json_value_not_the_dataclass_default(self) -> None:
        squat_config_path = PROJECT_ROOT / "configs" / "squat.json"
        expected = json.loads(squat_config_path.read_text(encoding="utf-8"))["runtime"]["repetition_counter"]

        response = self.client.get("/ping")

        self.assertIn(response.status_code, (200, 503))
        body = response.json()
        squat_runtime = body["squat_runtime"]
        self.assertNotIn("error", squat_runtime, squat_runtime.get("error"))
        self.assertEqual(squat_runtime["minimum_pelvis_displacement"], expected["minimum_pelvis_displacement"])
        self.assertEqual(squat_runtime["return_pelvis_tolerance"], expected["return_pelvis_tolerance"])
        self.assertTrue(squat_runtime["squat_runtime_module_path"].replace("\\", "/").endswith(
            "inference/squat_runtime.py"
        ))

    def test_health_and_ping_report_the_identical_squat_runtime_diagnostic(self) -> None:
        # /health and /ping share _health_payload() - pin that down so they
        # can never silently diverge.
        self.assertEqual(
            self.client.get("/health").json()["squat_runtime"],
            self.client.get("/ping").json()["squat_runtime"],
        )


if __name__ == "__main__":
    unittest.main()
