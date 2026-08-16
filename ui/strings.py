"""English MVP strings kept separate from application business logic."""

APP_TITLE = "AI Exercise Analysis"
APP_SUBTITLE = "Choose an exercise family to view its validated capabilities"
STATUS_LABELS = {
    "ready": "Ready",
    "partial": "Partially Available",
    "development": "In Development",
}
STATUS_COLORS = {
    "ready": "#1f9d68",
    "partial": "#d68a19",
    "development": "#667085",
}
from application.presentation_contract import (
    COMPLETED,
    HOLDING,
    NOT_AVAILABLE,
    PROCESSING,
    READY,
    WAITING_REPETITION,
)

IN_DEVELOPMENT = "In development"
PREPARATION_GUIDANCE = "Position your full body in frame. Repetitions start after the countdown."
