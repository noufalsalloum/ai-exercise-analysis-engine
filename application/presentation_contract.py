"""Small shared vocabulary for live and dashboard presentation."""

NOT_AVAILABLE = "Not Available"
READY = "Ready"
PROCESSING = "Processing"
WAITING_REPETITION = "Waiting for Repetition"
COMPLETED = "Completed"
HOLDING = "Holding"

ALLOWED_PROCESSING_STATES = frozenset(
    {READY, PROCESSING, WAITING_REPETITION, COMPLETED, NOT_AVAILABLE, HOLDING}
)


def completed_unit(index: object, *, plank: bool = False) -> str:
    label = "Unit" if plank else "Rep"
    return f"{label} {index} Completed"
