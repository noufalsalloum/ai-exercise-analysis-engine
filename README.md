# AI Exercise Analysis Engine

Desktop application and research code for pose-based exercise analysis. The project keeps runtime code, training, evaluation, tools, results, checkpoints, and historical artifacts in separate top-level areas.

## Current Architecture

```text
Video / Camera
  -> MediaPipe Pose
  -> preprocessing v4 (MP33 -> H36M17)
  -> rule-based family runtime
  -> optional learned family models
  -> session aggregation
  -> unified Tkinter UI
```

The reorganization changes module locations only. Model logic, thresholds, preprocessing behavior, UI behavior, dataset content, and checkpoint bytes are unchanged.

## Active Exercise Families

- **Squat:** rule-based official count plus experimental Boundary V2, Correctness V3, and Error V1.
- **Push-up:** rule-based floor/table variants; learned Boundary V1 and Correctness V1 are limited to Table/Incline Push-up.
- **Pull-up:** rule-based runtime; learned correctness is not available.
- **Marching Plank:** rule-based valid-hold timing; learned correctness is not available.
- **Lunge:** rule-based runtime plus development Boundary V1; learned correctness is not active.

## Project Structure

```text
ai_engine/
|-- application/       # contracts, registry, routing, sessions, workers
|-- backbone/          # MotionBERT/DSTformer source
|-- checkpoints/       # active checkpoints only
|-- configs/           # active exercise and AI configuration
|-- datasets/          # existing dataset junction and dataset utilities
|-- evaluation/        # evaluation entry points
|-- experts/           # exercise-specific expert modules
|-- heads/             # shared output heads
|-- inference/         # family runtimes and AI orchestration
|-- input_sources/     # camera, video, and pose streams
|-- models/            # active model definitions and MotionBERT weights
|-- preprocessing/     # preprocessing v4 and pose utilities
|-- prototypes/        # prototype representation support
|-- results/           # current model evidence and new session outputs
|-- splits/            # fixed development/test split manifests
|-- tests/             # unit and integration tests
|-- tools/             # audits, diagnostics, preparation, visualization
|-- training/          # reusable trainers, CLI modules, launch scripts
|-- ui/                # existing desktop UI and presentation adapters
|-- archive/           # historical checkpoints, reports, logs, entry points
|-- run_application.py # only executable Python entry point in the root
|-- README.md
`-- requirements.txt
```

Original dataset files are not moved or modified by project commands. Historical material in `archive/` is preserved for reproducibility and is not imported by the application runtime.

## Running the Application

From the project root:

```bat
C:\Users\JoudA\AppData\Local\Programs\Python\Python313\python.exe run_application.py
```

## Running Tests

```bat
C:\Users\JoudA\AppData\Local\Programs\Python\Python313\python.exe -m unittest discover -s tests -p "test_*.py"
```

## Training

Training never starts from the application. Current entry points are under `training/scripts/`:

```bat
python -m training.scripts.train_exercise_representation --help
python -m training.scripts.train_intellirehab_correctness --help
python -m training.scripts.run_squat_rep_boundary_v2 --help
python -m training.scripts.run_squat_correctness_v3 --help
python -m training.scripts.train_squat_error_v1 --help
```

The files `training/squat_rep_boundary.py`, `training/squat_correctness.py`, and `training/squat_correctness_v2.py` remain active dependencies of later model versions; their names do not mean they are unused.

## Evaluation

```bat
python -m evaluation.evaluate_squat_external --help
python -m evaluation.evaluate_squat_end_to_end --help
python -m evaluation.evaluate_squat_v2_locked_test --help
```

Reusable preparation and diagnostic commands live under `tools/`. Dataset-specific utilities remain under `datasets/tools/`.

## Model Status

All learned outputs are development or experimental outputs. The application does not claim production or clinical validation. Missing model inference fails open to the existing rule-based runtime.
