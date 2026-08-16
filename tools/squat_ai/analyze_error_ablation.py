"""Development-only Error V1 shortcut and aggregation ablations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "results" / "squat_ai_v4" / "error_domain"


def main() -> None:
    data = ROOT / "results" / "squat_error_v1" / "data"
    manifest = pd.read_csv(data / "development_split_manifest.csv")
    features = np.load(data / "pose_features.npy", mmap_mode="r")
    names = json.loads((data / "feature_names.json").read_text(encoding="utf-8"))["feature_names"]
    train_rows = manifest[(manifest["source_split"] == "train") & (manifest["development_split"] == "train")]
    validation_rows = manifest[(manifest["source_split"] == "train") & (manifest["development_split"] == "validation")]
    train_indices = train_rows["feature_index"].to_numpy(np.int64)
    validation_indices = validation_rows["feature_index"].to_numpy(np.int64)
    y_train = train_rows["label_index"].to_numpy(np.int64)
    y_validation = validation_rows["label_index"].to_numpy(np.int64)
    confidence = [index for index, name in enumerate(names) if "confidence" in name]
    pose_success = names.index("pose_success")
    variants = {
        "V1-A_current_64": list(range(len(names))),
        "V1-B_without_pose_success": [index for index in range(len(names)) if index != pose_success],
        "V1-C_without_pose_success_or_confidence": [
            index for index in range(len(names))
            if index != pose_success and index not in confidence
        ],
    }
    rows = []
    for name, columns in variants.items():
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
            ]
        )
        model.fit(np.asarray(features[train_indices])[:, columns], y_train)
        prediction = model.predict(np.asarray(features[validation_indices])[:, columns])
        recalls = recall_score(y_validation, prediction, labels=[0, 1, 2], average=None, zero_division=0)
        rows.append(
            {
                "variant": name,
                "feature_count": len(columns),
                "validation_macro_f1": f1_score(y_validation, prediction, average="macro", zero_division=0),
                "validation_balanced_accuracy": balanced_accuracy_score(y_validation, prediction),
                "good_recall": recalls[0],
                "bad_back_recall": recalls[1],
                "bad_heel_recall": recalls[2],
                "official_test_used": False,
                "candidate_only": True,
            }
        )
    RESULT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULT / "error_feature_ablation.csv", index=False)

    shortcut = (
        manifest[manifest["source_split"] == "train"]
        .groupby(["canonical_label", "pose_success"])
        .size()
        .reset_index(name="samples")
    )
    shortcut.to_csv(RESULT / "pose_success_by_class.csv", index=False)

    trace = json.loads(
        (ROOT / "results" / "squat_ai_v4" / "diagnostics" / "side_trace.json").read_text(encoding="utf-8")
    )
    aggregation_rows = []
    classes = ("good", "bad_back", "bad_heel")
    for rep in trace["segments"]:
        probabilities = np.asarray(rep["error_frame_probabilities"], dtype=np.float64)
        strategies = {
            "mean_probability": probabilities.mean(axis=0),
            "median_probability": np.median(probabilities, axis=0),
            "majority_vote": np.bincount(probabilities.argmax(axis=1), minlength=3) / len(probabilities),
            "confidence_weighted_mean": np.asarray(rep["error_confidence_weighted_probabilities"]),
        }
        for strategy, values in strategies.items():
            aggregation_rows.append(
                {
                    "rep_index": rep["rep_index"],
                    "strategy": strategy,
                    "predicted_class": classes[int(values.argmax())],
                    "confidence": float(values.max()),
                    "temporal_error_ground_truth": False,
                }
            )
    pd.DataFrame(aggregation_rows).to_csv(RESULT / "aggregation_stability.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
