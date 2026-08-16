from __future__ import annotations

import unittest

from datasets.tools.build_external_caches import stratified_group_split, subject_split
from datasets.adapters.base_adapter import UnifiedSample


def sample(group: str, label: str, subject: str | None = None) -> UnifiedSample:
    return UnifiedSample(
        sample_id=group,
        source_dataset="synthetic",
        exercise_id=label,
        subject_id=subject,
        video_id=group,
        sequence_id=group,
        rep_id=None,
        input_type="sequence",
        input_path="unused",
        frame_start=None,
        frame_end=None,
        exercise_label=label,
        correctness_label=None,
        split_group=group,
    )


class ExternalSplitTests(unittest.TestCase):
    def test_video_groups_never_cross_splits(self) -> None:
        samples = [sample(f"{label}_{index}", label) for label in ("a", "b") for index in range(10)]
        assignment = stratified_group_split(samples, seed=42)
        self.assertEqual(set(assignment), {item.split_group for item in samples})
        self.assertEqual(set(assignment.values()), {"train", "validation", "test"})
        self.assertEqual(assignment, stratified_group_split(samples, seed=42))

    def test_subject_split_assigns_every_sequence_from_subject_together(self) -> None:
        samples = [
            sample(f"sequence_{subject}_{rep}", "gesture", subject=f"s{subject}")
            for subject in range(10) for rep in range(3)
        ]
        assignment = subject_split(samples, seed=42)
        self.assertEqual(len(assignment), 10)
        self.assertEqual(set(assignment.values()), {"train", "validation", "test"})
        for item in samples:
            self.assertIn(item.subject_id, assignment)


if __name__ == "__main__":
    unittest.main()
