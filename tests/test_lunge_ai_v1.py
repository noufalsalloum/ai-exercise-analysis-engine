from __future__ import annotations

import json, unittest
from pathlib import Path

import torch

from application.exercise_registry import ExerciseRegistry
from application.runtime_router import FamilyRuntimeRouter
from datasets.adapters.rehab24_lunge import Rehab24LungeAdapter
from datasets.adapters.rehab24_lunge_split import balanced_lunge_subject_split
from inference.lunge_ai_mvp import LungeBoundaryExperimentalOrchestrator
from inference.lunge_runtime import LungeRepConfig, LungeRepetitionRuntime
from models.lunge_correctness import LungeCorrectnessModel
from models.lunge_rep_boundary import LungeRepBoundaryModel
from ui.family_parity_presentation import present_family_dashboard, present_live_family_status

ROOT=Path(__file__).resolve().parents[1]


class LungeAIV1Tests(unittest.TestCase):
    def test_ex5_adapter_and_subject_split_are_leak_free(self):
        samples=Rehab24LungeAdapter(ROOT/"datasets/external").samples(); self.assertEqual(len(samples),348)
        assignment,evidence=balanced_lunge_subject_split(samples)
        groups={name:set(value["subjects"]) for name,value in evidence["splits"].items()}
        self.assertFalse(groups["train"]&groups["validation"]); self.assertFalse(groups["train"]&groups["test"]); self.assertFalse(groups["validation"]&groups["test"])
        self.assertEqual(sum(value["repetitions"] for value in evidence["splits"].values()),174)

    def test_rule_runtime_counts_full_cycle_once_and_reset_clears(self):
        runtime=LungeRepetitionRuntime(LungeRepConfig())
        # Each state persists for the configured five-frame median window;
        # one-frame spikes are intentionally not valid phase transitions.
        sequence=[]
        for signal in ((170,170,.5),(140,145,.52),(110,120,.58),(140,150,.54),(170,170,.51)):
            sequence.extend([signal]*5)
        completed=[]
        for index,(knee,hip,pelvis) in enumerate(sequence):
            result=runtime.update_signals(knee,hip,pelvis,.9,index*.1,index,True)
            if result.completed_cycle: completed.append(result.completed_cycle)
        self.assertEqual(runtime.repetition_count,1); self.assertEqual(len(completed),1)
        runtime.reset(); self.assertEqual(runtime.repetition_count,0); self.assertEqual(runtime.phase,"READY")

    def test_countdown_does_not_count(self):
        runtime=LungeRepetitionRuntime()
        for index,knee in enumerate((170,140,110,140,170)):
            runtime.update_signals(knee,150,.5,.9,index*.2,index,False)
        self.assertEqual(runtime.repetition_count,0)

    def test_models_strict_load_and_motionbert_frozen(self):
        boundary=torch.load(ROOT/"checkpoints/lunge_ai_v1/boundary/best.pt",map_location="cpu",weights_only=True)
        model=LungeRepBoundaryModel(); model.load_state_dict(boundary["model_state_dict"],strict=True)
        self.assertEqual(boundary["exercise_scope"],"REHAB24 Ex5 Lunge")
        correctness=torch.load(ROOT/"archive/checkpoints/lunge_ai_v1/correctness/final_dev.pt",map_location="cpu",weights_only=True)
        tail=LungeCorrectnessModel(ROOT/"models/latest_epoch.bin")
        tail.expert.load_state_dict(correctness["expert_state_dict"],strict=True); tail.correctness_head.load_state_dict(correctness["correctness_head_state_dict"],strict=True)
        self.assertTrue(all(not value.requires_grad for value in tail.backbone.parameters()))
        self.assertFalse(correctness["product_activated"])

    def test_product_enables_boundary_but_not_failed_correctness(self):
        exercise=ExerciseRegistry().get("lunge"); self.assertTrue(exercise.can_analyze); self.assertIsNone(exercise.assessment_checkpoint)
        self.assertEqual(type(FamilyRuntimeRouter().create("lunge","video")).__name__,"LungeRepetitionRuntime")
        orchestrator=LungeBoundaryExperimentalOrchestrator.from_default_config()
        try: status=orchestrator.live_status()
        finally: orchestrator.close()
        self.assertIn("Boundary V1",status["model_status"]["detection"]); self.assertIn("Not Enabled",status["model_status"]["correctness"]); self.assertIsNone(status["performance_score"])

    def test_lunge_presentation_does_not_fabricate_pass_fail_or_score(self):
        status={"available":True,"ai_detected_reps":1,"last_ai_rep":{"rep_index":1},"per_rep_results":[{"rep_index":1,"start_frame":10,"end_frame":80}],"model_status":{},"scope":"Ex5","performance_score":None}
        self.assertEqual(present_live_family_status("lunge",status)["last"], "Rep 1 Completed")
        view=present_family_dashboard("lunge",status); self.assertTrue(view["detection_only"]); self.assertIsNone(view["score"]); self.assertIn("Assessment Not Available",view["rows"][0])


if __name__ == "__main__": unittest.main()
