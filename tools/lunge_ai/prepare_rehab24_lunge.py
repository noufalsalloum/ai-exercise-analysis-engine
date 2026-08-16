"""Build preprocessing-v4 caches for subject-safe REHAB24 Ex5 Lunge AI V1."""

from __future__ import annotations

import argparse, json, sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from datasets.adapters.rehab24_lunge import Rehab24LungeAdapter
from datasets.adapters.rehab24_lunge_split import balanced_lunge_subject_split
from preprocessing.h36m_coordinate_normalizer import H36MCoordinateNormalizer
from preprocessing.landmark_selector import LandmarkSelector
from tools.squat_ai.prepare_rehab24_squat import extract_video, fill_missing_pose, resample_sequence, sha256

CACHE_VERSION = "rehab24_ex5_lunge_mp33_h36m_v4_rep60_v1"


def _full_cache_path(full_dir: Path, path: Path) -> Path:
    stem=path.stem.replace("-30fps-transposed","_cam18").replace("-30fps","_cam17")
    return full_dir/f"{stem}.npz"


def _extract_full_video(task: tuple[str, str, str]) -> tuple[str, int, int]:
    path_text, output_text, pose_text = task
    path=Path(path_text); output=Path(output_text)
    raw,detected,fps=extract_video(path,Path(pose_text),None); filled,gaps=fill_missing_pose(raw,detected)
    selector=LandmarkSelector({"landmarks":{"selected_landmarks":[]}}); normalizer=H36MCoordinateNormalizer()
    full_h36m=selector.to_h36m_17(filled); full_motion,_=normalizer.normalize(full_h36m)
    np.savez_compressed(output,landmarks=filled,detected_mask=detected,motionbert_input=full_motion,fps=np.asarray(fps),cache_version=np.asarray(CACHE_VERSION),gap_diagnostics_json=np.asarray(json.dumps(gaps,sort_keys=True)))
    return path.name,len(filled),int(detected.sum())


def build_cache(project: Path, external: Path, cache_dir: Path, pose_model: Path, seed: int = 42):
    full_dir = cache_dir / "full_videos"; rep_dir = cache_dir / "repetitions"
    full_dir.mkdir(parents=True, exist_ok=True); rep_dir.mkdir(parents=True, exist_ok=True)
    adapter = Rehab24LungeAdapter(external); samples = adapter.samples()
    assignment, evidence = balanced_lunge_subject_split(samples, seed)
    result_dir = project / "results/full_exercise_ai_parity/lunge/data"; result_dir.mkdir(parents=True, exist_ok=True)
    manifest = result_dir / "repetition_manifest.csv"; adapter.write_manifest(manifest, assignment)
    split_dir = project / "splits"; split_dir.mkdir(parents=True, exist_ok=True)
    for split, filename in (("train","rehab24_lunge_train_subjects.json"),("validation","rehab24_lunge_val_subjects.json"),("test","rehab24_lunge_test_subjects.json")):
        subjects = sorted([s for s,v in assignment.items() if v == split], key=int)
        adapter.write_subject_file(split_dir / filename, split, subjects, {"seed":seed,"method":evidence["method"],"statistics":evidence["splits"][split],"assignment_sha256":evidence["assignment_sha256"]})
    by_path = defaultdict(list)
    for sample in samples: by_path[sample.video_path].append(sample)
    tasks=[]
    for path_text in sorted(by_path):
        path=Path(path_text); output=_full_cache_path(full_dir,path)
        if not output.is_file(): tasks.append((str(path),str(output),str(pose_model)))
    if tasks:
        with ProcessPoolExecutor(max_workers=min(4,len(tasks))) as pool:
            futures=[pool.submit(_extract_full_video,task) for task in tasks]
            for index,future in enumerate(as_completed(futures),1):
                name,frames,detected=future.result(); print(f"extracted {index}/{len(tasks)} {name} frames={frames} detected={detected}",flush=True)
    selector = LandmarkSelector({"landmarks":{"selected_landmarks":[]}}); normalizer = H36MCoordinateNormalizer()
    video_rows=[]; cached=0
    for number, path_text in enumerate(sorted(by_path), 1):
        path=Path(path_text); full_path=_full_cache_path(full_dir,path)
        if full_path.is_file():
            with np.load(full_path,allow_pickle=False) as archive:
                filled=np.asarray(archive["landmarks"],np.float32); detected=np.asarray(archive["detected_mask"],bool); fps=float(archive["fps"])
        else:
            raw,detected,fps=extract_video(path,pose_model,None); filled,gaps=fill_missing_pose(raw,detected)
            full_h36m=selector.to_h36m_17(filled); full_motion,_=normalizer.normalize(full_h36m)
            np.savez_compressed(full_path,landmarks=filled,detected_mask=detected,motionbert_input=full_motion,fps=np.asarray(fps),cache_version=np.asarray(CACHE_VERSION),gap_diagnostics_json=np.asarray(json.dumps(gaps,sort_keys=True)))
        video_rows.append({"video":path.name,"frames":len(filled),"detected":int(detected.sum()),"fps":fps})
        for sample in by_path[path_text]:
            segment=filled[sample.start_frame-1:sample.end_frame]; h36m=selector.to_h36m_17(segment); normalized,norm=normalizer.normalize(h36m); values=resample_sequence(normalized)
            values[...,2]=np.clip(values[...,2],0,1); mask=values[...,2].mean(axis=1)>0.01
            if not mask.any() or not np.isfinite(values).all(): raise ValueError(sample.sample_id)
            metadata={**sample.to_dict(),"split":assignment[sample.subject_id],"cache_version":CACHE_VERSION,"preprocessing_version":adapter.PREPROCESSING_VERSION,"source_frames":len(segment),"sequence_scale":norm.sequence_scale}
            np.savez_compressed(rep_dir/f"{sample.sample_id}.npz",motionbert_input=values,temporal_mask=mask,correctness=np.asarray(sample.correctness,np.int64),metadata_json=np.asarray(json.dumps(metadata,sort_keys=True))); cached+=1
        print(f"cached {number}/18 {path.name} frames={len(filled)} detected={int(detected.sum())}",flush=True)
    if cached != 348: raise ValueError(f"Expected 348 caches, got {cached}")
    summary={"cache_version":CACHE_VERSION,"samples":cached,"full_videos":len(video_rows),"manifest":str(manifest.resolve()),"manifest_sha256":sha256(manifest),"split_evidence":evidence,"videos":video_rows}
    (result_dir/"preprocessing_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8"); (cache_dir/"cache_metadata.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    return summary


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--pose-model",type=Path,default=Path(r"C:\MediaPipe\pose_landmarker_full.task")); parser.add_argument("--seed",type=int,default=42); args=parser.parse_args()
    print(json.dumps(build_cache(PROJECT_ROOT,PROJECT_ROOT/"datasets/external",PROJECT_ROOT/"datasets/window_cache/rehab24_lunge_v1",args.pose_model,args.seed),indent=2))


if __name__ == "__main__": main()
