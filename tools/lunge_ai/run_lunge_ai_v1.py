from pathlib import Path
import json, sys, torch

PROJECT_ROOT=Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
from training.lunge_ai_v1 import run

if __name__ == "__main__":
    print(json.dumps(run(PROJECT_ROOT,torch.device("cuda" if torch.cuda.is_available() else "cpu")),indent=2))
