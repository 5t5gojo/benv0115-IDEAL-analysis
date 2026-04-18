from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "script"
OUTPUT_DIR = SCRIPT_DIR / "outputs"

STAGE1_DIR = OUTPUT_DIR / "stage1_preparation"
STAGE2_DIR = OUTPUT_DIR / "stage2_clustering"
STAGE3_DIR = OUTPUT_DIR / "stage3_descriptive"
LEGACY_DIR = OUTPUT_DIR / "legacy"


def ensure_output_dirs() -> None:
    for path in [OUTPUT_DIR, STAGE1_DIR, STAGE2_DIR, STAGE3_DIR, LEGACY_DIR]:
        path.mkdir(parents=True, exist_ok=True)
