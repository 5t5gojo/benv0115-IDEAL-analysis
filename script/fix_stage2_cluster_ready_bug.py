from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from output_paths import STAGE1_DIR, STAGE2_DIR, ensure_output_dirs


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "script"

DAILY_PATH = STAGE1_DIR / "daily_electricity_features_enhanced.csv"
ARCH_PATH = STAGE2_DIR / "daily_archetypes_enhanced.csv"
HOUSEHOLD_PATH = STAGE2_DIR / "household_archetypes_enhanced.csv"

STAGE1_NOTEBOOK = SCRIPT_DIR / "0145(3).ipynb"
STAGE2_NOTEBOOK = SCRIPT_DIR / "0145_stage2_clustering.ipynb"

NORM_COLS = [f"norm_block_{hour:02d}" for hour in range(0, 24, 2)]


def recompute_cluster_ready(df: pd.DataFrame) -> pd.Series:
    all_norm_present = df[NORM_COLS].notna().all(axis=1)
    return (
        df["complete_day"].fillna(False).astype(bool)
        & df["normalisable_day"].fillna(False).astype(bool)
        & (~df["flat_day_flag"].fillna(False).astype(bool))
        & all_norm_present
    )


def fix_daily_features() -> pd.DataFrame:
    df = pd.read_csv(DAILY_PATH, low_memory=False)

    old_flag = df["cluster_ready_day"].fillna(False).astype(bool)
    new_flag = recompute_cluster_ready(df)

    print("[1/4] Fixing daily_electricity_features_enhanced.csv")
    print(f"old cluster_ready_day=True : {int(old_flag.sum())}")
    print(f"new cluster_ready_day=True : {int(new_flag.sum())}")
    print(f"flipped True -> False      : {int((old_flag & ~new_flag).sum())}")
    print(f"flipped False -> True      : {int((~old_flag & new_flag).sum())}")

    df["cluster_ready_day"] = new_flag
    df.to_csv(DAILY_PATH, index=False)
    print(f"saved: {DAILY_PATH}")
    return df


def fix_daily_archetypes(daily_df: pd.DataFrame) -> None:
    arch = pd.read_csv(ARCH_PATH, low_memory=False)

    print("\n[2/4] Syncing daily_archetypes_enhanced.csv")
    flag_lookup = daily_df.set_index(["homeid", "date"])["cluster_ready_day"]
    arch["cluster_ready_day"] = (
        arch.set_index(["homeid", "date"]).index.map(flag_lookup).fillna(False).astype(bool)
    )

    if not arch["cluster_ready_day"].all():
        raise ValueError("Some archetype rows still have cluster_ready_day=False after sync")

    print(f"rows with archetype label  : {len(arch)}")
    print(f"rows with ready=True       : {int(daily_df['cluster_ready_day'].sum())}")
    arch.to_csv(ARCH_PATH, index=False)
    print(f"saved: {ARCH_PATH}")


def sanity_check_households() -> None:
    household = pd.read_csv(HOUSEHOLD_PATH, low_memory=False)
    print("\n[3/4] Household file sanity check")
    print(f"household rows             : {len(household)}")
    print(f"unique homes               : {household['homeid'].nunique()}")


def _replace_cell_source(notebook_path: Path, old: str, new: str) -> None:
    nb = json.loads(notebook_path.read_text())
    replaced = False
    for cell in nb["cells"]:
        source = "".join(cell.get("source", []))
        if old in source:
            source = source.replace(old, new)
            cell["source"] = source.splitlines(keepends=True)
            replaced = True
    if not replaced:
        raise ValueError(f"Expected source block not found in {notebook_path}")
    notebook_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1))


def patch_notebooks() -> None:
    print("\n[4/4] Patching notebook logic")

    stage1_old = (
        'daily_electricity_features["complete_day"] = daily_electricity_features["block_count"] == 12\n'
        'daily_electricity_features["flat_day_flag"] = daily_electricity_features["complete_day"] & (daily_electricity_features["daily_range_w"] < FLAT_THRESHOLD_W)\n'
        'daily_electricity_features["cluster_ready_day"] = daily_electricity_features["complete_day"] & (daily_electricity_features["daily_range_w"] >= FLAT_THRESHOLD_W)\n'
    )
    stage1_new = (
        'daily_electricity_features["complete_day"] = daily_electricity_features["block_count"] == 12\n'
        'daily_electricity_features["flat_day_flag"] = daily_electricity_features["complete_day"] & (daily_electricity_features["daily_range_w"] < FLAT_THRESHOLD_W)\n'
        'daily_electricity_features["cluster_ready_day"] = False\n'
    )

    stage1_old_2 = (
        'daily_electricity_features["normalisable_day"] = daily_electricity_features["cluster_ready_day"] & daily_electricity_features["daily_range_w"].gt(0)\n'
        '\n'
        'block_min = daily_electricity_features[BLOCK_COLS].min(axis=1)\n'
        'block_range = daily_electricity_features[BLOCK_COLS].max(axis=1) - block_min\n'
        '\n'
        'norm_cols = [f"norm_{col}" for col in BLOCK_COLS]\n'
        'daily_electricity_features[norm_cols] = np.nan\n'
        '\n'
        'valid_norm = daily_electricity_features["normalisable_day"] & block_range.gt(0)\n'
    )
    stage1_new_2 = (
        'daily_electricity_features["normalisable_day"] = (\n'
        '    daily_electricity_features["complete_day"]\n'
        '    & (~daily_electricity_features["flat_day_flag"])\n'
        '    & daily_electricity_features["daily_range_w"].gt(0)\n'
        ')\n'
        '\n'
        'block_min = daily_electricity_features[BLOCK_COLS].min(axis=1)\n'
        'block_range = daily_electricity_features[BLOCK_COLS].max(axis=1) - block_min\n'
        '\n'
        'norm_cols = [f"norm_{col}" for col in BLOCK_COLS]\n'
        'daily_electricity_features[norm_cols] = np.nan\n'
        '\n'
        'valid_norm = daily_electricity_features["normalisable_day"] & block_range.gt(0)\n'
    )

    stage1_old_3 = (
        'daily_electricity_features.loc[valid_norm, norm_cols] = (\n'
        '    daily_electricity_features.loc[valid_norm, BLOCK_COLS]\n'
        '    .sub(block_min[valid_norm], axis=0)\n'
        '    .div(block_range[valid_norm], axis=0)\n'
        '    .to_numpy()\n'
        ')\n'
        '\n'
        'print("Normalisation complete.")\n'
    )
    stage1_new_3 = (
        'daily_electricity_features.loc[valid_norm, norm_cols] = (\n'
        '    daily_electricity_features.loc[valid_norm, BLOCK_COLS]\n'
        '    .sub(block_min[valid_norm], axis=0)\n'
        '    .div(block_range[valid_norm], axis=0)\n'
        '    .to_numpy()\n'
        ')\n'
        '\n'
        'daily_electricity_features["cluster_ready_day"] = (\n'
        '    daily_electricity_features["normalisable_day"]\n'
        '    & daily_electricity_features[norm_cols].notna().all(axis=1)\n'
        ')\n'
        '\n'
        'print("Normalisation complete.")\n'
    )

    stage2_old = (
        'cluster_df = daily_df[daily_df["cluster_ready_day"].fillna(False)].copy()\n'
        '\n'
        'if USE_FLAT_DAY_FILTER and FLAT_DAY_THRESHOLD_W is not None:\n'
        '    cluster_df = cluster_df[cluster_df["daily_range_w"] > FLAT_DAY_THRESHOLD_W].copy()\n'
        '\n'
        'cluster_df = cluster_df.dropna(subset=NORM_COLS).reset_index(drop=True)\n'
        'X = cluster_df[NORM_COLS].to_numpy()\n'
    )
    stage2_new = (
        'cluster_df = daily_df[daily_df["cluster_ready_day"].fillna(False)].copy()\n'
        '\n'
        'if USE_FLAT_DAY_FILTER and FLAT_DAY_THRESHOLD_W is not None:\n'
        '    cluster_df = cluster_df[cluster_df["daily_range_w"] > FLAT_DAY_THRESHOLD_W].copy()\n'
        '\n'
        'cluster_df = cluster_df[cluster_df[NORM_COLS].notna().all(axis=1)].reset_index(drop=True)\n'
        'X = cluster_df[NORM_COLS].to_numpy(dtype=float)\n'
    )

    _replace_cell_source(STAGE1_NOTEBOOK, stage1_old, stage1_new)
    _replace_cell_source(STAGE1_NOTEBOOK, stage1_old_2, stage1_new_2)
    _replace_cell_source(STAGE1_NOTEBOOK, stage1_old_3, stage1_new_3)
    _replace_cell_source(STAGE2_NOTEBOOK, stage2_old, stage2_new)

    print(f"patched: {STAGE1_NOTEBOOK}")
    print(f"patched: {STAGE2_NOTEBOOK}")


def main() -> None:
    ensure_output_dirs()
    daily_df = fix_daily_features()
    fix_daily_archetypes(daily_df)
    sanity_check_households()
    patch_notebooks()


if __name__ == "__main__":
    main()
