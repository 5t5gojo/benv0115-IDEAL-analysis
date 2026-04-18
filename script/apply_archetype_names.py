from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from output_paths import STAGE2_DIR, ensure_output_dirs


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "script"
NOTEBOOK_PATH = SCRIPT_DIR / "0145_stage2_clustering.ipynb"

DAILY_PATH = STAGE2_DIR / "daily_archetypes_enhanced.csv"
HOUSEHOLD_PATH = STAGE2_DIR / "household_archetypes_enhanced.csv"
CENTROID_PATH = STAGE2_DIR / "daily_archetype_centroids_enhanced.csv"
SUMMARY_PATH = STAGE2_DIR / "daily_archetype_summary_enhanced.csv"
PLOT_PATH = STAGE2_DIR / "daily_archetype_centroids_enhanced.png"

ARCHETYPE_NAMES = {
    0: "Morning peak",
    1: "All-daytime plateau",
    2: "Evening peak",
    3: "Late-afternoon peak",
    4: "Early-morning peak",
}

PATTERN_NOTES = {
    0: "broad daytime-elevated profile",
    1: "broad plateau profile",
    2: "broad evening-focused profile",
    3: "narrow-peak subtype",
    4: "narrow-peak subtype",
}

NORM_COLS = [f"norm_block_{hour:02d}" for hour in range(0, 24, 2)]
PLOT_HOURS = [int(col.replace("norm_block_", "")) for col in NORM_COLS]
PLOT_LABELS = [f"{hour:02d}:00" for hour in PLOT_HOURS]


def _mode_or_na(series: pd.Series):
    mode = series.dropna().mode()
    return mode.iat[0] if not mode.empty else pd.NA


def update_outputs() -> None:
    daily = pd.read_csv(DAILY_PATH, low_memory=False)
    household = pd.read_csv(HOUSEHOLD_PATH, low_memory=False)
    centroids = pd.read_csv(CENTROID_PATH, low_memory=False)

    daily["archetype_name"] = daily["daily_archetype"].map(ARCHETYPE_NAMES)
    daily["pattern_note"] = daily["daily_archetype"].map(PATTERN_NOTES)

    household["household_modal_archetype_name"] = household["household_modal_archetype"].map(ARCHETYPE_NAMES)
    household["household_pattern_note"] = household["household_modal_archetype"].map(PATTERN_NOTES)

    centroids["archetype_name"] = centroids["daily_archetype"].map(ARCHETYPE_NAMES)
    centroids["pattern_note"] = centroids["daily_archetype"].map(PATTERN_NOTES)
    centroids = centroids[["daily_archetype", "archetype_name", "pattern_note"] + NORM_COLS]

    home_modal = household[["homeid", "household_modal_archetype", "household_modal_archetype_name"]].rename(
        columns={
            "household_modal_archetype": "daily_archetype",
            "household_modal_archetype_name": "archetype_name",
        }
    )

    day_summary = (
        daily.groupby("daily_archetype").agg(
            n_days=("homeid", "size"),
            n_homes=("homeid", "nunique"),
            mean_daily_total_kwh=("daily_total_kwh", "mean"),
            median_daily_total_kwh=("daily_total_kwh", "median"),
            mean_daytime_fraction=("daytime_fraction", "mean"),
            mean_peak_hour=("day_peak_hour", "mean"),
            weekend_share=("is_weekend", "mean"),
        )
        .reset_index()
    )
    day_summary["archetype_name"] = day_summary["daily_archetype"].map(ARCHETYPE_NAMES)
    day_summary["pattern_note"] = day_summary["daily_archetype"].map(PATTERN_NOTES)

    season_share = (
        daily.assign(day_count=1)
        .pivot_table(index="daily_archetype", columns="season", values="day_count", aggfunc="sum", fill_value=0)
        .rename(columns=lambda c: f"{c.lower()}_share")
        .div(daily.groupby("daily_archetype").size(), axis=0)
        .reset_index()
    )

    home_summary = (
        home_modal.merge(household, on="homeid", how="left")
        .groupby(["daily_archetype", "archetype_name"]).agg(
            modal_homes=("homeid", "size"),
            mean_residents=("residents", "mean"),
            mean_major_app_total=("major_app_total", "mean"),
            mean_other_app_total=("other_app_total", "mean"),
            modal_hometype=("hometype", _mode_or_na),
            modal_income_band=("income_band", _mode_or_na),
        )
        .reset_index()
    )

    summary = (
        day_summary
        .merge(season_share, on="daily_archetype", how="left")
        .merge(home_summary, on=["daily_archetype", "archetype_name"], how="left")
        .sort_values("daily_archetype")
        .reset_index(drop=True)
    )

    ordered_cols = [
        "daily_archetype",
        "archetype_name",
        "pattern_note",
        "n_days",
        "n_homes",
        "modal_homes",
        "mean_daily_total_kwh",
        "median_daily_total_kwh",
        "mean_daytime_fraction",
        "mean_peak_hour",
        "weekend_share",
        "autumn_share",
        "spring_share",
        "summer_share",
        "winter_share",
        "mean_residents",
        "mean_major_app_total",
        "mean_other_app_total",
        "modal_hometype",
        "modal_income_band",
    ]
    summary = summary[ordered_cols]

    fig, ax = plt.subplots(figsize=(10, 6))
    for _, row in centroids.sort_values("daily_archetype").iterrows():
        label = f"{row['archetype_name']} ({row['pattern_note']})" if "narrow-peak" in row["pattern_note"] else row["archetype_name"]
        ax.plot(
            PLOT_HOURS,
            row[NORM_COLS].to_numpy(dtype=float),
            marker="o",
            linewidth=2,
            label=label,
        )

    ax.set_title("Daily Electricity Archetype Centroids")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Normalised load")
    ax.set_xticks(PLOT_HOURS)
    ax.set_xticklabels(PLOT_LABELS, rotation=45)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Archetype", frameon=False)
    plt.tight_layout()
    fig.savefig(PLOT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)

    daily.to_csv(DAILY_PATH, index=False)
    household.to_csv(HOUSEHOLD_PATH, index=False)
    centroids.to_csv(CENTROID_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    print(f"saved: {DAILY_PATH}")
    print(f"saved: {HOUSEHOLD_PATH}")
    print(f"saved: {CENTROID_PATH}")
    print(f"saved: {SUMMARY_PATH}")
    print(f"saved: {PLOT_PATH}")


def patch_notebook() -> None:
    nb = json.loads(NOTEBOOK_PATH.read_text())

    replacements = [
        (
            'SELECTED_K = 5\n\ncentroids, labels = kmeans2(X, SELECTED_K, minit="points", iter=100)\ncluster_df["daily_archetype"] = labels\n\ncentroids_df = pd.DataFrame(centroids, columns=NORM_COLS)\ncentroids_df.insert(0, "daily_archetype", range(SELECTED_K))\n\ncluster_df[["homeid", "date", "daily_archetype"] + NORM_COLS[:3]].head()\n',
            'SELECTED_K = 5\nARCHETYPE_NAMES = {\n    0: "Morning peak",\n    1: "All-daytime plateau",\n    2: "Evening peak",\n    3: "Late-afternoon peak",\n    4: "Early-morning peak",\n}\nPATTERN_NOTES = {\n    0: "broad daytime-elevated profile",\n    1: "broad plateau profile",\n    2: "broad evening-focused profile",\n    3: "narrow-peak subtype",\n    4: "narrow-peak subtype",\n}\n\ncentroids, labels = kmeans2(X, SELECTED_K, minit="points", iter=100)\ncluster_df["daily_archetype"] = labels\ncluster_df["archetype_name"] = cluster_df["daily_archetype"].map(ARCHETYPE_NAMES)\ncluster_df["pattern_note"] = cluster_df["daily_archetype"].map(PATTERN_NOTES)\n\ncentroids_df = pd.DataFrame(centroids, columns=NORM_COLS)\ncentroids_df.insert(0, "daily_archetype", range(SELECTED_K))\ncentroids_df["archetype_name"] = centroids_df["daily_archetype"].map(ARCHETYPE_NAMES)\ncentroids_df["pattern_note"] = centroids_df["daily_archetype"].map(PATTERN_NOTES)\ncentroids_df = centroids_df[["daily_archetype", "archetype_name", "pattern_note"] + NORM_COLS]\n\ncluster_df[["homeid", "date", "daily_archetype", "archetype_name"] + NORM_COLS[:3]].head()\n',
        ),
        (
            'household_archetypes = household_modal.merge(household_mix, on="homeid", how="left").merge(home_df, on="homeid", how="left")\nhousehold_archetypes.head()\n',
            'household_archetypes = household_modal.merge(household_mix, on="homeid", how="left").merge(home_df, on="homeid", how="left")\nhousehold_archetypes["household_modal_archetype_name"] = household_archetypes["household_modal_archetype"].map(ARCHETYPE_NAMES)\nhousehold_archetypes["household_pattern_note"] = household_archetypes["household_modal_archetype"].map(PATTERN_NOTES)\nhousehold_archetypes.head()\n',
        ),
        (
            'fig, ax = plt.subplots(figsize=(10, 6))\nfor _, row in centroids_df.iterrows():\n    ax.plot(\n        plot_hours,\n        row[NORM_COLS].to_numpy(dtype=float),\n        marker="o",\n        linewidth=2,\n        label=f"Archetype {int(row[\'daily_archetype\'])}",\n    )\n\nax.set_title("Daily Electricity Archetype Centroids")\n',
            'fig, ax = plt.subplots(figsize=(10, 6))\nfor _, row in centroids_df.sort_values("daily_archetype").iterrows():\n    label = row["archetype_name"]\n    if "narrow-peak" in row["pattern_note"]:\n        label = f"{label} ({row[\'pattern_note\']})"\n    ax.plot(\n        plot_hours,\n        row[NORM_COLS].to_numpy(dtype=float),\n        marker="o",\n        linewidth=2,\n        label=label,\n    )\n\nax.set_title("Daily Electricity Archetype Centroids")\n',
        ),
        (
            'ax.legend(title="Cluster")\n',
            'ax.legend(title="Archetype", frameon=False)\n',
        ),
        (
            'daily_archetype_summary = (\n    cluster_df.groupby("daily_archetype").agg(\n        n_days=("homeid", "size"),\n        n_homes=("homeid", "nunique"),\n        mean_daily_total_raw=("daily_total_raw", "mean"),\n        median_daily_total_raw=("daily_total_raw", "median"),\n        mean_daytime_fraction_raw=("daytime_fraction_raw", "mean"),\n        mean_peak_hour=("day_peak_hour", "mean"),\n        weekend_share=("is_weekend", "mean"),\n    )\n    .reset_index()\n)\n\nseason_mix = (\n    cluster_df.assign(day_count=1)\n    .pivot_table(index="daily_archetype", columns="season", values="day_count", aggfunc="sum", fill_value=0)\n    .add_prefix("season_days_")\n    .reset_index()\n)\n\nhome_modal = household_archetypes[["homeid", "household_modal_archetype"]].rename(columns={"household_modal_archetype": "daily_archetype"})\nhome_level_summary = (\n    home_modal.merge(home_df, on="homeid", how="left")\n    .groupby("daily_archetype").agg(\n        modal_homes=("homeid", "size"),\n        mean_residents=("residents", "mean"),\n        mean_major_app_total=("major_app_total", "mean"),\n        mean_other_app_total=("other_app_total", "mean"),\n    )\n    .reset_index()\n)\n\narchetype_summary_df = (\n    daily_archetype_summary\n    .merge(season_mix, on="daily_archetype", how="left")\n    .merge(home_level_summary, on="daily_archetype", how="left")\n    .sort_values("daily_archetype")\n    .reset_index(drop=True)\n)\n\narchetype_summary_df\n',
            'daily_archetype_summary = (\n    cluster_df.groupby("daily_archetype").agg(\n        n_days=("homeid", "size"),\n        n_homes=("homeid", "nunique"),\n        mean_daily_total_kwh=("daily_total_kwh", "mean"),\n        median_daily_total_kwh=("daily_total_kwh", "median"),\n        mean_daytime_fraction=("daytime_fraction", "mean"),\n        mean_peak_hour=("day_peak_hour", "mean"),\n        weekend_share=("is_weekend", "mean"),\n    )\n    .reset_index()\n)\ndaily_archetype_summary["archetype_name"] = daily_archetype_summary["daily_archetype"].map(ARCHETYPE_NAMES)\ndaily_archetype_summary["pattern_note"] = daily_archetype_summary["daily_archetype"].map(PATTERN_NOTES)\n\nseason_share = (\n    cluster_df.assign(day_count=1)\n    .pivot_table(index="daily_archetype", columns="season", values="day_count", aggfunc="sum", fill_value=0)\n    .rename(columns=lambda c: f"{c.lower()}_share")\n    .div(cluster_df.groupby("daily_archetype").size(), axis=0)\n    .reset_index()\n)\n\nhome_modal = household_archetypes[[\"homeid\", \"household_modal_archetype\", \"household_modal_archetype_name\"]].rename(\n    columns={\n        \"household_modal_archetype\": \"daily_archetype\",\n        \"household_modal_archetype_name\": \"archetype_name\",\n    }\n)\nhome_level_summary = (\n    home_modal.merge(home_df, on=\"homeid\", how=\"left\")\n    .groupby([\"daily_archetype\", \"archetype_name\"]).agg(\n        modal_homes=(\"homeid\", \"size\"),\n        mean_residents=(\"residents\", \"mean\"),\n        mean_major_app_total=(\"major_app_total\", \"mean\"),\n        mean_other_app_total=(\"other_app_total\", \"mean\"),\n        modal_hometype=(\"hometype\", lambda x: x.dropna().mode().iat[0] if not x.dropna().mode().empty else pd.NA),\n        modal_income_band=(\"income_band\", lambda x: x.dropna().mode().iat[0] if not x.dropna().mode().empty else pd.NA),\n    )\n    .reset_index()\n)\n\narchetype_summary_df = (\n    daily_archetype_summary\n    .merge(season_share, on=\"daily_archetype\", how=\"left\")\n    .merge(home_level_summary, on=[\"daily_archetype\", \"archetype_name\"], how=\"left\")\n    .sort_values(\"daily_archetype\")\n    .reset_index(drop=True)\n)\n\narchetype_summary_df = archetype_summary_df[[\n    \"daily_archetype\",\n    \"archetype_name\",\n    \"pattern_note\",\n    \"n_days\",\n    \"n_homes\",\n    \"modal_homes\",\n    \"mean_daily_total_kwh\",\n    \"median_daily_total_kwh\",\n    \"mean_daytime_fraction\",\n    \"mean_peak_hour\",\n    \"weekend_share\",\n    \"autumn_share\",\n    \"spring_share\",\n    \"summer_share\",\n    \"winter_share\",\n    \"mean_residents\",\n    \"mean_major_app_total\",\n    \"mean_other_app_total\",\n    \"modal_hometype\",\n    \"modal_income_band\",\n]]\n\narchetype_summary_df\n',
        ),
    ]

    for cell in nb["cells"]:
        source = "".join(cell.get("source", []))
        for old, new in replacements:
            if old in source:
                source = source.replace(old, new)
        cell["source"] = source.splitlines(keepends=True)

    NOTEBOOK_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    print(f"patched: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    ensure_output_dirs()
    update_outputs()
    patch_notebook()
