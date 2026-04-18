from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from output_paths import STAGE2_DIR, STAGE3_DIR, ensure_output_dirs


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "script"

DAILY_PATH = STAGE2_DIR / "daily_archetypes_enhanced.csv"
HOUSEHOLD_PATH = STAGE2_DIR / "household_archetypes_enhanced.csv"

DAY_PROFILE_OUT = STAGE3_DIR / "archetype_day_profile_enhanced.csv"
HOUSEHOLD_PROFILE_OUT = STAGE3_DIR / "archetype_household_profile_enhanced.csv"
TEMPORAL_PROFILE_OUT = STAGE3_DIR / "archetype_temporal_profile_enhanced.csv"
STABILITY_OUT = STAGE3_DIR / "household_archetype_stability_enhanced.csv"
STABILITY_SUMMARY_OUT = STAGE3_DIR / "household_archetype_stability_summary_enhanced.csv"

DAY_SHARE_PLOT = STAGE3_DIR / "archetype_day_share_enhanced.png"
TEMPORAL_PLOT = STAGE3_DIR / "archetype_temporal_mix_enhanced.png"
HOUSEHOLD_PLOT = STAGE3_DIR / "archetype_household_profile_enhanced.png"
STABILITY_PLOT = STAGE3_DIR / "household_archetype_stability_scatter_enhanced.png"

ARCHETYPE_ORDER = [
    "Morning peak",
    "All-daytime plateau",
    "Evening peak",
    "Late-afternoon peak",
    "Early-morning peak",
]


def mode_or_missing(series: pd.Series):
    s = series.dropna()
    non_missing = s[s != "Missing"]
    if not non_missing.empty:
        non_missing_mode = non_missing.mode()
        if not non_missing_mode.empty:
            return non_missing_mode.iat[0]
    mode = s.mode()
    return mode.iat[0] if not mode.empty else "Missing"


def build_day_profile(daily_df: pd.DataFrame) -> pd.DataFrame:
    total_days = len(daily_df)
    profile = (
        daily_df.groupby(["daily_archetype", "archetype_name"]).agg(
            n_days=("homeid", "size"),
            n_homes=("homeid", "nunique"),
            mean_daily_total_kwh=("daily_total_kwh", "mean"),
            median_daily_total_kwh=("daily_total_kwh", "median"),
            mean_peak_hour=("day_peak_hour", "mean"),
            weekend_share=("is_weekend", "mean"),
            mean_daytime_fraction=("daytime_fraction", "mean"),
        )
        .reset_index()
    )
    profile["day_share"] = profile["n_days"] / total_days
    return profile


def build_temporal_profile(daily_df: pd.DataFrame) -> pd.DataFrame:
    weekend_mix = (
        daily_df.assign(day_count=1)
        .pivot_table(
            index=["daily_archetype", "archetype_name"],
            columns="is_weekend",
            values="day_count",
            aggfunc="sum",
            fill_value=0,
        )
        .rename(columns={False: "weekday_days", True: "weekend_days"})
        .reset_index()
    )
    weekend_mix["weekday_share_within_archetype"] = (
        weekend_mix["weekday_days"] / (weekend_mix["weekday_days"] + weekend_mix["weekend_days"])
    )
    weekend_mix["weekend_share_within_archetype"] = (
        weekend_mix["weekend_days"] / (weekend_mix["weekday_days"] + weekend_mix["weekend_days"])
    )

    season_mix = (
        daily_df.assign(day_count=1)
        .pivot_table(
            index=["daily_archetype", "archetype_name"],
            columns="season",
            values="day_count",
            aggfunc="sum",
            fill_value=0,
        )
        .rename(columns=lambda c: f"{str(c).lower()}_days")
        .reset_index()
    )

    total_by_arch = daily_df.groupby(["daily_archetype", "archetype_name"]).size().rename("n_days").reset_index()
    temporal = weekend_mix.merge(season_mix, on=["daily_archetype", "archetype_name"], how="left").merge(
        total_by_arch, on=["daily_archetype", "archetype_name"], how="left"
    )

    for season in ["autumn", "spring", "summer", "winter"]:
        days_col = f"{season}_days"
        temporal[f"{season}_share_within_archetype"] = temporal[days_col] / temporal["n_days"]

    return temporal


def build_household_profile(household_df: pd.DataFrame) -> pd.DataFrame:
    profile = (
        household_df.groupby(["household_modal_archetype", "household_modal_archetype_name"]).agg(
            modal_homes=("homeid", "size"),
            mean_residents=("residents", "mean"),
            median_residents=("residents", "median"),
            mean_major_app_total=("major_app_total", "mean"),
            mean_other_app_total=("other_app_total", "mean"),
            modal_hometype=("hometype", mode_or_missing),
            modal_income_band=("income_band", mode_or_missing),
            modal_workingstatus=("workingstatus", mode_or_missing),
        )
        .reset_index()
        .rename(
            columns={
                "household_modal_archetype": "daily_archetype",
                "household_modal_archetype_name": "archetype_name",
            }
        )
    )
    return profile


def build_household_stability(daily_df: pd.DataFrame, household_df: pd.DataFrame) -> pd.DataFrame:
    home_mix = (
        daily_df.assign(day_count=1)
        .pivot_table(index="homeid", columns="archetype_name", values="day_count", aggfunc="sum", fill_value=0)
        .reset_index()
    )
    archetype_cols = [c for c in home_mix.columns if c != "homeid"]
    home_mix["clustered_days"] = home_mix[archetype_cols].sum(axis=1)
    home_mix["modal_days"] = home_mix[archetype_cols].max(axis=1)
    home_mix["modal_day_share"] = home_mix["modal_days"] / home_mix["clustered_days"]
    home_mix["n_observed_archetypes"] = (home_mix[archetype_cols] > 0).sum(axis=1)

    home_mix = home_mix.merge(
        household_df[
            [
                "homeid",
                "household_modal_archetype",
                "household_modal_archetype_name",
                "residents",
                "major_app_total",
                "other_app_total",
                "income_band",
                "hometype",
            ]
        ],
        on="homeid",
        how="left",
    )
    return home_mix


def build_stability_summary(stability_df: pd.DataFrame) -> pd.DataFrame:
    overall = pd.DataFrame(
        [
            {
                "scope": "overall",
                "households": len(stability_df),
                "mean_modal_day_share": stability_df["modal_day_share"].mean(),
                "median_modal_day_share": stability_df["modal_day_share"].median(),
                "min_modal_day_share": stability_df["modal_day_share"].min(),
                "max_modal_day_share": stability_df["modal_day_share"].max(),
                "mean_n_observed_archetypes": stability_df["n_observed_archetypes"].mean(),
                "homes_with_all_5_archetypes": int((stability_df["n_observed_archetypes"] == 5).sum()),
            }
        ]
    )

    by_archetype = (
        stability_df.groupby(["household_modal_archetype", "household_modal_archetype_name"]).agg(
            households=("homeid", "size"),
            mean_modal_day_share=("modal_day_share", "mean"),
            median_modal_day_share=("modal_day_share", "median"),
            min_modal_day_share=("modal_day_share", "min"),
            max_modal_day_share=("modal_day_share", "max"),
            mean_n_observed_archetypes=("n_observed_archetypes", "mean"),
        )
        .reset_index()
        .rename(
            columns={
                "household_modal_archetype": "daily_archetype",
                "household_modal_archetype_name": "scope",
            }
        )
    )

    return pd.concat([overall, by_archetype], ignore_index=True, sort=False)


def ordered(df: pd.DataFrame, name_col: str = "archetype_name") -> pd.DataFrame:
    df = df.copy()
    df[name_col] = pd.Categorical(df[name_col], categories=ARCHETYPE_ORDER, ordered=True)
    return df.sort_values(name_col).reset_index(drop=True)


def plot_outputs(
    day_profile: pd.DataFrame,
    temporal_profile: pd.DataFrame,
    household_profile: pd.DataFrame,
    stability_df: pd.DataFrame,
) -> None:
    day_profile = ordered(day_profile)
    temporal_profile = ordered(temporal_profile)
    household_profile = ordered(household_profile)

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(day_profile["archetype_name"], day_profile["day_share"], color="#4C78A8")
    ax1.set_ylabel("Share of clustered days")
    ax1.set_title("Daily archetype prevalence")
    ax1.tick_params(axis="x", rotation=25)
    ax1.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig1.savefig(DAY_SHARE_PLOT, dpi=200, bbox_inches="tight")
    plt.close(fig1)

    fig2, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(
        temporal_profile["archetype_name"],
        temporal_profile["weekday_share_within_archetype"],
        label="Weekday",
        color="#4C78A8",
    )
    axes[0].bar(
        temporal_profile["archetype_name"],
        temporal_profile["weekend_share_within_archetype"],
        bottom=temporal_profile["weekday_share_within_archetype"],
        label="Weekend",
        color="#F58518",
    )
    axes[0].set_title("Weekday vs weekend mix")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].set_ylabel("Share within archetype")
    axes[0].legend(frameon=False)

    season_cols = [
        "autumn_share_within_archetype",
        "spring_share_within_archetype",
        "summer_share_within_archetype",
        "winter_share_within_archetype",
    ]
    season_labels = ["Autumn", "Spring", "Summer", "Winter"]
    season_colors = ["#E45756", "#72B7B2", "#54A24B", "#B279A2"]
    bottom = pd.Series(0.0, index=temporal_profile.index)
    for col, label, color in zip(season_cols, season_labels, season_colors):
        axes[1].bar(temporal_profile["archetype_name"], temporal_profile[col], bottom=bottom, label=label, color=color)
        bottom = bottom + temporal_profile[col]
    axes[1].set_title("Seasonal mix")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].set_ylabel("Share within archetype")
    axes[1].legend(frameon=False)
    plt.tight_layout()
    fig2.savefig(TEMPORAL_PLOT, dpi=200, bbox_inches="tight")
    plt.close(fig2)

    fig3, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(household_profile["archetype_name"], household_profile["mean_residents"], color="#72B7B2")
    axes[0].set_title("Mean residents by modal archetype")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].set_ylabel("Mean residents")
    axes[0].grid(axis="y", alpha=0.3)

    x = range(len(household_profile))
    axes[1].bar(x, household_profile["mean_major_app_total"], label="Major appliances", color="#4C78A8")
    axes[1].bar(
        x,
        household_profile["mean_other_app_total"],
        bottom=household_profile["mean_major_app_total"],
        label="Other appliances",
        color="#F58518",
    )
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(household_profile["archetype_name"], rotation=25)
    axes[1].set_title("Mean appliance stock by modal archetype")
    axes[1].set_ylabel("Mean appliance count")
    axes[1].legend(frameon=False)
    plt.tight_layout()
    fig3.savefig(HOUSEHOLD_PLOT, dpi=200, bbox_inches="tight")
    plt.close(fig3)

    color_map = {
        "Morning peak": "#4C78A8",
        "All-daytime plateau": "#F58518",
        "Evening peak": "#54A24B",
        "Late-afternoon peak": "#E45756",
        "Early-morning peak": "#B279A2",
    }

    fig4, ax4 = plt.subplots(figsize=(8, 6))
    for archetype_name, group in stability_df.groupby("household_modal_archetype_name"):
        ax4.scatter(
            group["residents"],
            group["modal_day_share"],
            s=70,
            alpha=0.85,
            label=archetype_name,
            color=color_map.get(archetype_name, "#666666"),
        )
    ax4.set_title("Household-level archetype stability")
    ax4.set_xlabel("Residents")
    ax4.set_ylabel("Modal day share")
    ax4.grid(alpha=0.3)
    ax4.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    fig4.savefig(STABILITY_PLOT, dpi=200, bbox_inches="tight")
    plt.close(fig4)


def main() -> None:
    ensure_output_dirs()
    daily_df = pd.read_csv(DAILY_PATH, low_memory=False)
    household_df = pd.read_csv(HOUSEHOLD_PATH, low_memory=False)

    day_profile = build_day_profile(daily_df)
    temporal_profile = build_temporal_profile(daily_df)
    household_profile = build_household_profile(household_df)
    stability = build_household_stability(daily_df, household_df)
    stability_summary = build_stability_summary(stability)

    day_profile = ordered(day_profile)
    temporal_profile = ordered(temporal_profile)
    household_profile = ordered(household_profile)
    stability = stability.sort_values(["household_modal_archetype", "homeid"]).reset_index(drop=True)
    stability_summary = stability_summary.reset_index(drop=True)

    day_profile.to_csv(DAY_PROFILE_OUT, index=False)
    temporal_profile.to_csv(TEMPORAL_PROFILE_OUT, index=False)
    household_profile.to_csv(HOUSEHOLD_PROFILE_OUT, index=False)
    stability.to_csv(STABILITY_OUT, index=False)
    stability_summary.to_csv(STABILITY_SUMMARY_OUT, index=False)

    plot_outputs(day_profile, temporal_profile, household_profile, stability)

    print(f"saved: {DAY_PROFILE_OUT}")
    print(f"saved: {TEMPORAL_PROFILE_OUT}")
    print(f"saved: {HOUSEHOLD_PROFILE_OUT}")
    print(f"saved: {STABILITY_OUT}")
    print(f"saved: {STABILITY_SUMMARY_OUT}")
    print(f"saved: {DAY_SHARE_PLOT}")
    print(f"saved: {TEMPORAL_PLOT}")
    print(f"saved: {HOUSEHOLD_PLOT}")
    print(f"saved: {STABILITY_PLOT}")


if __name__ == "__main__":
    main()
