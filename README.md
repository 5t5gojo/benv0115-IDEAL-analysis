# IDEAL Enhanced Homes Analysis

This repository contains the analysis pipeline and summarised outputs for the 39 IDEAL enhanced homes electricity archetype study.

## What Is Included

- `script/0145(3).ipynb`: Stage 1 data preparation
- `script/0145_stage2_clustering.ipynb`: Stage 2 daily archetype clustering
- `script/0145_stage3_descriptive_analysis.ipynb`: Stage 3 descriptive profiling
- `script/0145_stage4_context_analysis.ipynb`: Stage 4 context and temporal stability analysis
- `script/*.py`: helper scripts for bug fixes, naming, output paths, and reproducible reruns
- `script/outputs/`: organised result tables and figures

## Output Structure

- `script/outputs/stage1_preparation/`: cleaned preparation outputs and daily feature tables
- `script/outputs/stage2_clustering/`: clustering results, centroids, validation tables, and clustering figures
- `script/outputs/stage3_descriptive/`: descriptive profiling tables and figures
- `script/outputs/stage4_context_analysis/`: context-specific clustering, switching, and temperature analyses
- `script/outputs/legacy/`: archived older files kept only for reference

## Not Included

The raw IDEAL source data are intentionally excluded from version control:

- `sensordata/`
- `metadata_and_surveys (1)/`
- `room_and_appliance_sensors/`

These data are large and may also be subject to access restrictions.

## Suggested Starting Point

If you only want to review the analysis results, start with:

- `script/outputs/stage2_clustering/daily_archetype_centroids_enhanced.png`
- `script/outputs/stage2_clustering/daily_archetype_summary_enhanced.csv`
- `script/outputs/stage3_descriptive/`
- `script/outputs/stage4_context_analysis/`

If you want to rerun the workflow, run the notebooks in stage order from Stage 1 to Stage 4.
