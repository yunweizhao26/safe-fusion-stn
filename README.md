# Signal-to-Noise
Key functions:
- Reproduce the **main table** at fill rate `0.081`.
- Reproduce **F1 vs fill-rate curves** for CBMC/PBMC/MNC.
- Reproduce the **MNC q-sweep** summary.

Large datasets and full imputed matrices are **not** included; see Data Setup.

## Environment

- Python 3.10
- Create conda env

Install minimal deps:
```bash
pip install -r requirements.txt
```

## Data setup

Download and place these h5ad files under `ep_dataset/`:

- CBMC: `ep_dataset/GSE100866/GSE100866_CBMC_8K_13AB_10X-RNA_umi.h5ad`
  - GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE100866
- PBMC: `ep_dataset/GSE100501/GSE100501_PBMC_RNA_umi.h5ad`
  - GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE100501
- MNC: `ep_dataset/GSE128639/GSE128639_MNC_RNA_umi.h5ad`
  - GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE128639

Marker files (already included in repo root):
- `output/cbmc_cluster_markers.json`
- `output/markers/pbmc_curated_markers.json`
- `output/markers/mnc_curated_markers.json`

## Methods

**Training / inference (fusion)**
- `scripts/run_fusion_methods.py`

**Evaluation + safe-fusion**
- `precision_validation_experiments.py`

**Baselines**
- `run_simple_imputers.py` (gene_median, svd_impute, graph_smooth)
- `run_magic.py`
- `run_scvi.py`

## the main table

Precomputed CSVs are in `results/main_tables/`.
These are the exact rows used in the paper table.

To regenerate from scratch:

1) Baselines (example: CBMC)
```bash
python run_simple_imputers.py \
  --input_file ep_dataset/GSE100866/GSE100866_CBMC_8K_13AB_10X-RNA_umi.h5ad \
  --disease CBMC --tissue CBMC \
  --methods gene_median,svd_impute,graph_smooth \
  --output_root output
```

2) Fusion (selection-only scVI)
```bash
python scripts/run_fusion_methods.py \
  --input_file ep_dataset/GSE100866/GSE100866_CBMC_8K_13AB_10X-RNA_umi.h5ad \
  --disease CBMC --tissue CBMC \
  --teacher_methods gene_median,svd_impute,graph_smooth,MAGIC,scVI \
  --teacher_likelihood_exclude scVI \
  --best_teacher_weight 0.8 \
  --teacher_dropout 0.4 \
  --best_teacher_temp 0.5 \
  --best_teacher_min_log 1.0 \
  --best_teacher_exclude gene_median \
  --methods latent_truth \
  --output_root output/latent_truth_scvi_select
```

3) Evaluation + safe-fusion (q=0.90)
```bash
python precision_validation_experiments.py \
  --input_file ep_dataset/GSE100866/GSE100866_CBMC_8K_13AB_10X-RNA_umi.h5ad \
  --disease CBMC --tissue CBMC \
  --methods raw,gene_median,svd_impute,graph_smooth,MAGIC,scVI,latent_truth_scvi_select \
  --markers_file output/cbmc_cluster_markers.json \
  --validation_panels ep_dataset/GSE100866/cbmc_cite_panel.csv \
  --latent_prob_eps 1.0 --latent_score_mode z \
  --output_dir results/main_tables/cbmc \
  --safe_fusion_method latent_truth_scvi_select \
  --safe_fusion_teachers scVI,MAGIC,graph_smooth,svd_impute \
  --safe_fusion_conf_quantile 0.90 \
  --safe_fusion_name safe_latent_truth_scvi_select
```

Repeat for PBMC/MNC (MNC uses `--max_cells 40000`).

## MNC q-sweep

Summary CSV (fixed grid) is at:
- `results/q_sweep/mnc_safe_fusion_q_sweep.csv`

To regenerate:
```bash
for q in 0.70 0.75 0.80 0.85 0.90 0.95; do
  qtag=${q/./p}
  python precision_validation_experiments.py \
    --input_file ep_dataset/GSE128639/GSE128639_MNC_RNA_umi.h5ad \
    --disease MNC --tissue MNC \
    --methods raw,gene_median,svd_impute,graph_smooth,MAGIC,scVI,latent_truth_scvi_select \
    --markers_file output/markers/mnc_curated_markers.json \
    --validation_panels ep_dataset/GSE128639/mnc_cite_panel.csv \
    --latent_prob_eps 1.0 --latent_score_mode z \
    --max_cells 40000 \
    --output_dir results/q_sweep/mnc_q${qtag} \
    --safe_fusion_method latent_truth_scvi_select \
    --safe_fusion_teachers scVI,MAGIC,graph_smooth,svd_impute \
    --safe_fusion_conf_quantile $q \
    --safe_fusion_name safe_latent_truth_scvi_select
done
```

## Seed sweep

Summary CSV:
- `results/seed_sweep/seed_summary.csv`

This file reports mean +- std and worst-case over 5 training seeds.

## Results

Small CSV artifacts are stored under `results/` and are safe to commit.
Large data (h5ad) and full imputed matrices are **not** included.
