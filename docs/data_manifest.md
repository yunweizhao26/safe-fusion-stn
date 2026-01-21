# Data manifest

This repo does **not** include large datasets or full imputed matrices.
Place files under the paths below after downloading.

## Required datasets

| Dataset | GEO | Local path |
| --- | --- | --- |
| CBMC (CITE-seq) | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE100866 | `ep_dataset/GSE100866/GSE100866_CBMC_8K_13AB_10X-RNA_umi.h5ad` |
| PBMC (CITE-seq) | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE100501 | `ep_dataset/GSE100501/GSE100501_PBMC_RNA_umi.h5ad` |
| MNC (CITE-seq) | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE128639 | `ep_dataset/GSE128639/GSE128639_MNC_RNA_umi.h5ad` |

## Validation panels

Place the CSVs under:
- `ep_dataset/GSE100866/cbmc_cite_panel.csv`
- `ep_dataset/GSE100501/pbmc_cite_panel.csv`
- `ep_dataset/GSE128639/mnc_cite_panel.csv`

## Marker files

Marker JSONs expected under:
- `output/cbmc_cluster_markers.json`
- `output/markers/pbmc_curated_markers.json`
- `output/markers/mnc_curated_markers.json`
