# Predicting Human Perfume Similarity Judgments from Description Embeddings and Fragrance Notes

Helin Ökmen - Tilburg University BSc Cognitive Science and Artificial Intelligence - 2026

## Project Overview

This project is the coding component of my bachelor thesis. It investigates whether computational similarity measures derived from perfume descriptions and fragrance notes can predict human olfactory similarity judgments collected in a smell-based experiment.

Three computational similarity measures were evaluated:

- MiniLM sentence-transformer description embedding similarity
- TF-IDF description similarity
- Fragrance-note overlap (Jaccard similarity)

A combined embedding and note-overlap regression model was also evaluated.
## Main Findings

The analyses showed that fragrance-note overlap substantially outperformed language-based description similarity measures in predicting human olfactory similarity judgments. MiniLM embeddings showed limited predictive performance, while TF-IDF similarity performed weakly overall.

## Folder Structure
thesis_coding_component/
├── data/
│   ├── final_pairs.csv                  # The 18 perfume pairs used in the experiment
│   ├── final_master_dataset.csv         # Pair-level computational similarity scores
│   ├── desc_emb_minilm.npy              # Pre-computed MiniLM embeddings (not included, 123MB)
│   ├── desc_emb_minilm_meta.csv         # Metadata aligned with embedding rows
│   └── pair_level_human_summary.csv     # Mean human similarity ratings per pair
├── scripts/
│   ├── make_desc_embeddings_safe.py          # Generates MiniLM embeddings from descriptions
│   ├── recompute_final_pairs_scores.py       # Computes similarity scores for the 18 pairs
│   ├── find_candidate_pairs.py               # Generates candidate pairs for manual selection
│   ├── prepare_human_ratings.py              # Processes Qualtrics data into pair-level ratings
│   ├── run_prediction_analysis.py            # Runs LOOCV prediction models (main analysis)
│   ├── run_model_diagnostics.py              # Runs OLS regression and generates diagnostic plots
│   ├── category_comparison_analysis.py       # Runs Kruskal-Wallis and Mann-Whitney U tests
│   ├── validating_analysis_dataset.py        # Validation checks on the merged dataset
│   └── plot_model_performance.py             # Generates bar chart of model performance
└── README.md

Note: `fragrantica_dataset/` is not included in this repository. The original dataset is not redistributed due to data ownership considerations (see Dataset Availability). `desc_emb_minilm.npy` exceeds GitHub file size limit and is also not included. Both are available upon request.

## How to Reproduce the Results

The scripts should be run in this order:

1. `make_desc_embeddings_safe.py` - generates MiniLM embeddings from perfume descriptions (see note below)
2. `recompute_final_pairs_scores.py` - computes similarity scores for the 18 final pairs
3. `prepare_human_ratings.py` - processes the Qualtrics export into pair-level ratings
4. `run_prediction_analysis.py` - runs the LOOCV prediction models and produces the main results
5. `run_model_diagnostics.py` - runs full-sample OLS regression and generates diagnostic figures
6. `category_comparison_analysis.py` - runs nonparametric category comparison tests
7. `validating_analysis_dataset.py` - runs validation checks on the merged dataset
8. `plot_model_performance.py` - generates the model performance bar chart

**Note:** Steps 3 onwards can be run directly using the files already in the `data/` folder. Step 1 requires PyTorch >= 2.4 and the original dataset. If your environment has an older PyTorch version, skip step 1 and request the pre-computed `desc_emb_minilm.npy` file.

## Dataset Availability

The original dataset was a publicly available Kaggle dataset containing perfume descriptions and fragrance-note information from Fragrantica.com (Hussein, 2024). The dataset was last accessed on April 17, 2026. At the time this project was finalized, the original dataset was no longer publicly accessible online.

The processed similarity scores, selected perfume pairs, and human rating data needed to reproduce the prediction analyses are included in this repository. Fully reproducing the preprocessing pipeline from scratch requires the original `perfumes_table.csv`, which is preserved locally but not redistributed due to data ownership considerations.

## Dependencies

Python 3.12.7 (Anaconda environment)

| Library | Version |
|---|---|
| pandas | 2.2.2 |
| NumPy | 1.26.4 |
| scikit-learn | 1.5.1 |
| SciPy | 1.13.1 |
| statsmodels | 0.14.2 |
| matplotlib | 3.9.2 |
| seaborn | 0.13.2 |
| sentence-transformers | 5.2.3 |

Embedding model: `all-MiniLM-L6-v2`

## Reference

Hussein, J. (2024). Fragrantica Data [Data set]. Kaggle. https://www.kaggle.com/datasets/joehusseinmama/fragrantica-data (Last accessed: April 17, 2026)
