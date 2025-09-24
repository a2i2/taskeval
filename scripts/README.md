# Scripts

This directory contains utility scripts for data preparation and evaluation tasks.

## Data Preparation

### `data_preparation/sample_qa.py`

Samples QA data from a CSV file and copies relevant PDFs to a target directory.

```bash
poetry run python scripts/data_preparation/sample_qa.py \
  --qa_csv_path="path/to/qa_data.csv" \
  --pdf_dir_path="path/to/pdf_directory" \
  --n_samples=30 \
  --output_dir_path="./outputs"
```

#### Data

Current sample data can be downloaded from here.

1. [Question set](https://drive.google.com/file/d/1DKNTLwweMWofCLK-NpVJgVuE45pglkJ6/view?usp=drive_link)
2. [PDFs](https://drive.google.com/file/d/1DGYqfiGHondqO5LfTMmtT1Mhn0rfJiTN/view?usp=drive_link)

**Notes**: Please contact the admin of the repo if you need access to this files.

## Evaluations

### `evaluations/qa_evals.py`

Runs QA evaluation on a dataset of sampled questions and documents, generating the outcome from the pipeline ready for manual evaluation.

```bash
poetry run python scripts/evaluations/qa_evals.py \
  --qa_csv_path="path/to/qa_data.csv" \
  --pdf_dir_path="path/to/pdf_directory" \
  --n_samples=5 \
  --output_dir_path="./outputs/qa_evals"
```

### `evaluations/figure_extractions.py`

Evaluates figure extraction capabilities on chart images, generating CSV outputs and regenerated charts ready for manual evaluation.

```bash
poetry run python scripts/evaluations/figure_extractions.py \
  --input_dir_path="path/to/chart_images" \
  --output_dir_path="./outputs/figure_extractions"
```

### `evaluations/expand_qa_results.py`

Expands QA evaluation results by flattening nested data structures into separate columns when required.

```bash
poetry run python scripts/evaluations/expand_qa_results.py \
  --result_csv_path="path/to/qa_evals.csv" \
  --output_csv_path="./outputs/qa_evals/expanded_qa_evals.csv"
```

## Visualisation

### `visualise_batch/figure_ui.py`

Interactive Streamlit app for visualizing figure extraction results. Compare original chart images with regenerated charts and view extracted table data.

```bash
streamlit run scripts/visualise_batch/figure_ui.py
```

**Features:**

- Side-by-side comparison of original and regenerated charts
- Interactive table viewer for extracted data
- Index-based navigation through results

**Default paths:**

- Original chart images: `./data/new_chart_images/`
- Generated results: `./outputs/figure_extractions/`

### `visualise_batch/qa_ui.py`

Interactive Streamlit app for visualizing QA evaluation results. View questions, answers, evaluation metrics, and highlighted source sections.

```bash
streamlit run scripts/visualise_batch/qa_ui.py
```

**Features:**

- Question and answer comparison
- Evaluation metrics display (Relevancy, Faithfulness, Contextual Precision/Recall)
- Highlighted source sections with match counts and ordered

**Default path:**

- QA evals results: `./outputs/qa_evals/expanded_qa_evals.csv`

## Notes

- All scripts use Python Fire for command-line argument parsing
- Make sure to run `poetry install` and set up environment variables as described in the root README.md
- Scripts will create output directories automatically if they don't exist
- Use `--help` with any script to see detailed parameter information
- For visualisation scripts, ensure you have the required output files from evaluation scripts before running
