# Student Engagement Digital Twin

A research prototype for generating synthetic learning-activity data and evaluating
whether models trained on that data can predict student engagement. The project
compares CTGAN, TVAE, and RealTabFormer, recreates observed missingness, evaluates
statistical fidelity and predictive utility, and provides a Streamlit dashboard.

This is an offline synthetic-data experiment, not a live learning-management-system
(LMS) digital twin. Predictions, explanations, and what-if plots are exploratory and
must not be used to make high-stakes decisions about individual students. Synthetic
data similarity is not a privacy guarantee, and SHAP explanations are not causal.

## What the project does

- Prepares Canvas Network Person-Course data and derives a binary engagement label
  from course-completion percentage.
- Preserves missingness using two strategies:
  - **Random / proportion-based:** restores missing values at rates observed in the
    real training data.
  - **From-column / indicator-based:** models missingness indicators and uses them
    to restore missing values in synthetic records.
- Trains and compares CTGAN, TVAE, and RealTabFormer synthetic-data generators.
- Measures synthetic-data fidelity using column-shape, pairwise trend/correlation,
  and missing-value similarity.
- Trains XGBoost classifiers on synthetic data and evaluates them on a held-out real
  test set, including precision, recall, and F1 for the high-engagement class.
- Uses SHAP to explain fitted model behavior and includes an interactive dashboard
  for predictions, comparisons, explanations, and exploratory what-if analysis.

## Results at a glance

The project dataset contains 325,199 raw records, of which 26,661 have a usable
engagement target. A stratified split with seed 42 produces 21,328 training records
and 5,333 test records. The four activity features are `nevents`, `ndays_act`,
`ncontent`, and `nforum_posts`.

| Synthetic training data | Missingness strategy | Weighted F1 | High-engagement F1 |
|---|---|---:|---:|
| CTGAN | Random | 0.8669 | 0.5814 |
| TVAE | Random | 0.8956 | 0.6858 |
| CTGAN | From-column | 0.8948 | 0.6835 |
| TVAE | From-column | 0.8961 | 0.7078 |
| RealTabFormer | Random | 0.9036 | 0.7149 |
| RealTabFormer | From-column | 0.9053 | 0.7198 |

Weighted F1 is affected by class imbalance: approximately 82.5% of the test set is
low-engagement. Read it alongside the high-engagement metrics. CTGAN/TVAE fidelity
results are summarized across four seeds; the classification results use one
classifier seed, and RealTabFormer results use one generator seed per strategy.
These results are descriptive, not evidence that one generator is universally best.

See [the project report](./project_report.md) for methodology, full result tables,
limitations, and references.

## Requirements

- Python 3.11 (the project metadata requires `>=3.11,<3.12`).
- Windows PowerShell examples are provided below; adapt activation commands for
  other shells.
- The pinned dependencies are listed in [`pyproject.toml`](./pyproject.toml).
  Generator training, particularly full RealTabFormer runs, can require substantial
  compute time and disk space.
- The raw dataset is not included in this repository.

## Setup

Create and activate a virtual environment, then install the project and development
dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Download the de-identified [Canvas Network Person-Course dataset](https://doi.org/10.7910/DVN/1XORAL)
from Harvard Dataverse. Place the tab-separated file at:

```text
data/raw/CNPC_1401-1509_DI_v1_1_2016-03-01.tab
```

The expected path and other project settings are defined in
[`configs/base.yaml`](./configs/base.yaml). The data preparation step expects the
source file to contain the configured activity, completion, and identifier columns.

## Run the dashboard

The dashboard uses saved evaluation reports and classifier models; it does not train
generators when it starts.

```powershell
python -m streamlit run app/streamlit_app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`). Dashboard
pages include:

- **Overview:** dataset and classification summaries.
- **Prediction:** estimated high-engagement probability from activity inputs.
- **Real vs synthetic:** saved fidelity comparisons.
- **Explainability:** saved global SHAP feature importance and plots.
- **What-if:** exploratory one-feature prediction sweeps.

Some views depend on saved reports or model files. If one is absent, the dashboard
reports that the content is unavailable rather than generating substitute results.

## Reproduce the experiments

The notebooks contain the main CTGAN/TVAE workflow. Run them in order in Jupyter:

| Notebook | Purpose |
|---|---|
| `00_api_checks.ipynb` | Check relevant library APIs and compatibility |
| `01_data.ipynb` | Clean, label, split, and audit the source data |
| `02_preprocessing.ipynb` | Fit preprocessing and prepare generator inputs |
| `03_labeler.ipynb` | Explore the separate labeler baseline |
| `04_generators.ipynb` | Train or load CTGAN/TVAE and generate synthetic data |
| `05_fidelity.ipynb` | Evaluate synthetic-data fidelity |
| `06_classification.ipynb` | Train/evaluate XGBoost models |
| `07_explainability.ipynb` | Calculate and visualize SHAP explanations |

Reusable pipeline components are in `src/engagement_twin/`. For example, after
installing the project and placing the raw data at the configured path, prepare the
training/test split and data audit with:

```powershell
python -c "from engagement_twin.data import prepare; train, test, audit = prepare(); print(audit)"
```

Full-size RealTabFormer runs are available as scripts. They require prepared data and
can take a long time:

```powershell
python scripts/run_realtabformer_full.py --strategy random
python scripts/run_realtabformer_full.py --strategy from_column
python scripts/evaluate_realtabformer_full.py --strategy both
```

The `--resume` option resumes an interrupted training run. Use
`--recover-checkpoint` only when a completed checkpoint already exists and needs to
be converted into a saved generator, synthetic dataset, and generation report. Check
the scripts' `--help` output for current options.

## Tests and lint

```powershell
python -m pytest tests -q
python -m ruff check src tests app scripts
```

The automated tests cover data validation and splitting, preprocessing and
missingness restoration, classifier behavior, evaluation metrics, and smoke-level
integration. They do not retrain the large synthetic-data generators.

## Repository layout

```text
app/                    Streamlit dashboard
artifacts/models/        Saved XGBoost classifiers used by the dashboard
artifacts/realtabformer/ RealTabFormer outputs, reports, and evaluation results
configs/base.yaml        Dataset paths and experiment settings
data/raw/                Downloaded source dataset (not tracked)
data/processed/          Prepared train/test data (generated locally)
notebooks/               Experiment workflow and analysis
reports/                 Evaluation summaries and figures
scripts/                 RealTabFormer and utility scripts
src/engagement_twin/     Reusable data, preprocessing, model, and evaluation code
tests/                   Automated tests
```

Large generator checkpoints and model weights are not included in the GitHub
repository. The raw dataset must be obtained from its source, and excluded model
artifacts must be regenerated to repeat those experiments.

## Data and interpretation limitations

- The work uses a single historical dataset and four activity features; results may
  not generalize to other courses, institutions, or current learners.
- Missingness is especially high for `nforum_posts` and may represent behavior as
  well as data collection.
- Class imbalance affects aggregate scores; a higher recall can come with lower
  precision and more false positives.
- Statistical fidelity does not establish privacy, fairness, or downstream safety.
- SHAP describes model behavior, not causes; dashboard what-if results are
  non-causal.
- The dashboard is a research prototype and is not connected to a live LMS.

## References

1. S. Kumi et al., “A Data-Driven Digital Twin for Student Engagement Prediction in
   e-Learning Systems,” *2025 IEEE World AI IoT Congress (AIIoT)*, 2025.
   [doi:10.1109/AIIoT65859.2025.11105257](https://doi.org/10.1109/AIIoT65859.2025.11105257).
2. Harvard Dataverse, “Canvas Network Person-Course (1/2014–9/2015) De-Identified
   Open Dataset.” [doi:10.7910/DVN/1XORAL](https://doi.org/10.7910/DVN/1XORAL).
