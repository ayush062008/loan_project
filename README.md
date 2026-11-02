# Mining Lab – Data Mining pipeline + Django web app

Upload a raw CSV/XLSX → inspect it → automatic cleaning and model training → charts → live predictions.
The same pipeline powers both a command-line run (Part 1) and the website (Part 2).

## Quick start

```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python manage.py runserver
```
Open http://127.0.0.1:8000 and click **Try the sample loan dataset**, or upload your own file.

Part 1 only (no web server):
```bash
python run_pipeline.py data/loan_approval_sample.csv Loan_Status      # charts + model go to ./outputs/
```

## Use a real Kaggle dataset
`data/loan_approval_sample.csv` is a generated stand-in (1,225 rows, Kaggle *Loan Prediction* schema, with deliberately
injected nulls, duplicates and outliers) so the project runs offline. For submission, download a real dataset
(e.g. *Loan Prediction Problem Dataset*, *Telco Customer Churn*, *House Prices*, *Heart Disease*) and upload it in the app,
or pass it to `run_pipeline.py`. No code changes are needed: the target column is chosen in the UI, the task
(classification / regression) is detected automatically, and the prediction form is generated from the columns.

## Project layout
```
config/                 Django settings, urls, wsgi
mining/
  ml_pipeline.py        PART 1 – all data mining + ML logic (no Django imports)
  views.py urls.py      PART 2 – upload, inspect, summary, dashboard, predict (+ JSON API)
  forms.py              upload validation (type, size)
  templates/mining/     base, upload, dataset, summary, dashboard, predict
  static/mining/        style.css
data/                   sample dataset
generate_sample_data.py recreates the sample
run_pipeline.py         command-line runner for Part 1
media/runs/<id>/        per-upload workspace: raw file, charts (PNG), model.joblib, result.json
```

## How the requirements map to the code

| Requirement | Where | What happens |
|---|---|---|
| Dataset (500+ rows, numeric + categorical) | `data/`, upload page | any CSV/XLSX; sample has 1,225 rows with numeric and categorical columns |
| Missing values | `clean_data` | detected per column; numeric → median (skewed) / mean (symmetric); discrete numeric and categorical → mode |
| Outliers | `clean_data` | detected with **IQR** and **Z-score**; values capped at the IQR fences (no rows lost) |
| Duplicates / irrelevant fields | `clean_data` | exact duplicate rows dropped; ID-like, constant and high-cardinality text columns dropped |
| Scaling + encoding | `build_model` | `StandardScaler` + `OneHotEncoder` inside a scikit-learn `Pipeline` / `ColumnTransformer` |
| Feature selection | `select_features`, `build_model` | correlation with target; redundant features (abs r > 0.9) removed; permutation feature importance |
| EDA & visuals | `make_charts` | target distribution, histograms, box plots (before/after), categorical-vs-outcome, scatter, correlation heatmap, confusion matrix, importance |
| Model + metrics | `build_model` | Logistic/Linear Regression, Random Forest, Gradient Boosting compared by cross-validation; best one evaluated on a 20 % hold-out: Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix (or RMSE, MAE, R²) |
| Upload page | `upload` view | drag-and-drop, validates extension and size |
| Data summary | `dataset`, `summary` views | shape, dtypes, top-10 rows, missing counts before **and** after processing |
| Dashboard | `dashboard` view | charts are generated per upload, not pre-made |
| Prediction | `predict` view + `predict_api` | metrics plus an auto-generated form; JSON POST returns the prediction and class probabilities without reloading the page |

## Notes for your viva / report
* **Why cap outliers instead of deleting them?** Keeps the sample size; extreme incomes are real people, just unusual.
* **Why impute both in `clean_data` and in the model pipeline?** The first is for reporting and EDA; the second guarantees any blank in the prediction form is handled identically in production.
* **Leakage:** the train/test split happens before model fitting, and scaling/encoding are fitted on the training split only (they live inside the pipeline).
* **Imbalanced data:** accuracy alone can mislead, which is why precision, recall, F1 and ROC-AUC are shown too.
* `DEBUG=True` and the fixed `SECRET_KEY` in `settings.py` are for local use only; change both before deploying.
