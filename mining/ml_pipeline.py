"""
Part 1 - Data Mining & ML pipeline (framework-independent; no Django imports).

Stages
  1. load_dataset      read CSV / XLSX
  2. clean_data        duplicates, irrelevant columns, missing values, outliers
  3. select_features   correlation analysis (redundant feature removal)
  4. build_model       scaling + encoding inside a scikit-learn Pipeline, model comparison
  5. make_charts       EDA + evaluation visuals saved as PNG
  6. run_pipeline      orchestrates everything and returns a JSON-friendly result dict
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LinearRegression, LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, classification_report, confusion_matrix, f1_score,
    mean_absolute_error, mean_squared_error, precision_score, r2_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

warnings.filterwarnings("ignore")

INK, TEAL, AMBER, SLATE = "#14212B", "#0B5563", "#E8A317", "#8A9BA8"
sns.set_theme(style="whitegrid", rc={
    "axes.edgecolor": "#C9D3DA", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.titleweight": "bold",
    "axes.titlesize": 11, "figure.dpi": 100,
})
CLASS_COLORS = [TEAL, AMBER, "#7B5EA7", "#C8553D", "#4F9D69", SLATE]

MAX_CATEGORY_LEVELS = 50     # categoricals above this are treated as free text / IDs
DISCRETE_LIMIT = 10          # numeric columns with <= this many distinct values are not outlier-capped
SAMPLE_FOR_PLOTS = 5000


# ----------------------------------------------------------------------------
# 1. Loading
# ----------------------------------------------------------------------------
def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    ext = path.suffix.lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif ext == ".csv":
        try:
            df = pd.read_csv(path)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1")
    else:
        raise ValueError("Unsupported file type. Please upload a .csv or .xlsx file.")

    df.columns = [str(c).strip() for c in df.columns]
    for c in df.columns:                                   # tidy text cells
        if not pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]):
            s = df[c].astype("string").str.strip()
            df[c] = s.replace({"": pd.NA, "NA": pd.NA, "N/A": pd.NA, "nan": pd.NA, "NaN": pd.NA,
                               "null": pd.NA, "NULL": pd.NA, "?": pd.NA}).astype(object)
            df[c] = df[c].where(df[c].notna(), np.nan)
    df = df.dropna(axis=1, how="all")
    if len(df) < 20:
        raise ValueError("The dataset needs at least 20 rows to train a model.")
    return df


def is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def raw_summary(df: pd.DataFrame) -> dict:
    """Quick facts shown right after upload (before any processing)."""
    miss = df.isna().sum()
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "duplicates": int(df.duplicated().sum()),
        "numeric_cols": [c for c in df.columns if is_numeric(df[c])],
        "categorical_cols": [c for c in df.columns if not is_numeric(df[c])],
        "missing_total": int(miss.sum()),
        "columns_info": [
            {"name": c, "dtype": str(df[c].dtype), "missing": int(miss[c]),
             "missing_pct": round(100 * miss[c] / len(df), 1), "unique": int(df[c].nunique())}
            for c in df.columns
        ],
        "preview_html": df.head(10).to_html(classes="grid", index=False, border=0, na_rep="–"),
    }


def suggest_target(df: pd.DataFrame) -> str:
    """Heuristic default for the target dropdown."""
    hints = ("status", "target", "label", "class", "outcome", "churn", "approved", "price", "default")
    for c in reversed(list(df.columns)):
        if any(h in c.lower() for h in hints):
            return c
    return df.columns[-1]


def detect_task(y: pd.Series) -> str:
    if not is_numeric(y):
        return "classification"
    return "classification" if y.nunique() <= DISCRETE_LIMIT else "regression"


# ----------------------------------------------------------------------------
# 2. Cleaning
# ----------------------------------------------------------------------------
def _find_irrelevant(df: pd.DataFrame, target: str) -> dict[str, str]:
    """Columns that carry no predictive signal: IDs, constants, free text."""
    drop: dict[str, str] = {}
    n = len(df)
    for c in df.columns:
        if c == target:
            continue
        nun = df[c].nunique(dropna=True)
        if nun <= 1:
            drop[c] = "constant column"
        elif re.search(r"(^id$|_id$|^id_|\bid\b|index|unnamed|serial)", c.lower()):
            drop[c] = "identifier column"
        elif not is_numeric(df[c]) and nun > MAX_CATEGORY_LEVELS:
            drop[c] = f"high-cardinality text ({nun} distinct values)"
        elif not is_numeric(df[c]) and nun / max(n, 1) > 0.9:
            drop[c] = "unique per row (identifier-like)"
    return drop


def clean_data(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, dict, dict]:
    """Returns (clean_df, report, bounds)."""
    rep: dict = {"shape_before": [int(len(df)), int(df.shape[1])]}
    df = df.copy()
    rep["missing_before"] = {c: int(v) for c, v in df.isna().sum().items() if v > 0}
    rep["missing_before_total"] = int(df.isna().sum().sum())

    # duplicates
    dup = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    rep["duplicates_removed"] = dup

    # unusable target rows
    bad_target = int(df[target].isna().sum())
    df = df.dropna(subset=[target]).reset_index(drop=True)
    rep["rows_dropped_missing_target"] = bad_target

    # irrelevant columns
    dropped = _find_irrelevant(df, target)
    df = df.drop(columns=list(dropped))
    rep["dropped_columns"] = dropped

    # missing value imputation
    impute: dict[str, str] = {}
    for c in df.columns:
        if c == target or df[c].isna().sum() == 0:
            continue
        if is_numeric(df[c]) and df[c].nunique() <= DISCRETE_LIMIT:
            df[c] = df[c].fillna(df[c].mode().iloc[0]); impute[c] = "mode (discrete / flag-like numeric column)"
        elif is_numeric(df[c]):
            if abs(df[c].skew()) > 1:
                df[c] = df[c].fillna(df[c].median()); impute[c] = "median (skewed distribution)"
            else:
                df[c] = df[c].fillna(df[c].mean()); impute[c] = "mean (symmetric distribution)"
        else:
            df[c] = df[c].fillna(df[c].mode().iloc[0]); impute[c] = "mode (most frequent category)"
    rep["imputation"] = impute
    rep["missing_after"] = {c: int(v) for c, v in df.isna().sum().items() if v > 0}
    rep["missing_after_total"] = int(df.isna().sum().sum())

    # outliers: detect with IQR and Z-score, treat by winsorizing to the IQR fences
    bounds: dict[str, list[float]] = {}
    outliers = []
    for c in df.columns:
        if c == target or not is_numeric(df[c]) or df[c].nunique() <= DISCRETE_LIMIT:
            continue
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_iqr = int(((df[c] < lo) | (df[c] > hi)).sum())
        std = df[c].std()
        n_z = int((((df[c] - df[c].mean()) / std).abs() > 3).sum()) if std else 0
        bounds[c] = [float(lo), float(hi)]
        if n_iqr:
            df[c] = df[c].clip(lo, hi)
        outliers.append({"column": c, "iqr_outliers": n_iqr, "zscore_outliers": n_z,
                         "lower": round(float(lo), 2), "upper": round(float(hi), 2),
                         "treatment": "capped to IQR fences" if n_iqr else "none needed"})
    rep["outliers"] = outliers
    rep["shape_after"] = [int(len(df)), int(df.shape[1])]
    return df, rep, bounds


# ----------------------------------------------------------------------------
# 3. Feature selection
# ----------------------------------------------------------------------------
def select_features(df: pd.DataFrame, target: str, task: str) -> tuple[list[str], dict]:
    feats = [c for c in df.columns if c != target]
    num = [c for c in feats if is_numeric(df[c])]
    info: dict = {"removed_redundant": [], "target_correlation": []}

    # redundancy: of two numeric features with |r| > 0.9 keep the first
    if len(num) > 1:
        corr = df[num].corr().abs()
        drop = set()
        for i, a in enumerate(num):
            for b in num[i + 1:]:
                if a not in drop and b not in drop and corr.loc[a, b] > 0.9:
                    drop.add(b)
                    info["removed_redundant"].append({"dropped": b, "kept": a, "r": round(float(corr.loc[a, b]), 3)})
        feats = [c for c in feats if c not in drop]

    # relevance: correlation of every feature with the (encoded) target
    y_code = df[target].astype("category").cat.codes if task == "classification" else df[target]
    rows = []
    for c in feats:
        x = df[c] if is_numeric(df[c]) else df[c].astype("category").cat.codes
        r = float(np.corrcoef(x, y_code)[0, 1]) if x.nunique() > 1 else 0.0
        rows.append({"feature": c, "correlation": round(0.0 if np.isnan(r) else r, 3)})
    info["target_correlation"] = sorted(rows, key=lambda d: -abs(d["correlation"]))
    return feats, info


# ----------------------------------------------------------------------------
# 4. Model
# ----------------------------------------------------------------------------
def _preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num, num_cols), ("cat", cat, cat_cols)])


def build_model(X: pd.DataFrame, y: pd.Series, task: str) -> dict:
    num_cols = [c for c in X.columns if is_numeric(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    stratify = y if task == "classification" and y.value_counts().min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)

    if task == "classification":
        candidates = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        }
        scoring, score_name = "f1_weighted", "CV F1 (weighted)"
    else:
        candidates = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        }
        scoring, score_name = "r2", "CV R²"

    comparison, best_name, best_score = [], None, -np.inf
    for name, est in candidates.items():
        pipe = Pipeline([("prep", _preprocessor(num_cols, cat_cols)), ("model", est)])
        cv = min(5, max(2, int(y_tr.value_counts().min()))) if task == "classification" else 5
        score = float(cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring=scoring).mean())
        comparison.append({"model": name, "cv_score": round(score, 4)})
        if score > best_score:
            best_name, best_score = name, score

    best = Pipeline([("prep", _preprocessor(num_cols, cat_cols)), ("model", candidates[best_name])])
    best.fit(X_tr, y_tr)
    pred, train_pred = best.predict(X_te), best.predict(X_tr)

    result: dict = {"task": task, "best_model": best_name, "score_name": score_name,
                    "comparison": comparison, "train_rows": int(len(X_tr)), "test_rows": int(len(X_te))}

    if task == "classification":
        labels = sorted(y.unique().tolist())
        avg = "binary" if len(labels) == 2 else "weighted"
        kw = {"average": avg, "zero_division": 0}
        if avg == "binary":
            kw["pos_label"] = labels[-1] if "Y" not in labels else "Y"
        m = {
            "accuracy": accuracy_score(y_te, pred),
            "precision": precision_score(y_te, pred, **kw),
            "recall": recall_score(y_te, pred, **kw),
            "f1": f1_score(y_te, pred, **kw),
            "train_accuracy": accuracy_score(y_tr, train_pred),
        }
        if len(labels) == 2:
            result["positive_label"] = kw["pos_label"]
            try:
                m["roc_auc"] = roc_auc_score(y_te, best.predict_proba(X_te)[:, list(best.classes_).index(kw["pos_label"])])
            except Exception:
                pass
        result["metrics"] = {k: round(float(v), 4) for k, v in m.items()}
        result["metric_note"] = ("Precision / recall / F1 computed for the positive class '%s'." % kw["pos_label"]
                                 if avg == "binary" else "Precision / recall / F1 are weighted averages across classes.")
        result["labels"] = labels
        result["confusion_matrix"] = confusion_matrix(y_te, pred, labels=labels).tolist()
        result["confusion_rows"] = [{"label": l, "counts": row} for l, row in zip(labels, result["confusion_matrix"])]
        rep = classification_report(y_te, pred, labels=labels, output_dict=True, zero_division=0)
        result["class_report"] = [{"label": l, "precision": round(rep[l]["precision"], 3),
                                   "recall": round(rep[l]["recall"], 3), "f1": round(rep[l]["f1-score"], 3),
                                   "support": int(rep[l]["support"])} for l in labels]
        result["headline"] = {"name": "Accuracy", "value": f"{m['accuracy'] * 100:.1f}%"}
    else:
        rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
        result["metrics"] = {"r2": round(float(r2_score(y_te, pred)), 4), "rmse": round(rmse, 4),
                             "mae": round(float(mean_absolute_error(y_te, pred)), 4),
                             "train_r2": round(float(r2_score(y_tr, train_pred)), 4)}
        result["headline"] = {"name": "R² score", "value": f"{result['metrics']['r2']:.3f}"}

    # model-agnostic importance on the ORIGINAL columns (works through one-hot encoding)
    n_imp = min(len(X_te), 1000)
    imp = permutation_importance(best, X_te.iloc[:n_imp], y_te.iloc[:n_imp], n_repeats=5, random_state=42,
                                 scoring="f1_weighted" if task == "classification" else "r2")
    result["importance"] = sorted(
        [{"feature": c, "importance": round(float(v), 4)} for c, v in zip(X.columns, imp.importances_mean)],
        key=lambda d: -d["importance"])
    result["_pipeline"] = best
    result["_pred"], result["_y_test"] = pred, y_te
    return result


# ----------------------------------------------------------------------------
# 5. Charts
# ----------------------------------------------------------------------------
def _grid(n: int, ncols: int = 3, w: float = 4.2, h: float = 3.2):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.axis("off")
    return fig, axes[:n]


def _save(fig, out: Path, name: str, title: str, caption: str, charts: list):
    fig.tight_layout()
    fig.savefig(out / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    charts.append({"file": name, "title": title, "caption": caption})


def make_charts(raw: pd.DataFrame, df: pd.DataFrame, target: str, task: str, features: list[str],
                model: dict, out: Path) -> dict[str, list]:
    out.mkdir(parents=True, exist_ok=True)
    eda, evalc = [], []
    sample = df.sample(min(len(df), SAMPLE_FOR_PLOTS), random_state=1)
    num = [c for c in features if is_numeric(df[c])]
    cat = [c for c in features if c not in num]
    classes = sorted(df[target].astype(str).unique()) if task == "classification" else []
    palette = {c: CLASS_COLORS[i % len(CLASS_COLORS)] for i, c in enumerate(classes)}

    # missing values (before treatment)
    miss = raw.isna().sum()
    miss = miss[miss > 0].sort_values()
    if len(miss):
        fig, ax = plt.subplots(figsize=(7, max(2.2, 0.45 * len(miss) + 1)))
        ax.barh(miss.index, miss.values, color=AMBER)
        for i, v in enumerate(miss.values):
            ax.text(v, i, f" {v}", va="center", fontsize=9)
        ax.set_xlabel("Missing cells in the raw upload")
        _save(fig, out, "missing_before.png", "Missing values (raw data)",
              "Columns with null cells before imputation. After cleaning every column has zero.", eda)

    # target distribution
    fig, ax = plt.subplots(figsize=(6, 3.6))
    if task == "classification":
        vc = df[target].astype(str).value_counts().reindex(classes)
        ax.bar(vc.index, vc.values, color=[palette[c] for c in vc.index])
        for i, v in enumerate(vc.values):
            ax.text(i, v, f"{v}\n({100 * v / vc.sum():.0f}%)", ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, vc.max() * 1.22)
        ax.set_ylabel("Records")
        cap = "How many records fall in each outcome class. A strong imbalance would call for resampling."
    else:
        sns.histplot(df[target], kde=True, color=TEAL, ax=ax)
        cap = "Distribution of the value being predicted."
    ax.set_title(f"Target: {target}")
    _save(fig, out, "target.png", "Target distribution", cap, eda)

    # numeric distributions
    if num:
        show = num[:9]
        fig, axes = _grid(len(show))
        for ax, c in zip(axes, show):
            sns.histplot(sample[c], kde=True, color=TEAL, ax=ax, edgecolor="white")
            ax.set_title(c); ax.set_xlabel(""); ax.set_ylabel("")
        _save(fig, out, "distributions.png", "Numeric feature distributions",
              "Histograms with density curves (after outlier treatment). Skewed shapes explain the median imputation.", eda)

        # box plots: before vs after outlier treatment
        show = [c for c in num if raw[c].nunique() > DISCRETE_LIMIT][:6] if all(c in raw for c in num) else []
        if show:
            fig, axes = _grid(len(show), ncols=3, h=3.0)
            for ax, c in zip(axes, show):
                d = pd.DataFrame({"Raw": raw[c].reset_index(drop=True),
                                  "Treated": df[c].reset_index(drop=True)})
                sns.boxplot(data=d, ax=ax, palette=[AMBER, TEAL], width=0.5, fliersize=2.5)
                ax.set_title(c)
            _save(fig, out, "boxplots.png", "Box plots: before and after outlier treatment",
                  "Amber is the raw column, teal is after IQR capping. Dots are points beyond 1.5×IQR.", eda)

    # categorical vs target
    if cat and task == "classification":
        show = [c for c in cat if df[c].nunique() <= 8][:6]
        if show:
            fig, axes = _grid(len(show))
            for ax, c in zip(axes, show):
                ct = pd.crosstab(df[c].astype(str), df[target].astype(str), normalize="index")[classes] * 100
                ct.plot(kind="bar", stacked=True, ax=ax, color=[palette[k] for k in classes], legend=False, width=0.75)
                ax.set_title(c); ax.set_xlabel(""); ax.set_ylabel("% of group")
                ax.tick_params(axis="x", rotation=25)
            axes[0].legend(title=target, fontsize=8, title_fontsize=8)
            _save(fig, out, "categorical.png", "Categorical features vs outcome",
                  "Each bar shows how the outcome splits inside a category - big differences mean a useful predictor.", eda)

    # scatter plots on the two most important numeric features
    imp_rank = [d["feature"] for d in model["importance"] if d["feature"] in num] or num
    if len(imp_rank) >= 2 and task == "classification":
        a, b = imp_rank[:2]
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        for k in classes:
            s = sample[sample[target].astype(str) == k]
            ax.scatter(s[a], s[b], s=16, alpha=0.55, color=palette[k], label=k, edgecolor="none")
        ax.set_xlabel(a); ax.set_ylabel(b); ax.legend(title=target)
        ax.set_title(f"{a} vs {b}")
        _save(fig, out, "scatter.png", "Scatter plot of the top numeric features",
              "The two numeric features the model relies on most, coloured by outcome.", eda)
    elif num and task == "regression":
        show = imp_rank[:3]
        fig, axes = _grid(len(show))
        for ax, c in zip(axes, show):
            sns.regplot(x=sample[c], y=sample[target], ax=ax, scatter_kws={"s": 12, "alpha": .4, "color": TEAL},
                        line_kws={"color": AMBER})
            ax.set_title(f"{c} vs {target}")
        _save(fig, out, "scatter.png", "Scatter plots against the target",
              "Top numeric features with a fitted trend line.", eda)

    # correlation heatmap
    hm = df[num].copy() if num else pd.DataFrame(index=df.index)
    hm[target if task == "regression" else f"{target} (encoded)"] = (
        df[target] if task == "regression" else df[target].astype("category").cat.codes)
    if hm.shape[1] > 1:
        fig, ax = plt.subplots(figsize=(min(11, 1.0 * hm.shape[1] + 3), min(8, 0.8 * hm.shape[1] + 2.5)))
        sns.heatmap(hm.corr(), annot=True, fmt=".2f", cmap="BrBG", center=0, ax=ax,
                    linewidths=.5, cbar_kws={"shrink": .8}, annot_kws={"size": 9})
        ax.tick_params(axis="x", rotation=35)
        _save(fig, out, "heatmap.png", "Correlation heatmap",
              "Pearson correlation between numeric features and the target. Values near ±1 mean a strong linear link.", eda)

    # --- evaluation charts ---
    imp = model["importance"][:12][::-1]
    fig, ax = plt.subplots(figsize=(6.5, max(2.5, 0.42 * len(imp) + 1)))
    ax.barh([d["feature"] for d in imp], [d["importance"] for d in imp], color=TEAL)
    ax.set_xlabel("Drop in score when the feature is shuffled")
    _save(fig, out, "importance.png", "Feature importance",
          "Permutation importance on held-out data: how much accuracy is lost if the feature is scrambled.", evalc)

    if task == "classification":
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        sns.heatmap(np.array(model["confusion_matrix"]), annot=True, fmt="d", cmap="GnBu", cbar=False,
                    xticklabels=model["labels"], yticklabels=model["labels"], ax=ax,
                    annot_kws={"size": 14, "weight": "bold"})
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        _save(fig, out, "confusion.png", "Confusion matrix",
              "Rows are the true class, columns are what the model predicted, on the 20% test split.", evalc)
    else:
        y_te, pred = model["_y_test"], model["_pred"]
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        ax.scatter(y_te, pred, s=14, alpha=.5, color=TEAL, edgecolor="none")
        lim = [min(y_te.min(), pred.min()), max(y_te.max(), pred.max())]
        ax.plot(lim, lim, color=AMBER, lw=2)
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
        _save(fig, out, "actual_vs_pred.png", "Actual vs predicted",
              "Points on the amber line are perfect predictions.", evalc)
    return {"eda": eda, "evaluation": evalc}


# ----------------------------------------------------------------------------
# 6. Orchestration
# ----------------------------------------------------------------------------
def _py(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _form_spec(df: pd.DataFrame, features: list[str]) -> list[dict]:
    """Describe each input so the web form can be generated automatically."""
    spec = []
    for c in features:
        s = df[c]
        if is_numeric(s) and s.nunique() > 6:
            is_int = bool((s.dropna() % 1 == 0).all())
            cast = (lambda v: int(round(float(v)))) if is_int else (lambda v: round(float(v), 2))
            spec.append({"name": c, "kind": "number", "numeric": True, "min": cast(s.min()),
                         "max": cast(s.max()), "default": cast(s.median()), "integer": is_int})
        elif is_numeric(s):
            opts = [int(v) if float(v).is_integer() else float(v) for v in sorted(s.unique())]
            spec.append({"name": c, "kind": "select", "numeric": True, "options": opts,
                         "default": int(s.mode().iloc[0]) if float(s.mode().iloc[0]).is_integer() else float(s.mode().iloc[0])})
        else:
            vc = s.astype(str).value_counts()
            spec.append({"name": c, "kind": "select", "numeric": False, "options": sorted(vc.index.tolist()),
                         "default": vc.index[0]})
    return spec


def run_pipeline(df_raw: pd.DataFrame, target: str, out_dir: str | Path) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if target not in df_raw.columns:
        raise ValueError(f"Target column '{target}' not found.")

    df, clean_rep, bounds = clean_data(df_raw, target)
    if len(df) < 20:
        raise ValueError("Fewer than 20 usable rows remain after cleaning.")
    task = detect_task(df[target])
    if task == "classification":
        df[target] = df[target].astype(str)
        if df[target].nunique() < 2:
            raise ValueError("The target column has only one class - nothing to predict.")

    features, fs_rep = select_features(df, target, task)
    if not features:
        raise ValueError("No usable feature columns left after cleaning.")
    X, y = df[features], df[target]
    model = build_model(X, y, task)
    charts = make_charts(df_raw, df, target, task, features, model, out)

    form = _form_spec(df, features)
    joblib.dump({"pipeline": model["_pipeline"], "features": features, "bounds": bounds,
                 "task": task, "target": target, "form": form}, out / "model.joblib")

    result = {
        "target": target, "task": task, "features": features,
        "clean": clean_rep, "feature_selection": fs_rep,
        "model": {k: v for k, v in model.items() if not k.startswith("_")},
        "charts": charts, "form": form,
        "clean_preview_html": df.head(10).to_html(classes="grid", index=False, border=0),
        "clean_missing": [{"column": c, "missing": int(df[c].isna().sum())} for c in df.columns],
        "describe_html": df.describe().round(2).to_html(classes="grid", border=0),
    }
    (out / "result.json").write_text(json.dumps(result, default=_py, indent=2))
    return result


def predict_one(bundle: dict, values: dict):
    """Real-time prediction for one record entered in the web form."""
    row = {}
    for f in bundle["form"]:
        name, raw = f["name"], values.get(f["name"])
        if raw in (None, ""):
            raise ValueError(f"Please provide a value for '{name}'.")
        if f["numeric"]:
            try:
                v = float(raw)
            except ValueError:
                raise ValueError(f"'{name}' must be a number.")
            if name in bundle["bounds"]:                       # same outlier treatment as training
                lo, hi = bundle["bounds"][name]
                v = min(max(v, lo), hi)
            row[name] = v
        else:
            row[name] = str(raw)
    X = pd.DataFrame([row], columns=bundle["features"])
    pipe = bundle["pipeline"]
    pred = pipe.predict(X)[0]
    out = {"task": bundle["task"], "target": bundle["target"]}
    if bundle["task"] == "classification":
        out["prediction"] = str(pred)
        if hasattr(pipe, "predict_proba"):
            out["probabilities"] = {str(c): round(float(p), 4) for c, p in zip(pipe.classes_, pipe.predict_proba(X)[0])}
    else:
        out["prediction"] = round(float(pred), 2)
    return out
