from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
from typing import Optional

import json

import numpy as np
np.random.seed(42)

import pandas as pd

from fastapi.responses import JSONResponse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO

from sklearn.model_selection import (
    cross_val_score,
    StratifiedKFold,
    KFold,
    train_test_split,
    RandomizedSearchCV,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.metrics import make_scorer, accuracy_score, r2_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.utils.class_weight import compute_class_weight
from scipy.stats import randint, uniform

from PIL import Image, ImageStat, ImageFilter
import pytesseract
import chardet
from transformers import pipeline

import textstat
import spacy
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
import nltk

from collections import Counter
from typing import Dict, List, Any


import spacy
import textstat
import nltk
from collections import Counter
from typing import Dict, List, Any


import re
import unicodedata
from docx import Document

from io import BytesIO
from io import StringIO
from sklearn.cluster import KMeans

MODEL_MAP = {
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "logistic_regression": LogisticRegression,
    "svm": SVC,
    "gradient_boosting": GradientBoostingClassifier,
    "knn": KNeighborsClassifier,
}

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



try:
    nlp = spacy.load("en_core_web_sm")
except:
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')




@app.get("/")
async def root():
    return {"message": "Server is running"}

@app.get("/test")
async def test():
    return {"status": "ok", "message": "Test endpoint working"}


def extract_docx_text(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name
    doc = Document(temp_path)
    text = "\n".join([p.text for p in doc.paragraphs])
    os.unlink(temp_path)
    return text

def clean_text_for_display(text):
    text = unicodedata.normalize('NFKC', text)
    return re.sub(r'[^\x20-\x7E\n\t]', '', text)


def fill_missing_safely(df: pd.DataFrame) -> pd.DataFrame:
    """Fill numeric with mean, non-numeric with mode (if present)."""
    df = df.copy()
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) > 0:
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    cat_cols = df.columns.difference(num_cols)
    for c in cat_cols:
        if df[c].isna().any():
            mode_vals = df[c].mode(dropna=True)
            if len(mode_vals) > 0:
                df[c] = df[c].fillna(mode_vals.iloc[0])
            else:
                df[c] = df[c].fillna("missing")
    return df

def zscore_mask(df_numeric: pd.DataFrame, threshold: float = 3.0) -> pd.Series:
    """Return boolean mask for rows that are NOT outliers by Z-score across numeric columns."""
    if df_numeric.shape[1] == 0:
        return pd.Series(True, index=df_numeric.index)
    mask = pd.Series(True, index=df_numeric.index)
    for col in df_numeric.columns:
        s = df_numeric[col]
        std = s.std(ddof=0)
        if std == 0 or pd.isna(std):
            continue
        z = (s - s.mean()) / std
        col_mask = z.abs() < threshold
        col_mask = col_mask.fillna(True)
        mask &= col_mask
    return mask

def iqr_mask(df_numeric: pd.DataFrame, k: float = 1.5) -> pd.Series:
    """Return boolean mask for rows that are NOT outliers by IQR across numeric columns."""
    if df_numeric.shape[1] == 0:
        return pd.Series(True, index=df_numeric.index)
    mask = pd.Series(True, index=df_numeric.index)
    for col in df_numeric.columns:
        s = df_numeric[col]
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        lower = q1 - k * iqr
        upper = q3 + k * iqr
        col_mask = (s >= lower) & (s <= upper)
        col_mask = col_mask.fillna(True)
        mask &= col_mask
    return mask


@app.post("/upload/")
async def upload_dataset(
    file: UploadFile = File(...),
    model_type: Optional[str] = Form(None),
    target_columns: Optional[str] = Form(None),
    target_column: Optional[str] = Form(None),
    model_params: Optional[str] = Form(None),
    handle_missing: str = Form("drop"),
    handle_outliers: str = Form("none"),
):
    """
    Robust training endpoint:
     - supports single or multiple target columns
     - auto-detects classification vs regression
     - StratifiedKFold for single-output classification
     - sensible defaults and class-weight handling
     - fallback to train/test if CV fails
    """
    import tempfile, os, json

    if model_type is None:
        return JSONResponse({"error": "Missing required form field: model_type"}, status_code=400)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_path = temp_file.name

    try:
        df = pd.read_csv(temp_path)
    except Exception as e:
        try: os.unlink(temp_path)
        except: pass
        return JSONResponse({"error": f"Failed to read CSV: {e}"}, status_code=400)

    selected_targets = []
    if target_columns:
        try:
            parsed = json.loads(target_columns)
            if isinstance(parsed, list):
                selected_targets = parsed
            elif isinstance(parsed, str):
                selected_targets = [parsed]
        except Exception:
            selected_targets = [target_columns]
    elif target_column:
        selected_targets = [target_column]

    if not selected_targets:
        try: os.unlink(temp_path)
        except: pass
        return JSONResponse({"error": "No target column(s) specified. Provide `target_columns` (JSON array) or `target_column`."}, status_code=400)

    for t in selected_targets:
        if t not in df.columns:
            try: os.unlink(temp_path)
            except: pass
            return JSONResponse({"error": f"Target column '{t}' not found in CSV."}, status_code=400)

    rows_before, cols = df.shape

    if handle_missing == "drop":
        df = df.dropna()
    elif handle_missing == "mean":
        df = fill_missing_safely(df)
    elif handle_missing == "none":
        pass
    else:
        try: os.unlink(temp_path)
        except: pass
        return JSONResponse({"error": f"Invalid handle_missing option: {handle_missing}"}, status_code=400)

    if df.empty:
        try: os.unlink(temp_path)
        except: pass
        return JSONResponse({"error": "After cleaning, dataset is empty."}, status_code=400)

    rows_after = len(df)
    rows_removed = rows_before - rows_after
    if rows_after < 2:
        try: os.unlink(temp_path)
        except: pass
        return JSONResponse({"error": "Not enough rows after cleaning to train a model (need at least 2)."}, status_code=400)

    X = df.drop(columns=selected_targets, errors="ignore")
    if len(selected_targets) > 1:
        y = df[selected_targets].copy()
    else:
        y = df[selected_targets[0]].copy()

    if model_params:
        try:
            params = json.loads(model_params)
            for k, v in list(params.items()):
                if isinstance(v, str):
                    if v.isdigit():
                        params[k] = int(v)
                    else:
                        try:
                            params[k] = float(v)
                        except Exception:
                            pass
        except Exception as e:
            try: os.unlink(temp_path)
            except: pass
            return JSONResponse({"error": f"Failed to parse model_params: {e}"}, status_code=400)
    else:
        params = {}

    def _is_numeric_series(s: pd.Series) -> bool:
        return pd.api.types.is_numeric_dtype(s)

    multi_output = isinstance(y, pd.DataFrame) and y.shape[1] > 1
    

    if multi_output:
        y_encoded = pd.DataFrame()
        for col in y.columns:
            if _is_numeric_series(y[col]):
   
                try:
                    y_encoded[col] = pd.qcut(y[col], q=5, labels=False, duplicates='drop')
                except Exception:
                   
                    y_encoded[col] = pd.cut(y[col], bins=5, labels=False)
            else:
                
                y_encoded[col] = pd.factorize(y[col])[0]
        y = y_encoded
        problem_type = "multioutput_classification"
    else:
 
        if _is_numeric_series(y) and (y.nunique() > 20 or y.dtype.kind == "f"):
            problem_type = "regression"
        else:
            problem_type = "classification"

    REGRESSOR_MAP = {
        "decision_tree": DecisionTreeRegressor,
        "random_forest": RandomForestRegressor,
        "gradient_boosting": GradientBoostingRegressor,
        "logistic_regression": LinearRegression,
        "svm": SVR,
        "knn": KNeighborsRegressor,
    }

    if problem_type.startswith("regress"):
        ModelClass = REGRESSOR_MAP.get(model_type)
        if ModelClass is None:
            try: os.unlink(temp_path)
            except: pass
            return JSONResponse({"error": f"Model '{model_type}' not supported for regression."}, status_code=400)
    else:
        ModelClass = MODEL_MAP.get(model_type)
        if ModelClass is None:
            try: os.unlink(temp_path)
            except: pass
            return JSONResponse({"error": f"Invalid model type '{model_type}'."}, status_code=400)

    valid_params = {
        'decision_tree': ['max_depth', 'min_samples_split', 'class_weight'],
        'random_forest': ['n_estimators', 'max_depth', 'min_samples_split', 'class_weight'],
        'logistic_regression': ['C', 'max_iter', 'class_weight'],
        'svm': ['C', 'kernel', 'class_weight'],
        'gradient_boosting': ['n_estimators', 'learning_rate', 'max_depth', 'min_samples_split'],
        'knn': ['n_neighbors', 'weights']
    }
    filtered_params = {k: v for k, v in params.items() if k in valid_params.get(model_type, [])}
    if model_type == 'random_forest' and 'n_estimators' not in filtered_params:
        filtered_params.setdefault('n_estimators', 200)
    if model_type == 'gradient_boosting' and 'n_estimators' not in filtered_params:
        filtered_params.setdefault('n_estimators', 150)

    is_single_classification = (problem_type == "classification" and not multi_output)
    if is_single_classification and model_type in ['decision_tree', 'random_forest', 'logistic_regression', 'svm']:
        try:
            vc = y.value_counts()
            if len(vc) >= 2:
                ratio = float(vc.max()) / max(1.0, float(vc.min()))
                if ratio > 2.0:
                    filtered_params.setdefault('class_weight', 'balanced')
        except Exception:
            pass

    try:
        base_model = ModelClass(**filtered_params)
    except Exception as e:
        try: os.unlink(temp_path)
        except: pass
        return JSONResponse({"error": f"Error initializing model: {e}"}, status_code=400)

    if problem_type in ("multioutput_classification", "classification"):
        final_estimator = MultiOutputClassifier(base_model) if multi_output else base_model
    else:
        final_estimator = MultiOutputRegressor(base_model) if multi_output else base_model

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if X[c].dtype == 'object' or X[c].dtype.name == 'category']
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', KNNImputer() if params.get('imputation', '') == 'knn' else SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ]), num_cols if num_cols else []),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), cat_cols if cat_cols else [])
        ],
        remainder='drop'
    )


    selector = None
    try:
        if X.shape[1] > 50 and problem_type.startswith("class"):
            k = min(50, X.shape[1])
            selector = SelectKBest(score_func=f_classif, k=k)
    except Exception:
        selector = None

    steps = [('preprocessor', preprocessor)]
    if selector is not None:
        steps.append(('selector', selector))
    steps.append(('estimator', final_estimator))
    pipe = Pipeline(steps)


    if problem_type.startswith("class"):
        def multioutput_accuracy(y_true, y_pred):
            try:
                y_true_arr = np.array(y_true)
                y_pred_arr = np.array(y_pred)
                if y_true_arr.ndim == 1 or y_pred_arr.ndim == 1:
                    return accuracy_score(y_true_arr, y_pred_arr)
                scores = []
                for i in range(y_true_arr.shape[1]):
                    scores.append(accuracy_score(y_true_arr[:, i], y_pred_arr[:, i]))
                return float(np.mean(scores))
            except Exception:
                return 0.0
        scorer = make_scorer(multioutput_accuracy)
    else:
        def multioutput_r2(y_true, y_pred):
            try:
                return float(r2_score(y_true, y_pred, multioutput='uniform_average'))
            except Exception:
                return float(-np.mean((np.array(y_true) - np.array(y_pred))**2))
        scorer = make_scorer(multioutput_r2)


    cv = min(5, rows_after)
    if cv < 2:
        try: os.unlink(temp_path)
        except: pass
        return JSONResponse({"error": "Not enough samples to perform cross-validation."}, status_code=400)

    if is_single_classification:
        try:
            cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        except Exception:
            cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=42)
    else:
        cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=42)

    try:
        if params.get('auto_tune', False) and not multi_output:
            param_distributions = {}
            if model_type in ['random_forest', 'gradient_boosting']:
                param_distributions = {
                    'estimator__n_estimators': randint(50, 400),
                    'estimator__max_depth': randint(3, 20),
                    'estimator__min_samples_split': randint(2, 20),
                }
            elif model_type == 'knn':
                param_distributions = {
                    'estimator__n_neighbors': randint(3, 30),
                    'estimator__weights': ['uniform', 'distance']
                }
            elif model_type == 'svm':
                param_distributions = {
                    'estimator__C': uniform(0.1, 10),
                    'estimator__kernel': ['linear', 'rbf', 'poly']
                }

            if param_distributions:
                search = RandomizedSearchCV(pipe, param_distributions, n_iter=25, cv=cv_strategy, scoring=scorer, n_jobs=-1, random_state=42, error_score='raise')
                search.fit(X, y)
                best_model = search.best_estimator_
                scores = cross_val_score(best_model, X, y, cv=cv_strategy, scoring=scorer, n_jobs=-1)
            else:
                scores = cross_val_score(pipe, X, y, cv=cv_strategy, scoring=scorer, n_jobs=-1)
        else:
            scores = cross_val_score(pipe, X, y, cv=cv_strategy, scoring=scorer, n_jobs=-1)
    except Exception as e:
        try:
            stratify = y if is_single_classification else None
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            if problem_type.startswith("class"):
                if isinstance(y_test, pd.DataFrame):
                    arr_true = np.array(y_test)
                    arr_pred = np.array(y_pred)
                    accs = [accuracy_score(arr_true[:, i], arr_pred[:, i]) for i in range(arr_true.shape[1])]
                    single_score = float(np.mean(accs))
                else:
                    single_score = float(accuracy_score(y_test, y_pred))
            else:
                single_score = float(r2_score(y_test, y_pred, multioutput='uniform_average'))
            scores = np.array([single_score])
        except Exception as ee:
            try: os.unlink(temp_path)
            except: pass
            return JSONResponse({"error": f"Training failed during cross-validation: {str(e)}; fallback error: {str(ee)}"}, status_code=500)

    try: os.unlink(temp_path)
    except: pass

    if problem_type.startswith("class"):
        accuracy = round(float(np.mean(scores)) * 100, 2)
        std_dev = round(float(np.std(scores)) * 100, 2)
    else:
        accuracy = round(float(np.mean(scores)) * 100, 2)
        std_dev = round(float(np.std(scores)) * 100, 2)

    return JSONResponse({
        "accuracy": accuracy,
        "std_dev": std_dev,
        "model_used": model_type,
        "params_used": params,
        "filename": file.filename,
        "file_size_kb": round(os.path.getsize(temp_path) / 1024.0, 2) if os.path.exists(temp_path) else None,
        "rows_before": int(rows_before),
        "rows_after": int(rows_after),
        "rows_removed": int(rows_removed),
        "columns": int(cols),
        "handle_missing": handle_missing,
        "handle_outliers": handle_outliers,
        "target_columns": selected_targets,
        "problem_type": problem_type
    }, status_code=200)


@app.post("/analyze-csv/")
async def analyze_csv(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        cols = []
        for col in df.columns:
            dtype = df[col].dtype
            if pd.api.types.is_numeric_dtype(dtype):
                col_type = "numerical"
            else:
                col_type = "categorical"
            cols.append({"name": col, "type": col_type})
        return {"columns": cols}
    except Exception as e:
        return {"error": f"Failed to analyze CSV: {str(e)}"}


@app.post("/eda/")
async def eda(
    file: UploadFile = File(...),
    columns: str = Form(...),
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_path = temp_file.name

    try:
        df = pd.read_csv(temp_path)
    except Exception as e:
        try: os.unlink(temp_path)
        except: pass
        return {"error": f"Failed to read CSV: {e}"}

    try:
        columns_info = json.loads(columns)
    except Exception as e:
        try: os.unlink(temp_path)
        except: pass
        return {"error": f"Failed to parse columns: {e}"}

    if not columns_info or not isinstance(columns_info, list):
        try: os.unlink(temp_path)
        except: pass
        return {"error": "Invalid columns info."}

    col_names = [c["name"] for c in columns_info]
    col_types = [c["type"] for c in columns_info]
    if not all(name in df.columns for name in col_names):
        try: os.unlink(temp_path)
        except: pass
        return {"error": "Some columns not found in dataset."}

    total_rows = len(df)
    total_columns = len(df.columns)
    missing_values = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    MAX_PLOT_ROWS = 1000
    try:
        if total_rows > MAX_PLOT_ROWS:
            plot_df = df.sample(n=MAX_PLOT_ROWS, random_state=42)
            sampled_note = f" (sampled {MAX_PLOT_ROWS} of {total_rows} rows for plotting)"
        else:
            plot_df = df.copy()
            sampled_note = ""
    except Exception:
        plot_df = df.copy()
        sampled_note = ""

    plt.clf()
    img_b64 = None

    try:
        if len(col_names) == 2:
            fig, ax = plt.subplots(figsize=(6, 4))
            t1, t2 = col_types
            c1, c2 = col_names

            if t1 == "numerical" and t2 == "numerical":
                sns.scatterplot(data=plot_df, x=c1, y=c2, ax=ax)
                ax.set_title(f"Scatterplot: {c1} vs {c2}{sampled_note}")
            elif t1 == "categorical" and t2 == "numerical":
                top_n = 30
                vals = plot_df[c1].value_counts().nlargest(top_n).index
                sns.boxplot(data=plot_df[plot_df[c1].isin(vals)], x=c1, y=c2, ax=ax)
                ax.set_title(f"Boxplot: {c2} by {c1}{sampled_note}")
            elif t1 == "numerical" and t2 == "categorical":
                top_n = 30
                vals = plot_df[c2].value_counts().nlargest(top_n).index
                sns.boxplot(data=plot_df[plot_df[c2].isin(vals)], x=c2, y=c1, ax=ax)
                ax.set_title(f"Boxplot: {c1} by {c2}{sampled_note}")
            else:
                ct = pd.crosstab(plot_df[c1], plot_df[c2])
                ct = ct.iloc[:50, :50]
                ct.plot(kind="bar", stacked=True, ax=ax)
                ax.set_title(f"Stacked Bar: {c1} vs {c2}{sampled_note}")

            plt.tight_layout()
            buf = BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode("utf-8")
            buf.close()
            plt.close(fig)

        elif len(col_names) == 1:
            fig, ax = plt.subplots(figsize=(6, 4))
            t1 = col_types[0]
            c1 = col_names[0]
            if t1 == "numerical":
                sns.histplot(plot_df[c1].dropna(), kde=True, ax=ax)
                ax.set_title(f"Histogram: {c1}{sampled_note}")
            else:
                top_n = 50
                vc = plot_df[c1].value_counts().nlargest(top_n)
                vc.plot(kind="bar", ax=ax)
                ax.set_title(f"Barplot: {c1}{sampled_note}")
            plt.tight_layout()
            buf = BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode("utf-8")
            buf.close()
            plt.close(fig)

        elif len(col_names) >= 3:
            num_cols = [c["name"] for c in columns_info if c["type"] == "numerical"]
            cat_cols = [c["name"] for c in columns_info if c["type"] == "categorical"]

            if len(num_cols) >= len(col_names) and len(num_cols) >= 2:
                MAX_PAIRVARS = 6
                pair_vars = num_cols
                if len(pair_vars) > MAX_PAIRVARS:
                    var_series = plot_df[num_cols].var().sort_values(ascending=False)
                    pair_vars = list(var_series.head(MAX_PAIRVARS).index)
                sns_plot = sns.pairplot(plot_df[pair_vars].dropna(), corner=True, diag_kind="kde")
                sns_plot.fig.suptitle(f"Pairplot of Numerical Columns{sampled_note}", y=1.02)
                buf = BytesIO()
                sns_plot.fig.savefig(buf, format="png")
                buf.seek(0)
                img_b64 = base64.b64encode(buf.read()).decode("utf-8")
                buf.close()
                plt.close('all')

            elif len(cat_cols) == len(col_names):
                limited_cols = []
                for c in cat_cols:
                    top_vals = plot_df[c].value_counts().nlargest(30).index
                    limited_cols.append(plot_df[plot_df[c].isin(top_vals)][c])
                if len(limited_cols) >= 2:
                    ct = pd.crosstab(limited_cols[0], limited_cols[1])
                else:
                    ct = pd.crosstab(plot_df[cat_cols[0]].value_counts().nlargest(30).index, plot_df[cat_cols[0]])
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(ct, annot=False, fmt="d", cmap="Blues", ax=ax)
                ax.set_title(f"Heatmap: {'/'.join(cat_cols[:2])}{sampled_note}")
                plt.tight_layout()
                buf = BytesIO()
                fig.savefig(buf, format="png")
                buf.seek(0)
                img_b64 = base64.b64encode(buf.read()).decode("utf-8")
                buf.close()
                plt.close(fig)

            else:
                if len(num_cols) >= 1 and len(cat_cols) >= 1:
                    MAX_MIX_NUM = 4
                    mix_num = num_cols[:MAX_MIX_NUM]
                    hue_col = cat_cols[0]
                    sns_plot = sns.pairplot(plot_df[mix_num + [hue_col]].dropna(), hue=hue_col, corner=True, diag_kind="kde")
                    sns_plot.fig.suptitle(f"Pairplot (hue={hue_col}){sampled_note}", y=1.02)
                    buf = BytesIO()
                    sns_plot.fig.savefig(buf, format="png")
                    buf.seek(0)
                    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
                    buf.close()
                    plt.close('all')
                else:
                    if num_cols:
                        corr = plot_df[num_cols].corr().fillna(0)
                        fig, ax = plt.subplots(figsize=(8,6))
                        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
                        ax.set_title(f"Correlation matrix{sampled_note}")
                        plt.tight_layout()
                        buf = BytesIO()
                        fig.savefig(buf, format="png")
                        buf.seek(0)
                        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
                        buf.close()
                        plt.close(fig)
                    else:
                        return {"error": "Unable to generate mixed plots for selected columns."}
        else:
            return {"error": "Please select at least 1 column for EDA."}
    except MemoryError as me:
        try: os.unlink(temp_path)
        except: pass
        return {"error": f"Plot generation failed due to memory constraints. Try using fewer columns or a smaller CSV (sample your data)."}
    except Exception as e:
        try: os.unlink(temp_path)
        except: pass
        return {"error": f"Failed to generate plot: {e}"}

    try: os.unlink(temp_path)
    except Exception:
        pass

    return JSONResponse(content={
        "image": img_b64,
        "totalRows": total_rows,
        "totalColumns": total_columns,
        "missingValues": missing_values,
        "duplicateRows": duplicate_rows
    })


@app.post("/process-text/")
async def process_text(file: UploadFile = File(...)):
    try:
        content = await file.read()
        filename = file.filename
        ext = os.path.splitext(filename)[1].lower()
        
        if ext == ".docx":
            text = extract_docx_text(content)
            encoding = 'docx'
        else:
            detection = chardet.detect(content)
            encoding = detection['encoding'] or 'utf-8'
            try:
                text = content.decode(encoding)
            except UnicodeDecodeError:
                for enc in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                    try:
                        text = content.decode(enc)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    return {"error": "Could not decode file with any supported encoding"}
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"[^\x20-\x7E\n\t]", "", text)
        text = re.sub(r'\s+', ' ', text).strip()
        word_count = len(text.split())
        line_count = len(text.splitlines())
        char_count = len(text)
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        sentence_count = len(sentences)

        try:
            sentiment_analyzer = pipeline("sentiment-analysis")
            sentiment = sentiment_analyzer(text[:512])[0]
        except Exception as e:
            sentiment = {"label": "N/A", "score": 0}

        return {
            "statistics": {
                "words": word_count,
                "lines": line_count,
                "characters": char_count,
                "sentences": sentence_count,
                "detected_encoding": encoding
            },
            "sentiment": sentiment,
            "preview": text[:200] + "..." if len(text) > 200 else text
        }
    except Exception as e:
        return {"error": f"Failed to process text: {str(e)}"}

@app.post("/analyze-readability/")
async def analyze_readability(file: UploadFile = File(...)):
    try:
        content = await file.read()
        filename = file.filename
        ext = os.path.splitext(filename)[1].lower()
        if ext == '.docx':
            text = extract_docx_text(content)
        else:
            try:
                text = content.decode('utf-8')
            except UnicodeDecodeError:
                text = content.decode('latin-1')

        text = clean_text_for_display(text)

        if not text.strip():
            return {"error": "Empty text content"}
        
        analysis = {
            "flesch_reading_ease": float(textstat.flesch_reading_ease(text)),
            "flesch_kincaid_grade": float(textstat.flesch_kincaid_grade(text)),
            "gunning_fog": float(textstat.gunning_fog(text)),
            "smog_index": float(textstat.smog_index(text)),
            "automated_readability_index": float(textstat.automated_readability_index(text)),
            "coleman_liau_index": float(textstat.coleman_liau_index(text)),
            "dale_chall_readability_score": float(textstat.dale_chall_readability_score(text)),
            "difficult_words": int(textstat.difficult_words(text)),
            "linsear_write_formula": float(textstat.linsear_write_formula(text)),
            "text_standard": str(textstat.text_standard(text))
        }
        
        return analysis
    except Exception as e:
        print(f"Readability analysis error: {str(e)}")  
        return {"error": f"Failed to analyze readability: {str(e)}"}

@app.post("/extract-topics/")
async def extract_topics(file: UploadFile = File(...)):
    try:
        content = await file.read()
        filename = file.filename
        ext = os.path.splitext(filename)[1].lower()
        if ext == '.docx':
            text = extract_docx_text(content)
        else:
            try:
                text = content.decode('utf-8')
            except UnicodeDecodeError:
                text = content.decode('latin-1')

        text = clean_text_for_display(text)

        if not text.strip():
            return {"error": "Empty text content"}
        doc = nlp(text)
        noun_phrases = [
            chunk.text for chunk in doc.noun_chunks 
            if len(chunk.text.split()) > 1
        ]
        keywords = [
            token.text.lower() for token in doc 
            if not token.is_stop and token.is_alpha and len(token.text) > 2
        ]
        word_freq = Counter(keywords)
        common_words = [
            [word, count] for word, count in word_freq.most_common(15)
        ]
        main_topics = list(set([
            ent.text for ent in doc.ents 
            if ent.label_ in ["ORG", "PRODUCT", "EVENT", "WORK_OF_ART", "TOPIC"]
            and len(ent.text) > 2
        ]))
        
        return {
            "noun_phrases": list(set(noun_phrases))[:15],
            "common_words": common_words,
            "main_topics": main_topics[:10]
        }
    except Exception as e:
        print(f"Topic extraction error: {str(e)}")
        return {"error": f"Failed to extract topics: {str(e)}"}

@app.post("/extract-entities/")
async def extract_entities(file: UploadFile = File(...)):
    try:
        content = await file.read()
        filename = file.filename
        ext = os.path.splitext(filename)[1].lower()
        if ext == '.docx':
            text = extract_docx_text(content)
        else:
            try:
                text = content.decode('utf-8')
            except UnicodeDecodeError:
                text = content.decode('latin-1')

        text = clean_text_for_display(text)

        if not text.strip():
            return {"error": "Empty text content"}
        doc = nlp(text)
        entity_categories = {
            "PERSON": set(),
            "ORG": set(),
            "GPE": set(),
            "DATE": set(),
            "TIME": set(),
            "MONEY": set(),
            "PERCENT": set()
        }
        
      
        for ent in doc.ents:
            if ent.label_ in entity_categories:

                clean_ent = ent.text.strip()
                if clean_ent and len(clean_ent) > 1:
                    entity_categories[ent.label_].add(clean_ent)
        return {
            category: list(entities)[:10]
            for category, entities in entity_categories.items()
        }
    except Exception as e:
        print(f"Entity extraction error: {str(e)}") 
        return {"error": f"Failed to extract entities: {str(e)}"}




def _img_to_base64(img: Image.Image, fmt="PNG") -> str:
    buf = BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def _plot_histogram_as_base64(hist_vals, title="Color Histogram"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6,3))
    ax.bar(range(len(hist_vals)), hist_vals)
    ax.set_title(title)
    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_b64

def _rgb_to_hex(rgb_tuple):
    return '#{:02x}{:02x}{:02x}'.format(int(rgb_tuple[0]), int(rgb_tuple[1]), int(rgb_tuple[2]))



@app.post("/process-image/")
async def process_image(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1] or ".png") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        img = Image.open(tmp_path)
        width, height = img.size
        fmt = img.format or "PNG"
        mode = img.mode

        extracted_text = ""
        try:
            ocr_img = img.convert("RGB")
            extracted_text = pytesseract.image_to_string(ocr_img).strip()
        except Exception:
            extracted_text = ""

        try:
            os.unlink(tmp_path)
        except Exception:
            pass

        return {
            "dimensions": {"width": width, "height": height},
            "format": fmt,
            "mode": mode,
            "extracted_text": extracted_text or ""
        }

    except Exception as e:
        return {"error": f"Failed to process image: {str(e)}"}

@app.post("/extract-image-features/")
async def extract_image_features(file: UploadFile = File(...), k_colors: int = Form(3)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1] or ".png") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        img = Image.open(tmp_path).convert("RGB")
        width, height = img.size

        stat = ImageStat.Stat(img.convert("L"))
        avg_brightness = float(stat.mean[0])
        contrast = float(stat.stddev[0])

        arr = np.array(img)
        pixels = arr.reshape(-1, 3).astype(np.float32)
        max_samples = 30000
        sample_pixels = pixels[np.random.choice(pixels.shape[0], min(pixels.shape[0], max_samples), replace=False)]

        try:
            km = KMeans(n_clusters=max(1, int(k_colors)), random_state=42)
            km.fit(sample_pixels)
            centers = km.cluster_centers_.astype(int)
            dominant_hex = [_rgb_to_hex(tuple(center)) for center in centers]
        except Exception:
            avg_col = tuple(np.mean(sample_pixels, axis=0).astype(int).tolist())
            dominant_hex = [_rgb_to_hex(avg_col)]

        hist_r, _ = np.histogram(arr[:,:,0].flatten(), bins=32, range=(0,255))
        hist_g, _ = np.histogram(arr[:,:,1].flatten(), bins=32, range=(0,255))
        hist_b, _ = np.histogram(arr[:,:,2].flatten(), bins=32, range=(0,255))
        hist_sum = (hist_r + hist_g + hist_b).tolist()
        histogram_base64 = _plot_histogram_as_base64(hist_sum, title="Color Intensity Histogram")

        edges = img.convert("L").filter(ImageFilter.FIND_EDGES)
        edge_b64 = _img_to_base64(edges, fmt="PNG")

        palette_img = Image.new("RGB", (len(dominant_hex)*40, 40))
        for i, hx in enumerate(dominant_hex):
            color = tuple(int(hx.lstrip("#")[j:j+2], 16) for j in (0,2,4))
            sw = Image.new("RGB", (40,40), color)
            palette_img.paste(sw, (i*40, 0))
        palette_b64 = _img_to_base64(palette_img, fmt="PNG")

        try:
            os.unlink(tmp_path)
        except Exception:
            pass

        return {
            "dominant_colors": dominant_hex,
            "palette_image": palette_b64,
            "histogram_image": histogram_base64,
            "avg_brightness": round(avg_brightness, 2),
            "contrast": round(contrast, 2),
            "edge_image": edge_b64,
            "width": width,
            "height": height,
            "format": img.format or "PNG"
        }
    except Exception as e:
        return {"error": f"Failed to extract features: {str(e)}"}

@app.post("/explain-image/")
async def explain_image(file: UploadFile = File(...), explanation_level: str = Form("short")):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1] or ".png") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        img = Image.open(tmp_path).convert("RGB")

        gray = img.convert("L")
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edges_arr = np.array(edges).astype(np.float32)
        edges_norm = (edges_arr / edges_arr.max()) * 255.0 if edges_arr.max() > 0 else edges_arr
        edges_vis = np.stack([edges_norm, edges_norm, edges_norm], axis=-1).astype(np.uint8)
        saliency_img = Image.fromarray(edges_vis)
        saliency_b64 = _img_to_base64(saliency_img, fmt="PNG")

        steps = [
            "1) Input image is loaded and converted to RGB.",
            "2) Basic pixel-level statistics are computed (brightness / contrast).",
            "3) Edge detection (simple FIND_EDGES) highlights high-frequency regions (text, boundaries).",
            "4) Color clustering (KMeans) identifies dominant palette colors.",
            "5) OCR (if requested previously) extracts embedded text via Tesseract.",
            "6) Results are packaged into human-readable features and visual maps."
        ]
        if explanation_level == "short":
            steps = steps[:3]

        stat = ImageStat.Stat(img.convert("L"))
        avg_brightness = float(stat.mean[0])
        contrast = float(stat.stddev[0])
        feature_summary = {
            "avg_brightness": round(avg_brightness, 2),
            "contrast": round(contrast, 2),
            "size": {"width": img.width, "height": img.height},
            "mode": img.mode,
        }

        try:
            os.unlink(tmp_path)
        except Exception:
            pass

        return {
            "explanation_steps": steps,
            "saliency_image": saliency_b64,
            "feature_summary": feature_summary
        }

    except Exception as e:
        return {"error": f"Failed to generate explanation: {str(e)}"}



def _detect_domains_from_features(features: dict, ocr_text: str):
    """
    Heuristic domain detection:
    - If OCR text length > threshold -> 'document'
    - If greens dominate -> 'nature'
    - If high contrast and compact palette -> 'product'
    - If portrait ratio -> 'portrait'
    - fallback -> 'general'
    """
    domains = []
    txt_len = len(ocr_text.strip()) if ocr_text else 0

    if txt_len > 20:
        domains.append("document")

    dom_colors = [c.lstrip("#") for c in features.get("dominant_colors", [])]
    hex_vals = []
    for h in dom_colors:
        try:
            r = int(h[0:2], 16); g = int(h[2:4], 16); b = int(h[4:6], 16)
            hex_vals.append((r,g,b))
        except Exception:
            continue

    greens = sum(1 for (r,g,b) in hex_vals if g > r and g > b and g > 100)
    if greens >= 1 and "nature" not in domains:
        domains.append("nature")

    if features.get("contrast", 0) > 30 and len(hex_vals) <= 4 and "product" not in domains:
        domains.append("product")

    size = features.get("size") or {"width": features.get("width"), "height": features.get("height")}
    try:
        w = int(size.get("width") or 0); h = int(size.get("height") or 0)
        if h > 1.2 * w and "portrait" not in domains:
            domains.append("portrait")
    except Exception:
        pass

    ordered = []
    for d in domains:
        if d not in ordered:
            ordered.append(d)
    if not ordered:
        ordered = ["general"]
    return ordered

@app.post("/advanced-analysis/")
async def advanced_analysis(file: UploadFile = File(...), k_colors: int = Form(4)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1] or ".png") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        img = Image.open(tmp_path).convert("RGB")
        width, height = img.size
        stat = ImageStat.Stat(img.convert("L"))
        avg_brightness = float(stat.mean[0])
        contrast = float(stat.stddev[0])

        arr = np.array(img)
        pixels = arr.reshape(-1,3).astype(np.float32)
        max_samples = 30000
        sample_pixels = pixels[np.random.choice(pixels.shape[0], min(pixels.shape[0], max_samples), replace=False)]

        try:
            km = KMeans(n_clusters=max(1, int(k_colors)), random_state=42)
            km.fit(sample_pixels)
            centers = km.cluster_centers_.astype(int)
            dominant_hex = ['#%02x%02x%02x' % tuple(c) for c in centers]
        except Exception:
            avg_col = tuple(np.mean(sample_pixels, axis=0).astype(int).tolist())
            dominant_hex = ['#%02x%02x%02x' % tuple(avg_col)]

        ocr_text = ""
        try:
            ocr_text = pytesseract.image_to_string(img.convert("RGB")).strip()
        except Exception:
            ocr_text = ""

        feature_summary = {
            "avg_brightness": round(avg_brightness,2),
            "contrast": round(contrast,2),
            "size": {"width": width, "height": height},
            "mode": img.mode
        }

        gray = img.convert("L")
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edges_arr = np.array(edges).astype(np.float32)
        edges_norm = (edges_arr / edges_arr.max()) * 255.0 if edges_arr.max()>0 else edges_arr
        saliency_img = Image.fromarray(np.stack([edges_norm, edges_norm, edges_norm], axis=-1).astype(np.uint8))
        saliency_b64 = _img_to_base64(saliency_img)

        feat_obj = {
            "dominant_colors": dominant_hex,
            "avg_brightness": round(avg_brightness,2),
            "contrast": round(contrast,2),
            "width": width,
            "height": height,
            "size": {"width": width, "height": height}
        }

        domains = _detect_domains_from_features(feat_obj, ocr_text)

        action_map = {
            "document": [
                {"id":"table_extract", "label":"Table / Layout Extraction", "description":"Try to extract tables, key-value fields and layout from document-like images.", "endpoint":"/specialized/table-extract/"},
                {"id":"document_ner", "label":"Document Entity Extraction", "description":"Run NER and structured parsing tailored to invoices/receipts.", "endpoint":"/specialized/document-ner/"}
            ],
            "product": [
                {"id":"logo_detect", "label":"Logo / Branding Detection", "description":"Detect logos, brand names and packaging text.", "endpoint":"/specialized/logo-detect/"},
                {"id":"color_palette", "label":"Packaging Color Analysis", "description":"Detailed color breakdown for packaging design.", "endpoint":"/specialized/packaging-analysis/"}
            ],
            "nature": [
                {"id":"plant_id", "label":"Plant/Species Detector (placeholder)", "description":"Attempt to identify plant/flower types using color/shape heuristics or models.", "endpoint":"/specialized/plant-id/"},
            ],
            "portrait": [
                {"id":"face_attributes", "label":"Face Attributes (age/gender)", "description":"Estimate facial attributes (placeholder).", "endpoint":"/specialized/face-attributes/"},
            ],
            "general": [
                {"id":"edge_stats", "label":"Edge / Texture Statistics", "description":"Deeper texture analysis and image quality metrics.", "endpoint":"/specialized/texture-analysis/"}
            ]
        }

        suggested = []
        seen = set()
        for d in domains:
            for a in action_map.get(d, action_map.get("general", [])):
                if a["id"] not in seen:
                    suggested.append(a)
                    seen.add(a["id"])

        try:
            os.unlink(tmp_path)
        except Exception:
            pass

        return {
            "features": feat_obj,
            "explanation": {
                "explanation_steps": [
                    "Loaded image and computed brightness/contrast.",
                    "Computed dominant colors via clustering.",
                    "Computed saliency via simple edge detection."
                ],
                "saliency_image": saliency_b64,
                "feature_summary": feature_summary
            },
            "ocr_text": ocr_text,
            "domain_suggestions": domains,
            "suggested_actions": suggested
        }

    except Exception as e:
        return {"error": f"advanced analysis failed: {str(e)}"}



@app.post("/specialized/table-extract/")
async def specialized_table_extract(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1] or ".png") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        img = Image.open(tmp_path)
        text = ""
        try:
            text = pytesseract.image_to_string(img).strip()
        except Exception:
            text = ""

        lines = [l for l in text.splitlines() if l.strip()]
        rows = []
        for ln in lines:
            if any(ch.isdigit() for ch in ln) or ('|' in ln) or (',' in ln and any(c.isdigit() for c in ln)):
                rows.append(ln.strip())

        try:
            os.unlink(tmp_path)
        except:
            pass

        return {"rows": rows, "raw_text": text}
    except Exception as e:
        return {"error": str(e)}

@app.post("/specialized/logo-detect/")
async def specialized_logo_detect(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1] or ".png") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        img = Image.open(tmp_path).convert("RGB")
        text = ""
        try:
            text = pytesseract.image_to_string(img).strip()
        except:
            text = ""
        arr = np.array(img)
        hist_r, _ = np.histogram(arr[:,:,0].flatten(), bins=8, range=(0,255))
        hist_g, _ = np.histogram(arr[:,:,1].flatten(), bins=8, range=(0,255))
        hist_b, _ = np.histogram(arr[:,:,2].flatten(), bins=8, range=(0,255))
        colors = {
            "r_hist": hist_r.tolist(),
            "g_hist": hist_g.tolist(),
            "b_hist": hist_b.tolist()
        }
        try:
            os.unlink(tmp_path)
        except:
            pass
        return {"ocr_text": text, "color_hist": colors}
    except Exception as e:
        return {"error": str(e)}

@app.post("/specialized/plant-id/")
async def specialized_plant_id(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1] or ".png") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        img = Image.open(tmp_path).convert("RGB")
        arr = np.array(img)
        avg = tuple(np.mean(arr.reshape(-1,3), axis=0).astype(int).tolist())
        try:
            os.unlink(tmp_path)
        except:
            pass
        return {"message": "Plant ID placeholder — result may be inaccurate", "avg_color": '#%02x%02x%02x' % tuple(avg)}
    except Exception as e:
        return {"error": str(e)}

@app.post("/specialized/face-attributes/")
async def specialized_face_attributes(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1] or ".png") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        img = Image.open(tmp_path)
        w,h = img.size
        try:
            os.unlink(tmp_path)
        except:
            pass
        return {"note":"Face attributes placeholder — use a face detector model for production", "width": w, "height": h}
    except Exception as e:
        return {"error": str(e)}

@app.post("/specialized/texture-analysis/")
async def specialized_texture_analysis(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1] or ".png") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        img = Image.open(tmp_path).convert("L")
        arr = np.array(img).astype(np.float32)
        mean = float(arr.mean())
        std = float(arr.std())
        hist, _ = np.histogram(arr.flatten(), bins=64, range=(0,255))
        probs = hist / (hist.sum() + 1e-9)
        entropy = -float((probs * np.log(probs + 1e-9)).sum())
        try:
            os.unlink(tmp_path)
        except:
            pass
        return {"mean": round(mean,2), "std": round(std,2), "entropy": round(entropy,3)}
    except Exception as e:
        return {"error": str(e)}
    
def _generate_advanced_features(df: pd.DataFrame, num_cols: list, cat_cols: list, target_col: Optional[str] = None) -> tuple:
    """
    Generate intelligent feature engineering suggestions and apply them.
    Returns (suggestions_list, engineered_df, feature_names)
    """
    suggestions = []
    df_engineered = df.copy()
    generated_features = []

    if len(num_cols) >= 2:
        top_two_num = sorted(num_cols, key=lambda c: df[c].std(), reverse=True)[:2]
        if len(top_two_num) == 2:
            try:
                col1, col2 = top_two_num
                interaction_name = f"{col1}_x_{col2}"
                df_engineered[interaction_name] = df[col1] * df[col2]
                generated_features.append(interaction_name)
                
                suggestions.append({
                    "name": interaction_name,
                    "type": "Interaction",
                    "description": f"Multiplicative interaction between {col1} and {col2}",
                    "confidence": 0.85,
                    "columns_involved": [col1, col2]
                })
                
                ratio_name = f"{col1}_div_{col2}"
                df_engineered[ratio_name] = df[col1] / (df[col2] + 1e-8)
                df_engineered[ratio_name] = df_engineered[ratio_name].replace([np.inf, -np.inf], 0).fillna(0)
                generated_features.append(ratio_name)
                
                suggestions.append({
                    "name": ratio_name,
                    "type": "Ratio",
                    "description": f"Ratio of {col1} to {col2} (useful for rate-based features)",
                    "confidence": 0.75,
                    "columns_involved": [col1, col2]
                })
            except Exception:
                pass

    for col in num_cols[:5]:  
        try:
            if (df[col] > 0).all():
                log_name = f"log_{col}"
                df_engineered[log_name] = np.log1p(df[col])
                generated_features.append(log_name)
                
                suggestions.append({
                    "name": log_name,
                    "type": "Transform",
                    "description": f"Natural log transformation of {col} (helps with skewed distributions)",
                    "confidence": 0.72,
                    "columns_involved": [col]
                })
        except Exception:
            pass

        try:
            squared_name = f"{col}_squared"
            df_engineered[squared_name] = df[col] ** 2
            generated_features.append(squared_name)
            
            suggestions.append({
                "name": squared_name,
                "type": "Polynomial",
                "description": f"Squared term of {col} (captures non-linear relationships)",
                "confidence": 0.70,
                "columns_involved": [col]
            })
        except Exception:
            pass

    for col in cat_cols[:3]:  
        try:
            n_unique = df[col].nunique()
            if 2 <= n_unique <= 10:
                suggestions.append({
                    "name": f"onehot_{col}",
                    "type": "Encoding",
                    "description": f"One-hot encode {col} ({n_unique} unique values) for ML models",
                    "confidence": 0.80,
                    "columns_involved": [col]
                })
        except Exception:
            pass

    for col in num_cols:
        try:
            if df[col].std() / (df[col].mean() + 1e-8) > 0.5 and df[col].nunique() > 20:
                bin_name = f"{col}_binned"
                suggestions.append({
                    "name": bin_name,
                    "type": "Binning",
                    "description": f"Discretize {col} into quantile bins (helps with tree models)",
                    "confidence": 0.65,
                    "columns_involved": [col]
                })
        except Exception:
            pass

    suggestions = sorted(suggestions, key=lambda x: x["confidence"], reverse=True)

    return suggestions, df_engineered, generated_features


@app.post("/analyze-features/")
async def analyze_features(
    file: UploadFile = File(...),
    target_column: Optional[str] = Form(None),
):
    """
    Analyze the CSV file and return feature engineering suggestions.
    """
    if not file.filename.lower().endswith(".csv"):
        return JSONResponse({"error": "Only CSV files are supported."}, status_code=400)

    raw_bytes = await file.read()
    try:
        try:
            df = pd.read_csv(BytesIO(raw_bytes))
        except Exception:
            txt = raw_bytes.decode("utf-8", errors="ignore")
            df = pd.read_csv(StringIO(txt))
    except Exception as e:
        return JSONResponse({"error": f"Failed to read CSV: {e}"}, status_code=400)

    if df.empty:
        return JSONResponse({"error": "CSV file is empty."}, status_code=400)


    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in df.columns if df[c].dtype == 'object' and c != target_column]

    suggestions, _, _ = _generate_advanced_features(df, num_cols, cat_cols, target_column)

    return JSONResponse({
        "status": "success",
        "suggestions": suggestions
    })


@app.post("/feature-engineering/csv")
async def feature_engineering_csv_endpoint(
    file: UploadFile = File(...),
    target_column: Optional[str] = Form(None),
    handle_missing: str = Form("mean"),
    handle_outliers: str = Form("none"),
    selected_features: str = Form("[]"),
):
    """
    Advanced feature engineering endpoint with intelligent feature generation.
    """
    if not file.filename.lower().endswith(".csv"):
        return JSONResponse({"error": "Only CSV files are supported."}, status_code=400)

    try:
        selected_feat_names = json.loads(selected_features)
        if not isinstance(selected_feat_names, list):
            selected_feat_names = []
    except Exception:
        selected_feat_names = []

    raw_bytes = await file.read()
    try:
        try:
            df = pd.read_csv(BytesIO(raw_bytes))
        except Exception:
            txt = raw_bytes.decode("utf-8", errors="ignore")
            df = pd.read_csv(StringIO(txt))
    except Exception as e:
        return JSONResponse({"error": f"Failed to read CSV: {e}"}, status_code=400)

    if df.empty:
        return JSONResponse({"error": "CSV file is empty."}, status_code=400)

    rows_before, cols = df.shape
    missing_before = int(df.isnull().sum().sum())
    column_names_original = list(df.columns)

    target_series = None
    if target_column and target_column in df.columns:
        target_series = df[target_column].copy()
        df_features = df.drop(columns=[target_column])
    else:
        df_features = df.copy()

    if handle_missing == "drop":
        df_features = df_features.dropna()
    elif handle_missing == "mean":
        df_features = fill_missing_safely(df_features)
    elif handle_missing == "none":
        pass
    else:
        return JSONResponse({"error": f"Invalid handle_missing option: {handle_missing}"}, status_code=400)

    num_cols = df_features.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in df_features.columns if df_features[c].dtype == 'object']

    num_feats = df_features[num_cols] if num_cols else pd.DataFrame()
    mask = pd.Series(True, index=df_features.index)
    if handle_outliers == "zscore":
        mask = zscore_mask(num_feats)
    elif handle_outliers == "iqr":
        mask = iqr_mask(num_feats)
    elif handle_outliers == "none":
        pass
    else:
        return JSONResponse({"error": f"Invalid handle_outliers option: {handle_outliers}"}, status_code=400)

    df_features = df_features[mask]

    suggestions, df_with_new_features, all_generated_features = _generate_advanced_features(
        df_features, num_cols, cat_cols, target_column
    )

    features_to_keep = [f for f in all_generated_features if f in selected_feat_names]

    df_processed = df_features.copy()
    
    for feat_name in features_to_keep:
        if feat_name in df_with_new_features.columns and feat_name not in df_processed.columns:
            df_processed[feat_name] = df_with_new_features[feat_name].copy()

    cat_cols_to_encode = [c for c in df_processed.columns 
                          if df_processed[c].dtype == 'object' 
                          and c not in features_to_keep]  
    
    for c in cat_cols_to_encode:
        try:
            df_processed[c] = df_processed[c].astype("category").cat.codes
        except Exception:
            df_processed[c] = df_processed[c].fillna("missing").astype(str)
            df_processed[c] = df_processed[c].astype("category").cat.codes

    if target_series is not None:
        try:
            target_series = target_series.loc[df_processed.index]
            df_processed[target_column] = target_series
        except Exception:
            df_processed = pd.concat([
                df_processed.reset_index(drop=True),
                target_series.reset_index(drop=True)
            ], axis=1)
            df_processed.columns = list(df_processed.columns[:-1]) + [target_column]

    try:
        processed_csv_text = df_processed.to_csv(index=False)
    except Exception as e:
        return JSONResponse({"error": f"Failed to serialize processed CSV: {e}"}, status_code=500)

    rows_after = int(len(df_processed))
    missing_after = int(df_processed.isnull().sum().sum())
    missing_values_handled = max(0, missing_before - missing_after)
    rows_removed = rows_before - rows_after
    outliers_removed = rows_before - rows_after if handle_outliers != "none" else 0
    columns_processed = int(len(column_names_original))

    final_columns = list(df_processed.columns)

    applied_methods = {
        "missing_value_strategy": ("drop rows" if handle_missing == "drop" else "fill with mean/mode"),
        "outlier_strategy": (handle_outliers if handle_outliers in ["zscore", "iqr"] else "none"),
        "categorical_encoding": "Label encoding (categorical -> integer codes)",
        "feature_generation": f"{len(features_to_keep)} features added",
        "scaling": "None (applied at model level)"
    }

    summary = {
        "rows_before": int(rows_before),
        "rows_after": int(rows_after),
        "rows_removed": int(rows_removed),
        "missing_values_before": int(missing_before),
        "missing_values_after": int(missing_after),
        "missing_values_handled": int(missing_values_handled),
        "outliers_removed": int(outliers_removed),
        "columns_processed": int(columns_processed),
        "column_names": final_columns,
        "generated_features": features_to_keep,
        "applied_methods": applied_methods
    }

    return JSONResponse({
        "status": "success",
        "message": "Feature engineering completed successfully",
        "summary": summary,
        "processed_csv": processed_csv_text,
        "filename": file.filename
    })