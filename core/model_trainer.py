"""
Model Trainer
Addestra il modello XGBoost con pesi delle feature configurabili.
Salva modello, metriche e configurazione su disco.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
)
from xgboost import XGBClassifier

MODEL_DIR = "models"


# ─────────────────────────────────────────────
# FEATURE WEIGHTS → SAMPLE WEIGHTS
# ─────────────────────────────────────────────

def build_sample_weights(X: np.ndarray, y: np.ndarray,
                          feature_cols: list, feature_weights: dict) -> np.ndarray:
    """
    Converte i pesi delle feature in sample weights per XGBoost.
    Dipendenti che hanno valori critici nelle feature ad alto peso
    ricevono un sample weight maggiore durante il training.

    Args:
        X:               matrice feature
        y:               label target
        feature_cols:    lista nomi feature nell'ordine delle colonne di X
        feature_weights: dict {feature_name: peso 0.0-2.0}

    Returns:
        array di sample weights
    """
    sample_weights = np.ones(len(X))

    for feat_name, weight in feature_weights.items():
        if feat_name not in feature_cols or weight == 1.0:
            continue
        col_idx = feature_cols.index(feat_name)
        col_vals = X[:, col_idx]

        # Normalizza i valori nella colonna 0-1
        col_min, col_max = col_vals.min(), col_vals.max()
        if col_max > col_min:
            normalized = (col_vals - col_min) / (col_max - col_min)
        else:
            normalized = np.zeros_like(col_vals)

        # I campioni positivi (churn=1) con valori alti della feature
        # ricevono più peso se la feature è pesata in alto
        for i in range(len(X)):
            if y[i] == 1:
                sample_weights[i] *= (1.0 + (weight - 1.0) * normalized[i])

    # Normalizza i pesi
    sample_weights = sample_weights / sample_weights.mean()
    return sample_weights


# ─────────────────────────────────────────────
# TRAINING PIPELINE
# ─────────────────────────────────────────────

def train(
    df: pd.DataFrame,
    schema: dict,
    feature_weights: dict = None,
    model_params: dict = None,
    test_size: float = 0.2,
    cv_folds: int = 5,
    seed: int = 42,
) -> dict:
    """
    Pipeline completa di training.

    Args:
        df:              DataFrame con i dati di training
        schema:          schema con definizione colonne
        feature_weights: dict {feature_name: peso} — default 1.0 per tutte
        model_params:    parametri XGBoost custom
        test_size:       proporzione test set
        cv_folds:        fold per cross-validation
        seed:            seed per riproducibilità

    Returns:
        dict con modello, metriche, importanze e artefatti
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Prepara feature columns dallo schema ──
    from core.schema_manager import get_numeric_features, get_categorical_features
    numeric_features     = get_numeric_features(schema)
    categorical_features = get_categorical_features(schema)

    # Filtra solo colonne presenti nel DataFrame
    numeric_features     = [c for c in numeric_features     if c in df.columns]
    categorical_features = [c for c in categorical_features if c in df.columns]

    if not numeric_features and not categorical_features:
        raise ValueError("Nessuna feature trovata nel DataFrame. Controlla lo schema.")

    if "churn_label" not in df.columns:
        raise ValueError("Colonna 'churn_label' non trovata nel DataFrame.")

    # ── Label encoding categoriche ──
    encoders = {}
    df_model = df.copy()
    for col in categorical_features:
        le = LabelEncoder()
        df_model[col + "_enc"] = le.fit_transform(df_model[col].astype(str))
        encoders[col] = le

    feature_cols = numeric_features + [c + "_enc" for c in categorical_features]
    X = df_model[feature_cols].values.astype(float)
    y = df_model["churn_label"].values.astype(int)

    # ── Split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # ── Sample weights dai feature weights ──
    sw_train = None
    if feature_weights:
        sw_train = build_sample_weights(X_train, y_train, feature_cols, feature_weights)

    # ── Bilanciamento classi ──
    neg, pos = np.bincount(y_train)
    scale_pos = neg / pos if pos > 0 else 1.0

    # ── Parametri modello ──
    default_params = {
        "n_estimators":    300,
        "max_depth":       5,
        "learning_rate":   0.05,
        "subsample":       0.8,
        "colsample_bytree":0.8,
        "scale_pos_weight":scale_pos,
        "eval_metric":     "logloss",
        "random_state":    seed,
        "n_jobs":          -1,
    }
    if model_params:
        default_params.update(model_params)

    model = XGBClassifier(**default_params)

    # ── Cross-validation (senza sample weights per compatibilità sklearn) ──
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")

    # ── Training finale ──
    fit_kwargs = {"eval_set": [(X_test, y_test)], "verbose": False}
    if sw_train is not None:
        fit_kwargs["sample_weight"] = sw_train
    model.fit(X_train, y_train, **fit_kwargs)

    # ── Metriche ──
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)
    ap      = average_precision_score(y_test, y_proba)
    cm      = confusion_matrix(y_test, y_pred).tolist()
    report  = classification_report(y_test, y_pred,
                                    target_names=["Stay", "Churn"], output_dict=True)

    # ── Feature importance ──
    importance_raw = model.feature_importances_
    importance = {
        feat: round(float(imp), 4)
        for feat, imp in sorted(zip(feature_cols, importance_raw),
                                key=lambda x: x[1], reverse=True)
    }

    # ── Metriche finali ──
    metrics = {
        "roc_auc":           round(float(auc), 4),
        "average_precision": round(float(ap), 4),
        "cv_roc_auc_mean":   round(float(cv_scores.mean()), 4),
        "cv_roc_auc_std":    round(float(cv_scores.std()), 4),
        "n_train":           int(len(X_train)),
        "n_test":            int(len(X_test)),
        "churn_rate_train":  round(float(y_train.mean()), 4),
        "confusion_matrix":  cm,
        "classification_report": report,
        "trained_at":        datetime.now().isoformat(),
        "feature_weights_used": feature_weights or {},
        "model_params":      default_params,
    }

    # ── Salva artefatti in session_state (isolato per utente) ──
    artifacts = {
        "model":         model,
        "encoders":      encoders,
        "feature_cols":  feature_cols,
        "metrics":       metrics,
        "importance":    importance,
        "X_test":        X_test,
        "y_test":        y_test,
        "y_proba":       y_proba,
    }

    import streamlit as st
    st.session_state["training_artifacts"] = artifacts

    # Backup su disco (solo locale, ignorato se fallisce)
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(model,        f"{MODEL_DIR}/churn_model.pkl")
        joblib.dump(encoders,     f"{MODEL_DIR}/encoders.pkl")
        joblib.dump(feature_cols, f"{MODEL_DIR}/feature_cols.pkl")
        with open(f"{MODEL_DIR}/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        with open(f"{MODEL_DIR}/feature_importance.json", "w") as f:
            json.dump(importance, f, indent=2)
    except OSError:
        pass

    return artifacts


def load_training_artifacts() -> dict | None:
    """
    Carica gli artefatti di training da st.session_state (isolato per utente).
    Fallback: disco locale (solo sviluppo).
    """
    import streamlit as st

    # Prima cerca in session_state
    if "training_artifacts" in st.session_state and st.session_state["training_artifacts"]:
        return st.session_state["training_artifacts"]

    # Fallback su disco (solo locale)
    required = [
        f"{MODEL_DIR}/churn_model.pkl",
        f"{MODEL_DIR}/encoders.pkl",
        f"{MODEL_DIR}/feature_cols.pkl",
        f"{MODEL_DIR}/metrics.json",
    ]
    if not all(os.path.exists(p) for p in required):
        return None

    try:
        with open(f"{MODEL_DIR}/metrics.json") as f:
            metrics = json.load(f)
        importance = {}
        if os.path.exists(f"{MODEL_DIR}/feature_importance.json"):
            with open(f"{MODEL_DIR}/feature_importance.json") as f:
                importance = json.load(f)
        artifacts = {
            "model":        joblib.load(f"{MODEL_DIR}/churn_model.pkl"),
            "encoders":     joblib.load(f"{MODEL_DIR}/encoders.pkl"),
            "feature_cols": joblib.load(f"{MODEL_DIR}/feature_cols.pkl"),
            "metrics":      metrics,
            "importance":   importance,
        }
        st.session_state["training_artifacts"] = artifacts
        return artifacts
    except Exception:
        return None
