# core/model.py — SVM training, evaluation, and persistence
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ─── Result container ─────────────────────────────────────────────────────────

@dataclass
class SVMResult:
    model:    SVC
    scaler:   StandardScaler
    # training config
    kernel:   str
    C:        float
    gamma:    Any
    degree:   int
    # dataset sizes
    n_train:  int
    n_test:   int
    n_sv:     int
    # metrics
    acc_train:    float = 0.0
    acc_test:     float = 0.0
    precision:    float = 0.0
    recall:       float = 0.0
    f1:           float = 0.0
    report:       str   = ""
    cm:           Any   = field(default=None)    # np.ndarray
    # predictions
    y_pred_test:  Any   = field(default=None)
    y_prob_all:   Any   = field(default=None)    # P(Bull) for full dataset


# ─── Core training function ───────────────────────────────────────────────────

def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    X_all:   np.ndarray,
    kernel:  str   = "rbf",
    C:       float = 1.0,
    gamma:   Any   = "scale",
    degree:  int   = 3,
) -> SVMResult:
    """
    Scale, train, and evaluate an SVM classifier.

    Parameters
    ----------
    kernel : 'rbf' | 'linear' | 'poly' | 'sigmoid'
    C      : regularisation / inverse margin width
    gamma  : kernel coefficient ('scale', 'auto', or float)
    degree : polynomial degree (only used when kernel='poly')
    """
    # ── Scale ──
    scaler   = StandardScaler()
    Xt_sc    = scaler.fit_transform(X_train)
    Xte_sc   = scaler.transform(X_test)
    Xall_sc  = scaler.transform(X_all)

    # ── Train ──
    model = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        degree=degree,
        probability=True,
        random_state=42,
        cache_size=800,
        class_weight="balanced",   # handle bull/bear imbalance
    )
    model.fit(Xt_sc, y_train)

    # ── Evaluate ──
    y_pred_train = model.predict(Xt_sc)
    y_pred_test  = model.predict(Xte_sc)
    y_prob_all   = model.predict_proba(Xall_sc)[:, 1]

    result = SVMResult(
        model         = model,
        scaler        = scaler,
        kernel        = kernel,
        C             = C,
        gamma         = gamma,
        degree        = degree,
        n_train       = len(X_train),
        n_test        = len(X_test),
        n_sv          = len(model.support_),
        acc_train     = accuracy_score(y_train, y_pred_train),
        acc_test      = accuracy_score(y_test,  y_pred_test),
        precision     = precision_score(y_test, y_pred_test, zero_division=0),
        recall        = recall_score(y_test,    y_pred_test, zero_division=0),
        f1            = f1_score(y_test,        y_pred_test, zero_division=0),
        report        = classification_report(
                            y_test, y_pred_test,
                            target_names=["Bear", "Bull"],
                            zero_division=0,
                        ),
        cm            = confusion_matrix(y_test, y_pred_test),
        y_pred_test   = y_pred_test,
        y_prob_all    = y_prob_all,
    )
    return result


# ─── Predict single observation ───────────────────────────────────────────────

def predict_latest(result: SVMResult, X_latest: np.ndarray) -> dict:
    """
    Classify the most recent feature vector.

    Returns a dict with keys: label (0/1), signal ('Bull'/'Bear'),
    probability (P(Bull)), confidence_pct.
    """
    x_sc   = result.scaler.transform(X_latest.reshape(1, -1))
    label  = int(result.model.predict(x_sc)[0])
    prob   = float(result.model.predict_proba(x_sc)[0, 1])
    conf   = max(prob, 1 - prob)
    return dict(
        label           = label,
        signal          = "Bull" if label == 1 else "Bear",
        probability     = prob,
        confidence_pct  = conf * 100,
    )


# ─── PCA 2-D projection (for scatter visualisation) ──────────────────────────

def pca_2d(result: SVMResult, X_all: np.ndarray):
    """
    Project scaled feature matrix to 2 components via PCA.
    Returns (X_2d, pca_model, explained_variance_ratio).
    """
    from sklearn.decomposition import PCA
    X_sc = result.scaler.transform(X_all)
    pca  = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_sc)
    return X_2d, pca, pca.explained_variance_ratio_


# ─── Model persistence ────────────────────────────────────────────────────────

MODEL_DIR = "models"

def save_model(result: SVMResult, symbol: str) -> str:
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f"svm_{symbol}_{result.kernel}_C{result.C}.pkl")
    joblib.dump(result, path)
    return path


def load_model(path: str) -> SVMResult:
    return joblib.load(path)
