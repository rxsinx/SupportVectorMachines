# components/metrics.py — Streamlit metric display helpers
from __future__ import annotations

import streamlit as st

from config import PALETTE as P


def metric_row(result) -> None:
    """Six KPI cards across the top of the page."""
    cols = st.columns(6)
    cards = [
        ("Test Accuracy",    f"{result.acc_test:.1%}",   None),
        ("Train Accuracy",   f"{result.acc_train:.1%}",  None),
        ("Precision",        f"{result.precision:.1%}",  None),
        ("Recall",           f"{result.recall:.1%}",     None),
        ("F1 Score",         f"{result.f1:.1%}",         None),
        ("Support Vectors",  str(result.n_sv),           None),
    ]
    for col, (label, val, delta) in zip(cols, cards):
        col.metric(label, val, delta)


def signal_badge(signal_info: dict) -> None:
    """
    Large coloured badge showing the latest regime signal.
    signal_info = {"signal": "Bull"|"Bear", "probability": float,
                   "confidence_pct": float}
    """
    sig   = signal_info["signal"]
    prob  = signal_info["probability"]
    conf  = signal_info["confidence_pct"]
    color = P["bull_dot"] if sig == "Bull" else P["bear_dot"]
    arrow = "▲" if sig == "Bull" else "▼"

    st.markdown(
        f"""
        <div style="
            background:{P['panel']};
            border:2px solid {color};
            border-radius:12px;
            padding:20px 28px;
            text-align:center;
            font-family:monospace;
        ">
          <div style="font-size:2.6rem;color:{color};font-weight:900;
                      letter-spacing:2px;">
            {arrow}  {sig.upper()}
          </div>
          <div style="color:{P['dim_text']};font-size:0.85rem;margin-top:8px;">
            P(Bull) = <b style='color:{color}'>{prob:.1%}</b>
            &nbsp;·&nbsp;
            Confidence <b style='color:{P['boundary']}'>{conf:.1f}%</b>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def classification_report_table(report_text: str) -> None:
    """Parse sklearn classification_report string into an st.dataframe."""
    import pandas as pd
    rows = []
    for line in report_text.strip().splitlines()[2:]:
        parts = line.split()
        if len(parts) >= 5:
            rows.append({
                "Class":     parts[0],
                "Precision": float(parts[1]),
                "Recall":    float(parts[2]),
                "F1":        float(parts[3]),
                "Support":   int(parts[4]),
            })
    if rows:
        df = pd.DataFrame(rows).set_index("Class")
        st.dataframe(
            df.style
              .format({"Precision": "{:.2f}", "Recall": "{:.2f}", "F1": "{:.2f}"})
              .background_gradient(subset=["F1"], cmap="RdYlGn"),
            use_container_width=True,
        )


def kernel_explainer(kernel: str) -> None:
    """Inline markdown explanation of the chosen kernel."""
    desc = {
        "rbf":     "**RBF (Gaussian)** — Most versatile. Creates non-linear, "
                   "radial boundaries that adapt to complex market structures. "
                   "Controlled by γ (gamma): larger γ → tighter fit.",
        "linear":  "**Linear** — Fastest. Draws a flat hyperplane through feature "
                   "space. Best when bull/bear regimes are linearly separable. "
                   "Weight vector doubles as feature importance.",
        "poly":    "**Polynomial** — Curves the decision boundary via a degree-d "
                   "polynomial. Captures interaction effects between features. "
                   "Degree 3 is a common sweet spot.",
        "sigmoid": "**Sigmoid** — Mimics a single-layer neural network. "
                   "Can produce non-convex regions; useful when RBF over-fits.",
    }
    st.info(desc.get(kernel, "Select a kernel to see its description."))
