"""
Pagina 2 — Training Modello
Permette di:
- Configurare i pesi delle feature prima del training
- Impostare i parametri del modello XGBoost
- Avviare il training con progress live
- Visualizzare metriche, feature importance e confusion matrix
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.schema_manager import load_schema, get_numeric_features, get_categorical_features
from core.model_trainer  import train, load_training_artifacts

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Training — HR Churn",
    page_icon="🏋️",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Sans:wght@300;400;500&family=DM+Mono&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1,h2,h3 { font-family: 'Syne', sans-serif !important; }
.block-container { padding-top: 1.5rem !important; max-width: 1300px !important; }
.stButton > button {
    background: #e94560 !important; color: white !important;
    border: none !important; border-radius: 8px !important; font-weight: 600 !important;
}
[data-testid="stSidebar"] { background: #13151c !important; }
.metric-box {
    background: #13151c; border: 1px solid #252836; border-radius: 10px;
    padding: 1rem 1.2rem; text-align: center;
}
.metric-num { font-family: 'DM Mono', monospace; font-size: 2rem;
              font-weight: 700; line-height: 1; }
.metric-lbl { font-size: 0.72rem; color: #6b7280; text-transform: uppercase;
              letter-spacing: 0.08em; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#94a3b8", size=12),
    margin=dict(l=10, r=10, t=30, b=10),
)

# ─────────────────────────────────────────────
# LOAD SCHEMA + ARTIFACTS
# ─────────────────────────────────────────────
schema    = load_schema()
artifacts = load_training_artifacts()

numeric_feats     = get_numeric_features(schema)
categorical_feats = get_categorical_features(schema)
all_features      = numeric_feats + categorical_feats

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("# 🏋️ Training Modello")
st.markdown(
    "<p style='color:#6b7280;'>Configura i pesi delle feature, imposta i parametri "
    "XGBoost e avvia il training. Visualizza metriche e feature importance.</p>",
    unsafe_allow_html=True,
)

# Stato modello esistente
if artifacts:
    m = artifacts["metrics"]
    trained_at = m.get("trained_at", "")[:16].replace("T", " ")
    st.success(
        f"✓ Modello già trainato — ROC-AUC: **{m['roc_auc']:.3f}** | "
        f"CV: **{m['cv_roc_auc_mean']:.3f} ±{m['cv_roc_auc_std']:.3f}** | "
        f"Trainato il: {trained_at}"
    )
else:
    st.warning("⚠ Nessun modello trovato. Configura e avvia il training.")

st.markdown("---")

# ─────────────────────────────────────────────
# LAYOUT PRINCIPALE
# ─────────────────────────────────────────────
col_config, col_results = st.columns([1, 1.5])


# ══════════════════════════════════════════════
# COLONNA SINISTRA — CONFIGURAZIONE
# ══════════════════════════════════════════════
with col_config:

    # ── Pesi feature ──
    st.markdown("### 🎚️ Pesi delle feature")
    st.markdown(
        "<p style='color:#6b7280; font-size:0.85rem;'>"
        "Aumenta il peso delle feature che ritieni più importanti nel tuo contesto. "
        "Il modello darà più enfasi ai campioni critici su quelle dimensioni.</p>",
        unsafe_allow_html=True,
    )

    # Carica pesi precedenti se disponibili
    saved_weights = {}
    if artifacts:
        saved_weights = artifacts["metrics"].get("feature_weights_used", {})

    feature_weights = {}

    # Raggruppa feature per categoria
    weight_groups = {
        "💼 Carriera": [
            "anni_senza_promozione", "num_promozioni",
            "crescita_stipendio_percentuale", "stipendio_annuo",
        ],
        "⏱️ Workload": [
            "ore_settimanali_medie", "ferie_residue",
            "giorni_assenza_anno", "num_progetti_anno",
        ],
        "💬 Engagement": [
            "manager_score", "enps_score",
            "giorni_ultimo_checkin_hr",
        ],
        "👤 Anagrafica": [
            "eta", "anni_in_azienda",
        ],
    }

    # Aggiungi feature custom non ancora in nessun gruppo
    grouped_features = {f for fs in weight_groups.values() for f in fs}
    custom_numeric   = [f for f in numeric_feats if f not in grouped_features]
    if custom_numeric:
        weight_groups["🧩 Custom"] = custom_numeric

    for group_name, group_features in weight_groups.items():
        present = [f for f in group_features if f in all_features]
        if not present:
            continue

        with st.expander(group_name, expanded=(group_name in ("💼 Carriera", "⏱️ Workload"))):
            for feat in present:
                default_w = saved_weights.get(feat, 1.0)
                w = st.slider(
                    feat.replace("_", " ").title(),
                    min_value=0.1,
                    max_value=3.0,
                    value=float(default_w),
                    step=0.1,
                    key=f"weight_{feat}",
                    help=f"1.0 = peso normale | >1.0 = più importante | <1.0 = meno importante",
                )
                feature_weights[feat] = w

    # Feature categoriche — peso fisso
    for feat in categorical_feats:
        feature_weights[feat + "_enc"] = 1.0

    st.markdown("---")

    # ── Parametri XGBoost ──
    st.markdown("### ⚙️ Parametri XGBoost")
    with st.expander("Parametri avanzati", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            n_estimators   = st.slider("n_estimators",   50,  1000, 300, 50)
            max_depth      = st.slider("max_depth",       2,   10,   5,   1)
            cv_folds       = st.slider("CV folds",        3,   10,   5,   1)
        with c2:
            learning_rate  = st.slider("learning_rate",  0.01, 0.3, 0.05, 0.01)
            subsample      = st.slider("subsample",      0.5,  1.0, 0.8,  0.05)
            test_size      = st.slider("Test size (%)",  10,   40,  20,   5)

        seed = st.number_input("Seed", value=42, step=1)

    model_params = {
        "n_estimators":    n_estimators,
        "max_depth":       max_depth,
        "learning_rate":   learning_rate,
        "subsample":       subsample,
    }

    st.markdown("---")

    # ── Bottone training ──
    df_ready = os.path.exists("data/hr_dataset.csv")
    if not df_ready:
        st.error("❌ Dataset non trovato. Vai alla pagina Configurazione.")
    else:
        df_info = pd.read_csv("data/hr_dataset.csv", nrows=1)
        st.caption(f"Dataset: `data/hr_dataset.csv`")

    train_btn = st.button(
        "🚀 Avvia Training",
        width="stretch",
        disabled=not df_ready,
    )

    # ── Peso summary ──
    non_default = {k: v for k, v in feature_weights.items()
                   if abs(v - 1.0) > 0.05 and not k.endswith("_enc")}
    if non_default:
        st.markdown("**Feature con pesi modificati:**")
        for feat, w in sorted(non_default.items(), key=lambda x: x[1], reverse=True):
            color = "#22c55e" if w > 1 else "#f97316"
            arrow = "▲" if w > 1 else "▼"
            st.markdown(
                f"<div style='display:flex; justify-content:space-between; "
                f"padding:3px 0; font-size:0.85rem;'>"
                f"<span>{feat.replace('_',' ')}</span>"
                f"<span style='color:{color}; font-family:DM Mono,monospace; font-weight:700;'>"
                f"{arrow} {w:.1f}</span></div>",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════
if train_btn and df_ready:
    with col_results:
        progress = st.progress(0, text="Caricamento dataset...")

        try:
            df = pd.read_csv("data/hr_dataset.csv")
            progress.progress(15, text="Preparazione feature...")

            progress.progress(30, text="Cross-validation in corso...")
            result = train(
                df=df,
                schema=schema,
                feature_weights=feature_weights,
                model_params=model_params,
                test_size=test_size / 100,
                cv_folds=cv_folds,
                seed=int(seed),
            )
            progress.progress(100, text="Training completato!")
            import time; time.sleep(0.5)
            progress.empty()

            artifacts = result  # aggiorna per i grafici sotto
            st.success("✓ Modello trainato e salvato in `models/`")

        except Exception as ex:
            progress.empty()
            import traceback
            st.error(f"Errore durante il training: {ex}")
            st.code(traceback.format_exc())
            st.stop()


# ══════════════════════════════════════════════
# COLONNA DESTRA — RISULTATI
# ══════════════════════════════════════════════
with col_results:
    if not artifacts:
        st.markdown(
            "<div style='text-align:center; padding:4rem 2rem; color:#4b5563;'>"
            "<div style='font-size:3rem;'>🏋️</div>"
            "<div style='font-size:1rem; margin-top:1rem;'>"
            "Configura i parametri e avvia il training</div></div>",
            unsafe_allow_html=True,
        )
    else:
        m    = artifacts["metrics"]
        imp  = artifacts["importance"]

        # ── KPI metriche ──
        st.markdown("### 📊 Metriche")
        k1, k2, k3, k4 = st.columns(4)
        kpis = [
            (k1, f"{m['roc_auc']:.3f}",            "ROC-AUC",         "#e94560"),
            (k2, f"{m['average_precision']:.3f}",   "Avg Precision",   "#f97316"),
            (k3, f"{m['cv_roc_auc_mean']:.3f}",     "CV ROC-AUC",      "#06b6d4"),
            (k4, f"±{m['cv_roc_auc_std']:.3f}",     "CV Std Dev",      "#6b7280"),
        ]
        for col, val, label, color in kpis:
            with col:
                st.markdown(
                    f"<div class='metric-box'>"
                    f"<div class='metric-num' style='color:{color};'>{val}</div>"
                    f"<div class='metric-lbl'>{label}</div></div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Classification report ──
        report = m.get("classification_report", {})
        if report:
            cr1, cr2 = st.columns(2)
            for col_r, class_name in zip([cr1, cr2], ["Stay", "Churn"]):
                cls = report.get(class_name, {})
                with col_r:
                    st.markdown(
                        f"<div class='metric-box'>"
                        f"<div style='font-size:0.8rem; color:#6b7280; "
                        f"text-transform:uppercase; margin-bottom:6px;'>{class_name}</div>"
                        f"<div style='display:flex; justify-content:space-around;'>"
                        f"<div><div style='font-size:1.1rem; font-weight:700; color:#e2e8f0;'>"
                        f"{cls.get('precision',0):.2f}</div>"
                        f"<div style='font-size:0.68rem; color:#6b7280;'>Precision</div></div>"
                        f"<div><div style='font-size:1.1rem; font-weight:700; color:#e2e8f0;'>"
                        f"{cls.get('recall',0):.2f}</div>"
                        f"<div style='font-size:0.68rem; color:#6b7280;'>Recall</div></div>"
                        f"<div><div style='font-size:1.1rem; font-weight:700; color:#e2e8f0;'>"
                        f"{cls.get('f1-score',0):.2f}</div>"
                        f"<div style='font-size:0.68rem; color:#6b7280;'>F1</div></div>"
                        f"</div></div>",
                        unsafe_allow_html=True,
                    )

        st.markdown("---")

        # ── Feature importance ──
        st.markdown("### 🔍 Feature Importance")
        st.markdown(
            "<p style='color:#6b7280; font-size:0.85rem;'>"
            "Quali feature influenzano di più le predizioni del modello.</p>",
            unsafe_allow_html=True,
        )

        top_n_imp = st.slider("Mostra top N feature", 5, min(30, len(imp)), 15, 1)
        imp_items = list(imp.items())[:top_n_imp]
        feat_names = [i[0].replace("_enc", " (cat)").replace("_", " ") for i in imp_items]
        feat_vals  = [i[1] for i in imp_items]

        if feat_vals:
            fig_imp = go.Figure(go.Bar(
                x=feat_vals,
                y=feat_names,
                orientation="h",
                marker=dict(
                    color=feat_vals,
                    colorscale=[[0, "#06b6d4"], [0.5, "#e94560"], [1, "#ef4444"]],
                ),
                text=[f"{v:.3f}" for v in feat_vals],
                textposition="outside",
                textfont=dict(color="#94a3b8", size=10),
            ))
            fig_imp.update_layout(
                **PLOTLY_THEME,
                height=max(300, top_n_imp * 28),
                yaxis=dict(categoryorder="total ascending"),
                xaxis=dict(range=[0, max(feat_vals) * 1.15]),
            )
            st.plotly_chart(fig_imp, width="stretch")
        else:
            st.info("Nessuna feature importance disponibile — riaddestra il modello.")

        # ── Confronto pesi impostati vs importanza reale ──
        saved_w = m.get("feature_weights_used", {})
        if saved_w:
            st.markdown("### ⚖️ Pesi impostati vs Importanza reale")
            st.markdown(
                "<p style='color:#6b7280; font-size:0.85rem;'>"
                "Confronto tra i pesi che hai impostato prima del training "
                "e l'importanza che il modello ha effettivamente assegnato.</p>",
                unsafe_allow_html=True,
            )

            compare_data = []
            for feat, imp_val in list(imp.items())[:20]:
                base_feat = feat.replace("_enc", "")
                weight    = saved_w.get(base_feat, saved_w.get(feat, 1.0))
                compare_data.append({
                    "Feature":    feat.replace("_enc", " (cat)").replace("_", " "),
                    "Peso impostato": weight,
                    "Importanza modello": imp_val,
                })

            df_compare = pd.DataFrame(compare_data)

            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Bar(
                name="Peso impostato",
                x=df_compare["Feature"],
                y=df_compare["Peso impostato"],
                marker_color="#06b6d4",
                opacity=0.8,
            ))
            fig_cmp.add_trace(go.Bar(
                name="Importanza modello",
                x=df_compare["Feature"],
                y=df_compare["Importanza modello"] * 10,  # scala per leggibilità
                marker_color="#e94560",
                opacity=0.8,
            ))
            fig_cmp.update_layout(
                **PLOTLY_THEME,
                barmode="group",
                height=360,
                xaxis=dict(tickangle=-35),
                legend=dict(font=dict(color="#94a3b8")),
                annotations=[dict(
                    text="* Importanza modello scalata x10 per confronto visivo",
                    xref="paper", yref="paper", x=0, y=-0.25,
                    showarrow=False, font=dict(size=10, color="#6b7280"),
                )],
            )
            st.plotly_chart(fig_cmp, width="stretch")

        # ── Confusion matrix ──
        st.markdown("### 🎯 Confusion Matrix")
        cm_data = m.get("confusion_matrix", [[0, 0], [0, 0]])
        labels  = ["Stay (0)", "Churn (1)"]

        fig_cm = go.Figure(go.Heatmap(
            z=cm_data,
            x=labels, y=labels,
            colorscale=[[0, "#13151c"], [1, "#e94560"]],
            text=[[str(v) for v in row] for row in cm_data],
            texttemplate="%{text}",
            textfont=dict(size=18, color="white"),
            showscale=False,
        ))
        fig_cm.update_layout(
            **PLOTLY_THEME,
            height=280,
            xaxis=dict(title="Predetto"),
            yaxis=dict(title="Reale", autorange="reversed"),
        )
        st.plotly_chart(fig_cm, width="stretch")

        st.markdown(
            f"<div style='color:#6b7280; font-size:0.8rem;'>"
            f"Train: {m['n_train']} campioni | Test: {m['n_test']} campioni | "
            f"Churn rate train: {m['churn_rate_train']:.1%}</div>",
            unsafe_allow_html=True,
        )

        # ── Download metriche ──
        st.markdown("---")
        metrics_json = json.dumps(m, indent=2).encode("utf-8")
        st.download_button(
            "⬇️ Esporta metriche (JSON)",
            data=metrics_json,
            file_name="training_metrics.json",
            mime="application/json",
        )
