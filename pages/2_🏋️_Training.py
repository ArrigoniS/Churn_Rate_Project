"""
Pagina 2 — Training Modello / Model Training
"""
import os, sys, json, time, traceback
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.schema_manager import load_schema, get_numeric_features, get_categorical_features
from core.model_trainer  import train, load_training_artifacts
from utils.i18n          import t, render_lang_toggle

st.set_page_config(page_title="Training — HR Churn", page_icon="🏋️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Sans:wght@300;400;500&family=DM+Mono&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1,h2,h3 { font-family: 'Syne', sans-serif !important; }
.block-container { padding-top: 1.5rem !important; max-width: 1300px !important; }
.stButton > button { background:#e94560 !important; color:white !important;
    border:none !important; border-radius:8px !important; font-weight:600 !important; }
[data-testid="stSidebar"] { background: #13151c !important; }
.metric-box { background:#13151c; border:1px solid #252836; border-radius:10px; padding:1rem 1.2rem; text-align:center; }
.metric-num { font-family:'DM Mono',monospace; font-size:2rem; font-weight:700; line-height:1; }
.metric-lbl { font-size:0.72rem; color:#6b7280; text-transform:uppercase; letter-spacing:0.08em; margin-top:4px; }
</style>
""", unsafe_allow_html=True)

render_lang_toggle()

PLOTLY_THEME = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="DM Sans", color="#94a3b8", size=12),
                    margin=dict(l=10, r=10, t=30, b=10))

schema    = load_schema()
artifacts = load_training_artifacts()
numeric_feats     = get_numeric_features(schema)
categorical_feats = get_categorical_features(schema)
all_features      = numeric_feats + categorical_feats

st.markdown(t("trn_title"))
st.markdown(f"<p style='color:#6b7280;'>{t('trn_sub')}</p>", unsafe_allow_html=True)

if artifacts:
    m = artifacts["metrics"]
    trained_at = m.get("trained_at","")[:16].replace("T"," ")
    st.success(f"✓ {t('trn_model_trained')} — ROC-AUC: **{m['roc_auc']:.3f}** | CV: **{m['cv_roc_auc_mean']:.3f} ±{m['cv_roc_auc_std']:.3f}** | {trained_at}")
else:
    st.warning(t("trn_no_model"))
st.markdown("---")

col_config, col_results = st.columns([1, 1.5])

with col_config:
    st.markdown(t("trn_weights_title"))
    st.markdown(f"<p style='color:#6b7280;font-size:0.85rem;'>{t('trn_weights_sub')}</p>", unsafe_allow_html=True)

    saved_weights = artifacts["metrics"].get("feature_weights_used", {}) if artifacts else {}
    feature_weights = {}

    weight_groups = {
        "💼 Career":    ["anni_senza_promozione","num_promozioni","crescita_stipendio_percentuale","stipendio_annuo"],
        "⏱️ Workload":  ["ore_settimanali_medie","ferie_residue","giorni_assenza_anno","num_progetti_anno"],
        "💬 Engagement":["manager_score","enps_score","giorni_ultimo_checkin_hr"],
        "👤 Profile":   ["eta","anni_in_azienda"],
    }
    grouped = {f for fs in weight_groups.values() for f in fs}
    custom_numeric = [f for f in numeric_feats if f not in grouped]
    if custom_numeric:
        weight_groups["🧩 Custom"] = custom_numeric

    for group_name, group_features in weight_groups.items():
        present = [f for f in group_features if f in all_features]
        if not present: continue
        with st.expander(group_name, expanded=(group_name in ("💼 Career","⏱️ Workload"))):
            for feat in present:
                w = st.slider(feat.replace("_"," ").title(), 0.1, 3.0,
                              float(saved_weights.get(feat, 1.0)), 0.1,
                              key=f"weight_{feat}", help=t("trn_weight_help"))
                feature_weights[feat] = w

    for feat in categorical_feats:
        feature_weights[feat + "_enc"] = 1.0

    st.markdown("---")
    st.markdown(t("trn_params_title"))
    with st.expander(t("trn_params_advanced"), expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            n_estimators  = st.slider("n_estimators",  50, 1000, 300, 50)
            max_depth     = st.slider("max_depth",       2,   10,   5,  1)
            cv_folds      = st.slider("CV folds",        3,   10,   5,  1)
        with c2:
            learning_rate = st.slider("learning_rate", 0.01, 0.3, 0.05, 0.01)
            subsample     = st.slider("subsample",      0.5,  1.0,  0.8, 0.05)
            test_size     = st.slider("Test size (%)",  10,   40,   20,  5)
        seed = st.number_input("Seed", value=42, step=1)

    model_params = {"n_estimators":n_estimators,"max_depth":max_depth,
                    "learning_rate":learning_rate,"subsample":subsample}

    st.markdown("---")
    df_ready = (st.session_state.get("df_train_raw") is not None or os.path.exists("data/hr_dataset.csv"))
    if not df_ready:
        st.error(t("trn_no_dataset"))
    else:
        st.caption("Dataset: `data/hr_dataset.csv`")

    train_btn = st.button(t("trn_start_btn"), width="stretch", disabled=not df_ready)

    non_default = {k:v for k,v in feature_weights.items() if abs(v-1.0)>0.05 and not k.endswith("_enc")}
    if non_default:
        st.markdown(t("trn_modified_weights"))
        for feat, w in sorted(non_default.items(), key=lambda x: x[1], reverse=True):
            color = "#22c55e" if w > 1 else "#f97316"
            arrow = "▲" if w > 1 else "▼"
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;padding:3px 0;font-size:0.85rem;'>"
                f"<span>{feat.replace('_',' ')}</span>"
                f"<span style='color:{color};font-family:DM Mono,monospace;font-weight:700;'>{arrow} {w:.1f}</span></div>",
                unsafe_allow_html=True)

if train_btn and df_ready:
    with col_results:
        progress = st.progress(0, text=t("trn_loading"))
        try:
            df = st.session_state.get("df_train_raw")
            if df is None and os.path.exists("data/hr_dataset.csv"):
                df = pd.read_csv("data/hr_dataset.csv")
            if df is None:
                st.error(t("trn_no_dataset"))
                st.stop()
            progress.progress(15, text=t("trn_preparing"))
            progress.progress(30, text=t("trn_cv"))
            result = train(df=df, schema=schema, feature_weights=feature_weights,
                           model_params=model_params, test_size=test_size/100,
                           cv_folds=cv_folds, seed=int(seed))
            progress.progress(100, text=t("trn_complete"))
            time.sleep(0.5)
            progress.empty()
            artifacts = result
            st.success(t("trn_success"))
        except Exception as ex:
            progress.empty()
            st.error(f"{t('trn_error')} {ex}")
            st.code(traceback.format_exc())
            st.stop()

with col_results:
    if not artifacts:
        st.markdown(
            f"<div style='text-align:center;padding:4rem 2rem;color:#4b5563;'>"
            f"<div style='font-size:3rem;'>🏋️</div>"
            f"<div style='font-size:1rem;margin-top:1rem;'>{t('trn_empty_state')}</div></div>",
            unsafe_allow_html=True)
    else:
        m   = artifacts["metrics"]
        imp = artifacts["importance"]

        st.markdown(t("trn_metrics_title"))
        k1, k2, k3, k4 = st.columns(4)
        for col, val, label, color in [
            (k1, f"{m['roc_auc']:.3f}",           "ROC-AUC",       "#e94560"),
            (k2, f"{m['average_precision']:.3f}",  "Avg Precision", "#f97316"),
            (k3, f"{m['cv_roc_auc_mean']:.3f}",    "CV ROC-AUC",    "#06b6d4"),
            (k4, f"±{m['cv_roc_auc_std']:.3f}",    "CV Std Dev",    "#6b7280"),
        ]:
            with col:
                st.markdown(f"<div class='metric-box'><div class='metric-num' style='color:{color};'>{val}</div><div class='metric-lbl'>{label}</div></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        report = m.get("classification_report", {})
        if report:
            cr1, cr2 = st.columns(2)
            for col_r, class_name in zip([cr1,cr2], ["Stay","Churn"]):
                cls = report.get(class_name, {})
                with col_r:
                    st.markdown(
                        f"<div class='metric-box'>"
                        f"<div style='font-size:0.8rem;color:#6b7280;text-transform:uppercase;margin-bottom:6px;'>{class_name}</div>"
                        f"<div style='display:flex;justify-content:space-around;'>"
                        f"<div><div style='font-size:1.1rem;font-weight:700;color:#e2e8f0;'>{cls.get('precision',0):.2f}</div><div style='font-size:0.68rem;color:#6b7280;'>Precision</div></div>"
                        f"<div><div style='font-size:1.1rem;font-weight:700;color:#e2e8f0;'>{cls.get('recall',0):.2f}</div><div style='font-size:0.68rem;color:#6b7280;'>Recall</div></div>"
                        f"<div><div style='font-size:1.1rem;font-weight:700;color:#e2e8f0;'>{cls.get('f1-score',0):.2f}</div><div style='font-size:0.68rem;color:#6b7280;'>F1</div></div>"
                        f"</div></div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(t("trn_importance_title"))
        st.markdown(f"<p style='color:#6b7280;font-size:0.85rem;'>{t('trn_importance_sub')}</p>", unsafe_allow_html=True)

        top_n_imp  = st.slider(t("trn_top_n"), 5, min(30, max(5, len(imp))), min(15, max(5, len(imp))), 1)
        imp_items  = list(imp.items())[:top_n_imp]
        feat_names = [i[0].replace("_enc"," (cat)").replace("_"," ") for i in imp_items]
        feat_vals  = [i[1] for i in imp_items]

        if feat_vals:
            fig_imp = go.Figure(go.Bar(x=feat_vals, y=feat_names, orientation="h",
                marker=dict(color=feat_vals, colorscale=[[0,"#06b6d4"],[0.5,"#e94560"],[1,"#ef4444"]]),
                text=[f"{v:.3f}" for v in feat_vals], textposition="outside",
                textfont=dict(color="#94a3b8", size=10)))
            fig_imp.update_layout(**PLOTLY_THEME, height=max(300, top_n_imp*28),
                                  yaxis=dict(categoryorder="total ascending"),
                                  xaxis=dict(range=[0, max(feat_vals)*1.15]))
            st.plotly_chart(fig_imp, width="stretch")
        else:
            st.info(t("trn_no_importance"))

        saved_w = m.get("feature_weights_used", {})
        if saved_w:
            st.markdown(t("trn_compare_title"))
            st.markdown(f"<p style='color:#6b7280;font-size:0.85rem;'>{t('trn_compare_sub')}</p>", unsafe_allow_html=True)
            compare_data = []
            for feat, imp_val in list(imp.items())[:20]:
                base_feat = feat.replace("_enc","")
                weight    = saved_w.get(base_feat, saved_w.get(feat, 1.0))
                compare_data.append({"Feature": feat.replace("_enc"," (cat)").replace("_"," "),
                                     t("trn_set_weight"): weight,
                                     t("trn_model_importance"): imp_val})
            df_compare = pd.DataFrame(compare_data)
            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Bar(name=t("trn_set_weight"), x=df_compare["Feature"],
                                     y=df_compare[t("trn_set_weight")], marker_color="#06b6d4", opacity=0.8))
            fig_cmp.add_trace(go.Bar(name=t("trn_model_importance"), x=df_compare["Feature"],
                                     y=df_compare[t("trn_model_importance")]*10, marker_color="#e94560", opacity=0.8))
            fig_cmp.update_layout(**PLOTLY_THEME, barmode="group", height=360,
                                  xaxis=dict(tickangle=-35), legend=dict(font=dict(color="#94a3b8")),
                                  annotations=[dict(text=t("trn_compare_note"), xref="paper", yref="paper",
                                                    x=0, y=-0.25, showarrow=False, font=dict(size=10, color="#6b7280"))])
            st.plotly_chart(fig_cmp, width="stretch")

        st.markdown(t("trn_cm_title"))
        cm_data = m.get("confusion_matrix", [[0,0],[0,0]])
        labels  = ["Stay (0)", "Churn (1)"]
        fig_cm  = go.Figure(go.Heatmap(z=cm_data, x=labels, y=labels,
            colorscale=[[0,"#13151c"],[1,"#e94560"]],
            text=[[str(v) for v in row] for row in cm_data],
            texttemplate="%{text}", textfont=dict(size=18, color="white"), showscale=False))
        fig_cm.update_layout(**PLOTLY_THEME, height=280,
                             xaxis=dict(title=t("trn_predicted")),
                             yaxis=dict(title=t("trn_actual"), autorange="reversed"))
        st.plotly_chart(fig_cm, width="stretch")

        st.markdown(f"<div style='color:#6b7280;font-size:0.8rem;'>Train: {m['n_train']} | Test: {m['n_test']} | Churn rate: {m['churn_rate_train']:.1%}</div>", unsafe_allow_html=True)
        st.markdown("---")
        st.download_button(t("trn_export_metrics"), data=json.dumps(m, indent=2).encode(),
                           file_name="training_metrics.json", mime="application/json")
