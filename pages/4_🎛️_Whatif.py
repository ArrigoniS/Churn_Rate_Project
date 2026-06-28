"""
Pagina 4 — What-if Simulator
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_trainer  import load_training_artifacts
from core.schema_manager import load_schema
from utils.i18n          import t, render_lang_toggle

st.set_page_config(page_title="What-if — HR Churn", page_icon="🎛️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Sans:wght@300;400;500&family=DM+Mono&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1,h2,h3 { font-family: 'Syne', sans-serif !important; }
.block-container { padding-top: 1.5rem !important; max-width: 1300px !important; }
.stButton > button { background:#e94560 !important; color:white !important;
    border:none !important; border-radius:8px !important; font-weight:600 !important; }
[data-testid="stSidebar"] { background: #13151c !important; }
.emp-card { background:#13151c; border:1px solid #252836; border-radius:10px; padding:1rem 1.2rem; }
.badge { display:inline-block; padding:3px 10px; border-radius:20px;
         font-size:0.72rem; font-weight:600; letter-spacing:0.05em; text-transform:uppercase; }
.badge-critico { background:rgba(239,68,68,0.15);  color:#ef4444; border:1px solid rgba(239,68,68,0.3); }
.badge-alto    { background:rgba(249,115,22,0.15); color:#f97316; border:1px solid rgba(249,115,22,0.3); }
.badge-medio   { background:rgba(234,179,8,0.15);  color:#eab308; border:1px solid rgba(234,179,8,0.3); }
.badge-basso   { background:rgba(34,197,94,0.15);  color:#22c55e; border:1px solid rgba(34,197,94,0.3); }
</style>
""", unsafe_allow_html=True)

render_lang_toggle()

def risk_label(key):
    from utils.i18n import get_lang
    lang = get_lang()
    mapping = {
        "it": {"Critico":"Critico","Alto":"Alto","Medio":"Medio","Basso":"Basso"},
        "en": {"Critico":"Critical","Alto":"High","Medio":"Medium","Basso":"Low"},
    }
    return mapping[lang].get(key, key)

def badge_html(risk):
    translated = risk_label(risk)
    css = {"Critico":"critico","Alto":"alto","Medio":"medio","Basso":"basso"}.get(risk,"basso")
    return f'<span class="badge badge-{css}">{translated}</span>'

def score_color(s):
    if s >= 75: return "#ef4444"
    if s >= 55: return "#f97316"
    if s >= 30: return "#eab308"
    return "#22c55e"

artifacts = load_training_artifacts()
schema    = load_schema()

st.markdown(t("wif_title"))
st.markdown(f"<p style='color:#6b7280;'>{t('wif_sub')}</p>", unsafe_allow_html=True)

if not artifacts:
    st.error(t("no_model_error"))
    st.stop()
if not os.path.exists("data/hr_scored.csv"):
    st.error(t("no_scored_error"))
    st.stop()

df_scored = pd.read_csv("data/hr_scored.csv")
st.markdown("---")

def rescore_employee(emp_dict):
    model = artifacts["model"]
    encoders = artifacts["encoders"]
    feature_cols = artifacts["feature_cols"]
    row = emp_dict.copy()
    for col, le in encoders.items():
        if col in row:
            known = set(le.classes_)
            row[col + "_enc"] = le.transform([str(row[col])])[0] if str(row[col]) in known else 0
    for c in [c for c in feature_cols if c not in row]:
        row[c] = 0
    X = np.array([[row[c] for c in feature_cols]], dtype=float)
    proba = model.predict_proba(X)[0][1]
    score = round(proba * 100, 1)
    risk  = pd.cut([score], bins=[0,30,55,75,100], labels=["Basso","Medio","Alto","Critico"])[0]
    return score, str(risk)

at_risk = df_scored[df_scored["risk_level"].isin(["Critico","Alto","Medio"])].copy()
if at_risk.empty:
    st.info(t("wif_no_at_risk"))
    st.stop()

has_name = "nome" in at_risk.columns and "cognome" in at_risk.columns
options = [
    f"{r['nome']} {r['cognome']} — {r['dipartimento']} · {r['ruolo']} (Score: {r['churn_score']:.0f})"
    if has_name else
    f"{r['employee_id']} — {r['dipartimento']} · {r['ruolo']} (Score: {r['churn_score']:.0f})"
    for _, r in at_risk.iterrows()
]

selected_label = st.selectbox(t("wif_select"), options)
selected_idx   = options.index(selected_label)
base           = at_risk.iloc[selected_idx].to_dict()
base_score     = float(base.get("churn_score", 0))
base_risk      = str(base.get("risk_level", "Medio"))
st.markdown("---")

col_base, col_sim, col_result = st.columns([1, 1.4, 1])

with col_base:
    st.markdown(t("wif_current_profile"))
    nome_display = f"{base.get('nome','')} {base.get('cognome','')}".strip() or base.get("employee_id","")
    st.markdown(
        f"<div class='emp-card' style='border-left:3px solid {score_color(base_score)};'>"
        f"<div style='font-family:Syne,sans-serif;font-weight:700;font-size:1rem;'>{nome_display}</div>"
        f"<div style='font-size:0.82rem;color:#94a3b8;margin-bottom:0.6rem;'>{base.get('ruolo','')} · {base.get('dipartimento','')}</div>"
        f"<div style='font-family:DM Mono,monospace;font-size:2rem;font-weight:800;color:{score_color(base_score)};'>{base_score:.0f}"
        f"<span style='font-size:1rem;color:#6b7280;'>/100</span></div>"
        f"{badge_html(base_risk)}</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    for label, val in [
        (t("wif_salary_lbl"),  f"€{base.get('stipendio_annuo',0):,}"),
        (t("wif_hours_lbl"),   f"{base.get('ore_settimanali_medie',0):.0f}h"),
        (t("wif_promo_lbl"),   f"{base.get('anni_senza_promozione',0):.1f}"),
        (t("wif_growth_lbl"),  f"{base.get('crescita_stipendio_percentuale',0):.1f}%"),
        (t("wif_manager_lbl"), f"{base.get('manager_score',0):.1f}/10"),
        ("eNPS",               f"{base.get('enps_score',0):.1f}/10"),
        (t("wif_checkin_lbl"), f"{base.get('giorni_ultimo_checkin_hr',0)} {t('wif_days_ago')}"),
    ]:
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;padding:4px 0;"
            f"border-bottom:1px solid #1a1d27;font-size:0.85rem;'>"
            f"<span style='color:#6b7280;'>{label}</span><span style='font-weight:500;'>{val}</span></div>",
            unsafe_allow_html=True)

with col_sim:
    st.markdown(t("wif_simulate"))
    new_stipendio  = st.slider(t("wif_salary"), int(base.get("stipendio_annuo",20000)*0.8), int(base.get("stipendio_annuo",20000)*1.6), int(base.get("stipendio_annuo",30000)), 500)
    new_ore        = st.slider(t("wif_hours"), 25.0, 70.0, float(base.get("ore_settimanali_medie",40)), 1.0)
    new_anni_promo = st.slider(t("wif_promo"), 0.0, 8.0, float(base.get("anni_senza_promozione",2)), 0.5, help=t("wif_promo_help"))
    new_crescita   = st.slider(t("wif_salary_growth"), -5.0, 20.0, float(base.get("crescita_stipendio_percentuale",3)), 0.5)
    new_manager    = st.slider(t("wif_manager"), 1.0, 10.0, float(base.get("manager_score",6)), 0.5)
    new_enps       = st.slider(t("wif_enps"), 1.0, 10.0, float(base.get("enps_score",6)), 0.5)
    new_checkin    = st.slider(t("wif_checkin"), 0, 365, int(base.get("giorni_ultimo_checkin_hr",90)), 7, help=t("wif_checkin_help"))

sim = base.copy()
sim.update({"stipendio_annuo":new_stipendio,"ore_settimanali_medie":new_ore,
            "anni_senza_promozione":new_anni_promo,"crescita_stipendio_percentuale":new_crescita,
            "manager_score":new_manager,"enps_score":new_enps,"giorni_ultimo_checkin_hr":new_checkin})
sim_score, sim_risk = rescore_employee(sim)
delta = sim_score - base_score
delta_color = "#22c55e" if delta < 0 else "#ef4444"
delta_icon  = "▼" if delta < 0 else "▲"

with col_result:
    st.markdown(t("wif_result"))
    st.markdown(
        f"<div class='emp-card' style='border-left:3px solid {score_color(sim_score)};text-align:center;'>"
        f"<div style='font-size:0.72rem;color:#6b7280;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;'>{t('wif_after')}</div>"
        f"<div style='font-family:DM Mono,monospace;font-size:2.4rem;font-weight:800;color:{score_color(sim_score)};'>{sim_score:.0f}"
        f"<span style='font-size:1rem;color:#6b7280;'>/100</span></div>"
        f"<div style='margin:0.5rem 0;'>{badge_html(sim_risk)}</div>"
        f"<div style='font-family:DM Mono,monospace;font-size:1.6rem;font-weight:800;color:{delta_color};margin-top:0.8rem;'>{delta_icon} {abs(delta):.1f} pt</div>"
        f"<div style='font-size:0.72rem;color:#6b7280;'>{t('wif_variation')}</div></div>",
        unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if delta <= -20:   st.success(t("wif_very_effective"))
    elif delta <= -10: st.success(t("wif_good"))
    elif delta <= -5:  st.info(t("wif_partial"))
    elif delta >= 5:   st.error(t("wif_worse"))
    else:              st.info(t("wif_limited"))
    if base_risk != sim_risk:
        st.markdown(
            f"<div style='margin-top:0.8rem;text-align:center;'>"
            f"<span style='font-size:0.78rem;color:#6b7280;'>{t('wif_risk_level')}</span><br>"
            f"{badge_html(base_risk)} → {badge_html(sim_risk)}</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown(t("wif_summary_title"))
changes = []
for label, orig, new, fmt in [
    (t("wif_salary"),        base.get("stipendio_annuo",0),               new_stipendio,   lambda v: f"€{v:,}"),
    (t("wif_hours"),         base.get("ore_settimanali_medie",0),          new_ore,         lambda v: f"{v:.0f}h"),
    (t("wif_promo"),         base.get("anni_senza_promozione",0),          new_anni_promo,  lambda v: f"{v:.1f}"),
    (t("wif_salary_growth"), base.get("crescita_stipendio_percentuale",0), new_crescita,    lambda v: f"{v:.1f}%"),
    (t("wif_manager"),       base.get("manager_score",0),                  new_manager,     lambda v: f"{v:.1f}/10"),
    (t("wif_enps"),          base.get("enps_score",0),                     new_enps,        lambda v: f"{v:.1f}/10"),
    (t("wif_checkin"),       base.get("giorni_ultimo_checkin_hr",0),       new_checkin,     lambda v: f"{v} {t('wif_days_ago')}"),
]:
    if abs(float(orig) - float(new)) > 0.01:
        changes.append({t("wif_col_intervention"):label, t("wif_col_before"):fmt(orig), t("wif_col_after"):fmt(new)})

if changes:
    st.dataframe(pd.DataFrame(changes), width="stretch", hide_index=True)
    st.markdown(
        f"<div style='background:#13151c;border:1px solid #252836;border-radius:10px;"
        f"padding:1rem 1.5rem;display:flex;justify-content:space-around;align-items:center;margin-top:0.5rem;'>"
        f"<div style='text-align:center;'><div style='font-size:0.72rem;color:#6b7280;'>{t('wif_score_current')}</div>"
        f"<div style='font-family:DM Mono,monospace;font-size:1.8rem;font-weight:800;color:{score_color(base_score)};'>{base_score:.0f}</div></div>"
        f"<div style='font-size:2rem;color:#6b7280;'>→</div>"
        f"<div style='text-align:center;'><div style='font-size:0.72rem;color:#6b7280;'>{t('wif_score_sim')}</div>"
        f"<div style='font-family:DM Mono,monospace;font-size:1.8rem;font-weight:800;color:{score_color(sim_score)};'>{sim_score:.0f}</div></div>"
        f"<div style='text-align:center;'><div style='font-size:0.72rem;color:#6b7280;'>{t('wif_variation_lbl')}</div>"
        f"<div style='font-family:DM Mono,monospace;font-size:1.8rem;font-weight:800;color:{delta_color};'>{delta_icon}{abs(delta):.1f}</div></div>"
        f"<div style='text-align:center;'><div style='font-size:0.72rem;color:#6b7280;'>{t('wif_risk_lbl')}</div>"
        f"<div style='margin-top:4px;'>{badge_html(base_risk)} → {badge_html(sim_risk)}</div></div></div>",
        unsafe_allow_html=True)
else:
    st.caption(t("wif_no_changes"))
