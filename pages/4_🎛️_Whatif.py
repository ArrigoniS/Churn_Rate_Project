"""
Pagina 4 — What-if Simulator
Seleziona un dipendente a rischio e simula interventi HR.
Lo score si ricalcola in tempo reale senza riaddestare il modello.
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_trainer  import load_training_artifacts
from core.schema_manager import load_schema, get_categorical_features

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="What-if — HR Churn",
    page_icon="🎛️",
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
.emp-card {
    background: #13151c; border: 1px solid #252836;
    border-radius: 10px; padding: 1rem 1.2rem;
}
.badge { display:inline-block; padding:3px 10px; border-radius:20px;
         font-size:0.72rem; font-weight:600; letter-spacing:0.05em; text-transform:uppercase; }
.badge-critico { background:rgba(239,68,68,0.15);  color:#ef4444; border:1px solid rgba(239,68,68,0.3); }
.badge-alto    { background:rgba(249,115,22,0.15); color:#f97316; border:1px solid rgba(249,115,22,0.3); }
.badge-medio   { background:rgba(234,179,8,0.15);  color:#eab308; border:1px solid rgba(234,179,8,0.3); }
.badge-basso   { background:rgba(34,197,94,0.15);  color:#22c55e; border:1px solid rgba(34,197,94,0.3); }
</style>
""", unsafe_allow_html=True)

RISK_COLORS = {"Critico":"#ef4444","Alto":"#f97316","Medio":"#eab308","Basso":"#22c55e"}

def badge_html(risk):
    return f'<span class="badge badge-{risk.lower()}">{risk}</span>'

def score_color(s):
    if s >= 75: return "#ef4444"
    if s >= 55: return "#f97316"
    if s >= 30: return "#eab308"
    return "#22c55e"

# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────
artifacts = load_training_artifacts()
schema    = load_schema()

st.markdown("# 🎛️ What-if Simulator")
st.markdown(
    "<p style='color:#6b7280;'>Seleziona un dipendente a rischio e simula interventi HR. "
    "Lo score si ricalcola in tempo reale senza riaddestare il modello.</p>",
    unsafe_allow_html=True,
)

if not artifacts:
    st.error("❌ Nessun modello trovato. Vai alla pagina **🏋️ Training** prima.")
    st.stop()

if not os.path.exists("data/hr_scored.csv"):
    st.error("❌ Nessun dataset scorato trovato. Vai alla pagina **🔮 Analisi** e genera/carica un dataset prima.")
    st.stop()

df_scored = pd.read_csv("data/hr_scored.csv")
st.markdown("---")

# ─────────────────────────────────────────────
# SCORING HELPER
# ─────────────────────────────────────────────
def rescore_employee(emp_dict: dict) -> tuple:
    """Ricalcola churn score per un singolo dipendente modificato."""
    model        = artifacts["model"]
    encoders     = artifacts["encoders"]
    feature_cols = artifacts["feature_cols"]

    row = emp_dict.copy()
    for col, le in encoders.items():
        if col in row:
            known = set(le.classes_)
            row[col + "_enc"] = le.transform([str(row[col])])[0] if str(row[col]) in known else 0

    missing = [c for c in feature_cols if c not in row]
    for c in missing:
        row[c] = 0

    X = np.array([[row[c] for c in feature_cols]], dtype=float)
    proba = model.predict_proba(X)[0][1]
    score = round(proba * 100, 1)
    risk  = pd.cut([score], bins=[0,30,55,75,100],
                   labels=["Basso","Medio","Alto","Critico"])[0]
    return score, str(risk)


# ─────────────────────────────────────────────
# SELEZIONE DIPENDENTE
# ─────────────────────────────────────────────
at_risk = df_scored[df_scored["risk_level"].isin(["Critico","Alto","Medio"])].copy()

if at_risk.empty:
    st.info("Nessun dipendente a rischio Medio/Alto/Critico nel dataset.")
    st.stop()

has_name = "nome" in at_risk.columns and "cognome" in at_risk.columns
if has_name:
    options = [
        f"{r['nome']} {r['cognome']} — {r['dipartimento']} · {r['ruolo']} (Score: {r['churn_score']:.0f})"
        for _, r in at_risk.iterrows()
    ]
else:
    options = [
        f"{r['employee_id']} — {r['dipartimento']} · {r['ruolo']} (Score: {r['churn_score']:.0f})"
        for _, r in at_risk.iterrows()
    ]

selected_label = st.selectbox("👤 Seleziona dipendente", options)
selected_idx   = options.index(selected_label)
base           = at_risk.iloc[selected_idx].to_dict()
base_score     = float(base.get("churn_score", 0))
base_risk      = str(base.get("risk_level", "Medio"))

st.markdown("---")

# ─────────────────────────────────────────────
# LAYOUT: profilo | sliders | risultato
# ─────────────────────────────────────────────
col_base, col_sim, col_result = st.columns([1, 1.4, 1])

# ── Profilo attuale ──
with col_base:
    st.markdown("##### Profilo attuale")
    nome_display = f"{base.get('nome','')} {base.get('cognome','')}".strip() or base.get("employee_id","")
    st.markdown(
        f"<div class='emp-card' style='border-left:3px solid {score_color(base_score)};'>"
        f"<div style='font-family:Syne,sans-serif; font-weight:700; font-size:1rem;'>{nome_display}</div>"
        f"<div style='font-size:0.82rem; color:#94a3b8; margin-bottom:0.6rem;'>"
        f"{base.get('ruolo','')} · {base.get('dipartimento','')}</div>"
        f"<div style='font-family:DM Mono,monospace; font-size:2rem; font-weight:800; "
        f"color:{score_color(base_score)};'>{base_score:.0f}"
        f"<span style='font-size:1rem; color:#6b7280;'>/100</span></div>"
        f"{badge_html(base_risk)}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    for label, val in [
        ("Stipendio",      f"€{base.get('stipendio_annuo', 0):,}"),
        ("Ore/sett",       f"{base.get('ore_settimanali_medie', 0):.0f}h"),
        ("Anni senza promo", f"{base.get('anni_senza_promozione', 0):.1f}"),
        ("Crescita stip.", f"{base.get('crescita_stipendio_percentuale', 0):.1f}%"),
        ("Manager score",  f"{base.get('manager_score', 0):.1f}/10"),
        ("eNPS",           f"{base.get('enps_score', 0):.1f}/10"),
        ("HR check-in",    f"{base.get('giorni_ultimo_checkin_hr', 0)} gg fa"),
    ]:
        st.markdown(
            f"<div style='display:flex; justify-content:space-between; padding:4px 0; "
            f"border-bottom:1px solid #1a1d27; font-size:0.85rem;'>"
            f"<span style='color:#6b7280;'>{label}</span>"
            f"<span style='font-weight:500;'>{val}</span></div>",
            unsafe_allow_html=True,
        )

# ── Sliders simulazione ──
with col_sim:
    st.markdown("##### Simula interventi HR")

    new_stipendio = st.slider(
        "💰 Stipendio annuo (€)",
        min_value=int(base.get("stipendio_annuo", 20000) * 0.8),
        max_value=int(base.get("stipendio_annuo", 20000) * 1.6),
        value=int(base.get("stipendio_annuo", 30000)),
        step=500,
    )
    new_ore = st.slider(
        "⏱️ Ore settimanali medie",
        min_value=25.0, max_value=70.0,
        value=float(base.get("ore_settimanali_medie", 40)),
        step=1.0,
    )
    new_anni_promo = st.slider(
        "📈 Anni dall'ultima promozione",
        min_value=0.0, max_value=8.0,
        value=float(base.get("anni_senza_promozione", 2)),
        step=0.5,
        help="Abbassa per simulare una promozione recente",
    )
    new_crescita = st.slider(
        "📊 Crescita stipendio % annua",
        min_value=-5.0, max_value=20.0,
        value=float(base.get("crescita_stipendio_percentuale", 3)),
        step=0.5,
    )
    new_manager = st.slider(
        "👤 Manager score",
        min_value=1.0, max_value=10.0,
        value=float(base.get("manager_score", 6)),
        step=0.5,
    )
    new_enps = st.slider(
        "💬 eNPS score",
        min_value=1.0, max_value=10.0,
        value=float(base.get("enps_score", 6)),
        step=0.5,
    )
    new_checkin = st.slider(
        "📅 Giorni dall'ultimo HR check-in",
        min_value=0, max_value=365,
        value=int(base.get("giorni_ultimo_checkin_hr", 90)),
        step=7,
        help="Abbassa per simulare un check-in recente",
    )

# ── Calcolo score simulato ──
sim = base.copy()
sim["stipendio_annuo"]                 = new_stipendio
sim["ore_settimanali_medie"]           = new_ore
sim["anni_senza_promozione"]           = new_anni_promo
sim["crescita_stipendio_percentuale"]  = new_crescita
sim["manager_score"]                   = new_manager
sim["enps_score"]                      = new_enps
sim["giorni_ultimo_checkin_hr"]        = new_checkin

sim_score, sim_risk = rescore_employee(sim)
delta       = sim_score - base_score
delta_color = "#22c55e" if delta < 0 else "#ef4444"
delta_icon  = "▼" if delta < 0 else "▲"

# ── Risultato ──
with col_result:
    st.markdown("##### Score simulato")
    st.markdown(
        f"<div class='emp-card' style='border-left:3px solid {score_color(sim_score)}; text-align:center;'>"
        f"<div style='font-size:0.72rem; color:#6b7280; text-transform:uppercase; "
        f"letter-spacing:0.08em; margin-bottom:6px;'>DOPO GLI INTERVENTI</div>"
        f"<div style='font-family:DM Mono,monospace; font-size:2.4rem; font-weight:800; "
        f"color:{score_color(sim_score)};'>{sim_score:.0f}"
        f"<span style='font-size:1rem; color:#6b7280;'>/100</span></div>"
        f"<div style='margin:0.5rem 0;'>{badge_html(sim_risk)}</div>"
        f"<div style='font-family:DM Mono,monospace; font-size:1.6rem; font-weight:800; "
        f"color:{delta_color}; margin-top:0.8rem;'>{delta_icon} {abs(delta):.1f} pt</div>"
        f"<div style='font-size:0.72rem; color:#6b7280;'>variazione score</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    if delta <= -20:
        st.success("🎯 Intervento molto efficace")
    elif delta <= -10:
        st.success("✓ Buon intervento — rischio in calo")
    elif delta <= -5:
        st.info("→ Intervento parziale")
    elif delta >= 5:
        st.error("⚠ Questo scenario peggiora la situazione")
    else:
        st.info("≈ Impatto limitato sullo score")

    # Cambio livello rischio
    if base_risk != sim_risk:
        st.markdown(
            f"<div style='margin-top:0.8rem; text-align:center;'>"
            f"<span style='font-size:0.78rem; color:#6b7280;'>Livello rischio</span><br>"
            f"{badge_html(base_risk)} → {badge_html(sim_risk)}"
            f"</div>",
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────
# RIEPILOGO INTERVENTI
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("##### Riepilogo interventi simulati")

changes = []
for label, key, orig, new, fmt in [
    ("💰 Stipendio",          "stipendio_annuo",                  base.get("stipendio_annuo",0),     new_stipendio,   lambda v: f"€{v:,}"),
    ("⏱️ Ore/sett",           "ore_settimanali_medie",            base.get("ore_settimanali_medie",0), new_ore,        lambda v: f"{v:.0f}h"),
    ("📈 Anni senza promo",   "anni_senza_promozione",            base.get("anni_senza_promozione",0), new_anni_promo, lambda v: f"{v:.1f}"),
    ("📊 Crescita stipendio", "crescita_stipendio_percentuale",   base.get("crescita_stipendio_percentuale",0), new_crescita, lambda v: f"{v:.1f}%"),
    ("👤 Manager score",      "manager_score",                    base.get("manager_score",0),        new_manager,    lambda v: f"{v:.1f}/10"),
    ("💬 eNPS",               "enps_score",                       base.get("enps_score",0),           new_enps,       lambda v: f"{v:.1f}/10"),
    ("📅 HR check-in",        "giorni_ultimo_checkin_hr",         base.get("giorni_ultimo_checkin_hr",0), new_checkin, lambda v: f"{v} gg"),
]:
    if abs(float(orig) - float(new)) > 0.01:
        changes.append({"Intervento": label, "Prima": fmt(orig), "Dopo": fmt(new)})

if changes:
    st.dataframe(pd.DataFrame(changes), width='stretch', hide_index=True)

    st.markdown(
        f"<div style='background:#13151c; border:1px solid #252836; border-radius:10px; "
        f"padding:1rem 1.5rem; display:flex; justify-content:space-around; "
        f"align-items:center; margin-top:0.5rem;'>"
        f"<div style='text-align:center;'>"
        f"<div style='font-size:0.72rem; color:#6b7280;'>SCORE ATTUALE</div>"
        f"<div style='font-family:DM Mono,monospace; font-size:1.8rem; font-weight:800; "
        f"color:{score_color(base_score)};'>{base_score:.0f}</div></div>"
        f"<div style='font-size:2rem; color:#6b7280;'>→</div>"
        f"<div style='text-align:center;'>"
        f"<div style='font-size:0.72rem; color:#6b7280;'>SCORE SIMULATO</div>"
        f"<div style='font-family:DM Mono,monospace; font-size:1.8rem; font-weight:800; "
        f"color:{score_color(sim_score)};'>{sim_score:.0f}</div></div>"
        f"<div style='text-align:center;'>"
        f"<div style='font-size:0.72rem; color:#6b7280;'>VARIAZIONE</div>"
        f"<div style='font-family:DM Mono,monospace; font-size:1.8rem; font-weight:800; "
        f"color:{delta_color};'>{delta_icon}{abs(delta):.1f}</div></div>"
        f"<div style='text-align:center;'>"
        f"<div style='font-size:0.72rem; color:#6b7280;'>RISCHIO</div>"
        f"<div style='margin-top:4px;'>{badge_html(base_risk)} → {badge_html(sim_risk)}</div>"
        f"</div></div>",
        unsafe_allow_html=True,
    )
else:
    st.caption("Muovi gli slider per simulare interventi HR.")
