"""
HR Churn Predictor — Landing Page
"""

import os
import streamlit as st

st.set_page_config(
    page_title="HR Churn Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── i18n ──
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.i18n import t, render_lang_toggle
render_lang_toggle()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&family=DM+Mono:wght@400;500&display=swap');
:root {
    --bg:#0d0f14; --bg2:#13151c; --bg3:#1a1d27; --border:#252836;
    --accent:#e94560; --teal:#06b6d4; --green:#22c55e; --orange:#f97316;
    --muted:#6b7280; --text:#e2e8f0; --text2:#94a3b8;
}
html, body, [class*="css"] { font-family:'DM Sans',sans-serif !important; background:var(--bg) !important; color:var(--text) !important; }
h1,h2,h3,h4 { font-family:'Syne',sans-serif !important; }
.block-container { padding:0 !important; max-width:100% !important; }
header[data-testid="stHeader"] { background:var(--bg) !important; border-bottom:1px solid var(--border); }
[data-testid="stSidebar"] { background:var(--bg2) !important; }

.hero { background:linear-gradient(135deg,#0d0f14 0%,#13151c 50%,#0d0f14 100%);
    border-bottom:1px solid var(--border); padding:5rem 2rem 4rem; text-align:center; position:relative; overflow:hidden; }
.hero::before { content:''; position:absolute; inset:0;
    background:radial-gradient(ellipse 60% 50% at 50% 0%,rgba(233,69,96,0.12) 0%,transparent 70%); pointer-events:none; }
.hero-badge { display:inline-block; background:rgba(233,69,96,0.1); border:1px solid rgba(233,69,96,0.3);
    color:#e94560; font-size:0.75rem; font-weight:600; letter-spacing:0.12em; text-transform:uppercase;
    padding:5px 16px; border-radius:20px; margin-bottom:1.5rem; }
.hero-title { font-size:clamp(2.5rem,5vw,4rem); font-weight:800; line-height:1.1; letter-spacing:-0.03em;
    margin-bottom:1.2rem; background:linear-gradient(135deg,#e2e8f0 0%,#94a3b8 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }
.hero-title span { background:linear-gradient(135deg,#e94560,#f97316);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }
.hero-sub { font-size:1.15rem; color:var(--text2); max-width:600px; margin:0 auto 2.5rem; line-height:1.7; font-weight:300; }
.hero-chips { display:flex; gap:0.6rem; justify-content:center; flex-wrap:wrap; margin-bottom:3rem; }
.chip { background:var(--bg3); border:1px solid var(--border); color:var(--text2);
    font-size:0.78rem; font-family:'DM Mono',monospace; padding:4px 14px; border-radius:20px; }

.stats-bar { display:flex; justify-content:center; gap:3rem; padding:2rem;
    background:var(--bg2); border-top:1px solid var(--border); border-bottom:1px solid var(--border); }
.stat-item { text-align:center; }
.stat-num { font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; color:var(--accent); }
.stat-lbl { font-size:0.78rem; color:var(--muted); margin-top:2px; }

.section { padding:4rem 3rem; max-width:1100px; margin:0 auto; }
.section-label { font-size:0.72rem; font-weight:700; letter-spacing:0.15em; text-transform:uppercase; color:var(--accent); margin-bottom:0.6rem; }
.section-title { font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; letter-spacing:-0.02em; margin-bottom:0.8rem; color:var(--text); }
.section-sub { font-size:1rem; color:var(--text2); line-height:1.7; max-width:620px; margin-bottom:2.5rem; font-weight:300; }

.features-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:1.2rem; margin-top:1rem; }
.feat-card { background:var(--bg2); border:1px solid var(--border); border-radius:14px; padding:1.6rem; transition:border-color 0.2s,transform 0.2s; }
.feat-card:hover { border-color:var(--accent); transform:translateY(-2px); }
.feat-icon { font-size:1.8rem; margin-bottom:0.8rem; }
.feat-title { font-family:'Syne',sans-serif; font-size:1rem; font-weight:700; margin-bottom:0.4rem; color:var(--text); }
.feat-desc { font-size:0.85rem; color:var(--text2); line-height:1.6; }

.steps-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin-top:1rem; }
.step-card { background:var(--bg2); border:1px solid var(--border); border-radius:14px; padding:1.6rem; }
.step-card.done { border-color:var(--green); }
.step-card.done .step-num { color:var(--green); }
.step-num { font-family:'DM Mono',monospace; font-size:0.72rem; font-weight:700; letter-spacing:0.12em; color:var(--accent); text-transform:uppercase; margin-bottom:0.5rem; }
.step-icon { font-size:1.6rem; margin-bottom:0.6rem; }
.step-title { font-family:'Syne',sans-serif; font-size:0.95rem; font-weight:700; margin-bottom:0.4rem; color:var(--text); }
.step-desc { font-size:0.82rem; color:var(--text2); line-height:1.5; }
.step-status { margin-top:0.8rem; font-size:0.75rem; font-weight:600; }
.status-ok { color:var(--green); } .status-no { color:var(--orange); }

.philosophy-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:1.2rem; }
.philo-card { background:var(--bg2); border:1px solid var(--border); border-radius:14px; padding:1.8rem; border-top:3px solid transparent; }
.philo-card.p1 { border-top-color:var(--accent); }
.philo-card.p2 { border-top-color:var(--teal); }
.philo-card.p3 { border-top-color:var(--green); }
.philo-icon { font-size:2rem; margin-bottom:1rem; }
.philo-title { font-family:'Syne',sans-serif; font-size:1.05rem; font-weight:700; margin-bottom:0.5rem; }
.philo-desc { font-size:0.85rem; color:var(--text2); line-height:1.65; }

.stack-row { display:flex; flex-wrap:wrap; gap:0.8rem; margin-top:1rem; }
.stack-badge { background:var(--bg3); border:1px solid var(--border); border-radius:8px; padding:0.5rem 1rem;
    font-size:0.82rem; font-family:'DM Mono',monospace; color:var(--text2); display:flex; align-items:center; gap:6px; }
.stack-badge span { color:var(--text); font-weight:500; }
.section-divider { border:none; border-top:1px solid var(--border); margin:0; }
.footer { background:var(--bg2); border-top:1px solid var(--border); padding:2rem 3rem; text-align:center; color:var(--muted); font-size:0.82rem; }
</style>
""", unsafe_allow_html=True)

# ── Stato progetto ──
has_dataset = (st.session_state.get("df_train_raw") is not None or os.path.exists("data/hr_dataset.csv"))
has_model   = (st.session_state.get("training_artifacts") is not None or os.path.exists("models/churn_model.pkl"))
has_scored  = (st.session_state.get("df_scored") is not None or os.path.exists("data/hr_scored.csv"))

# ── HERO ──
st.markdown(f"""
<div class='hero'>
    <div class='hero-badge'>{t('hero_badge')}</div>
    <div class='hero-title'>
        {t('hero_title_1')}<br><span>{t('hero_title_2')}</span><br>{t('hero_title_3')}
    </div>
    <div class='hero-sub'>{t('hero_sub')}</div>
    <div class='hero-chips'>
        <div class='chip'>XGBoost</div>
        <div class='chip'>LiteLLM</div>
        <div class='chip'>Privacy-first</div>
        <div class='chip'>GDPR-ready</div>
        <div class='chip'>100% free</div>
        <div class='chip'>No vendor lock-in</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── STATS BAR ──
st.markdown(f"""
<div class='stats-bar'>
    <div class='stat-item'><div class='stat-num'>15+</div><div class='stat-lbl'>{t('stat_features')}</div></div>
    <div class='stat-item'><div class='stat-num'>100+</div><div class='stat-lbl'>{t('stat_providers')}</div></div>
    <div class='stat-item'><div class='stat-num'>4</div><div class='stat-lbl'>{t('stat_levels')}</div></div>
    <div class='stat-item'><div class='stat-num'>0€</div><div class='stat-lbl'>{t('stat_cost')}</div></div>
</div>
""", unsafe_allow_html=True)

# ── PROBLEMA ──
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.markdown(f"""
<div class='section'>
    <div class='section-label'>{t('problem_label')}</div>
    <div class='section-title'>{t('problem_title')}</div>
    <div class='section-sub'>{t('problem_sub')}</div>
    <div class='features-grid'>
        <div class='feat-card'><div class='feat-icon'>📉</div><div class='feat-title'>{t('prob_card1_title')}</div><div class='feat-desc'>{t('prob_card1_desc')}</div></div>
        <div class='feat-card'><div class='feat-icon'>⏳</div><div class='feat-title'>{t('prob_card2_title')}</div><div class='feat-desc'>{t('prob_card2_desc')}</div></div>
        <div class='feat-card'><div class='feat-icon'>🔍</div><div class='feat-title'>{t('prob_card3_title')}</div><div class='feat-desc'>{t('prob_card3_desc')}</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── COME FUNZIONA ──
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

s1_done = "done" if has_dataset else ""
s2_done = "done" if has_model   else ""
s3_done = "done" if has_scored  else ""

def step_status(ok):
    if ok: return f"<div class='step-status status-ok'>✓ {t('step_ready')}</div>"
    return f"<div class='step-status status-no'>◦ {t('step_waiting')}</div>"

st.markdown(f"""
<div class='section'>
    <div class='section-label'>{t('how_label')}</div>
    <div class='section-title'>{t('how_title')}</div>
    <div class='section-sub'>{t('how_sub')}</div>
    <div class='steps-grid'>
        <div class='step-card {s1_done}'>
            <div class='step-num'>Step 01</div><div class='step-icon'>⚙️</div>
            <div class='step-title'>{t('step1_title')}</div>
            <div class='step-desc'>{t('step1_desc')}</div>
            {step_status(has_dataset)}
        </div>
        <div class='step-card {s2_done}'>
            <div class='step-num'>Step 02</div><div class='step-icon'>🏋️</div>
            <div class='step-title'>{t('step2_title')}</div>
            <div class='step-desc'>{t('step2_desc')}</div>
            {step_status(has_model)}
        </div>
        <div class='step-card {s3_done}'>
            <div class='step-num'>Step 03</div><div class='step-icon'>🔮</div>
            <div class='step-title'>{t('step3_title')}</div>
            <div class='step-desc'>{t('step3_desc')}</div>
            {step_status(has_scored)}
        </div>
        <div class='step-card'>
            <div class='step-num'>Step 04</div><div class='step-icon'>🎛️</div>
            <div class='step-title'>{t('step4_title')}</div>
            <div class='step-desc'>{t('step4_desc')}</div>
            <div class='step-status' style='color:var(--teal);'>◦ {t('step_always')}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── FILOSOFIA ──
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.markdown(f"""
<div class='section'>
    <div class='section-label'>{t('philo_label')}</div>
    <div class='section-title'>{t('philo_title')}</div>
    <div class='section-sub'>{t('philo_sub')}</div>
    <div class='philosophy-grid'>
        <div class='philo-card p1'><div class='philo-icon'>🎯</div><div class='philo-title'>{t('philo1_title')}</div><div class='philo-desc'>{t('philo1_desc')}</div></div>
        <div class='philo-card p2'><div class='philo-icon'>🔐</div><div class='philo-title'>{t('philo2_title')}</div><div class='philo-desc'>{t('philo2_desc')}</div></div>
        <div class='philo-card p3'><div class='philo-icon'>🔓</div><div class='philo-title'>{t('philo3_title')}</div><div class='philo-desc'>{t('philo3_desc')}</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── FEATURE ──
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.markdown(f"""
<div class='section'>
    <div class='section-label'>{t('features_label')}</div>
    <div class='section-title'>{t('features_title')}</div>
    <div class='section-sub'>{t('features_sub')}</div>
    <div class='features-grid'>
        <div class='feat-card'><div class='feat-icon'>💼</div><div class='feat-title'>{t('feat_career')}</div><div class='feat-desc'>{t('feat_career_desc')}</div></div>
        <div class='feat-card'><div class='feat-icon'>⏱️</div><div class='feat-title'>{t('feat_workload')}</div><div class='feat-desc'>{t('feat_workload_desc')}</div></div>
        <div class='feat-card'><div class='feat-icon'>💬</div><div class='feat-title'>{t('feat_engagement')}</div><div class='feat-desc'>{t('feat_engagement_desc')}</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── STACK ──
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.markdown(f"""
<div class='section'>
    <div class='section-label'>{t('stack_label')}</div>
    <div class='section-title'>{t('stack_title')}</div>
    <div class='section-sub'>{t('stack_sub')}</div>
    <div class='stack-row'>
        <div class='stack-badge'>🤖 <span>XGBoost</span></div>
        <div class='stack-badge'>🔗 <span>LiteLLM</span></div>
        <div class='stack-badge'>🎨 <span>Streamlit</span></div>
        <div class='stack-badge'>📄 <span>ReportLab</span></div>
        <div class='stack-badge'>📊 <span>openpyxl</span></div>
        <div class='stack-badge'>🐍 <span>scikit-learn</span></div>
        <div class='stack-badge'>🎭 <span>Faker</span></div>
        <div class='stack-badge'>📈 <span>Plotly</span></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── CTA ──
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

if has_dataset and has_model and has_scored:
    cta_text, cta_sub, cta_icon = t("cta_all_ready"), t("cta_all_ready_sub"), "🔮"
elif has_dataset and has_model:
    cta_text, cta_sub, cta_icon = t("cta_model_ready"), t("cta_model_ready_sub"), "🎯"
elif has_dataset:
    cta_text, cta_sub, cta_icon = t("cta_dataset_ready"), t("cta_dataset_ready_sub"), "🏋️"
else:
    cta_text, cta_sub, cta_icon = t("cta_start"), t("cta_start_sub"), "⚙️"

st.markdown(f"""
<div style='background:linear-gradient(135deg,#13151c,#1a1d27);border-top:1px solid #252836;border-bottom:1px solid #252836;padding:4rem 2rem;text-align:center;'>
    <div style='font-size:2.8rem;margin-bottom:1rem;'>{cta_icon}</div>
    <div style='font-family:Syne,sans-serif;font-size:1.8rem;font-weight:800;letter-spacing:-0.02em;margin-bottom:0.6rem;'>{cta_text}</div>
    <div style='color:#6b7280;font-size:0.95rem;margin-bottom:2rem;'>{cta_sub}</div>
    <div style='color:#4b5563;font-size:0.82rem;'>{t('cta_nav_hint')}</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='footer'>
    HR Churn Predictor · Built with Streamlit · 100% open source · Privacy-first · GDPR-ready
</div>
""", unsafe_allow_html=True)
