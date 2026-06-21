"""
HR Churn Predictor — Landing Page
Homepage completa con hero, filosofia, flusso e stato progetto.
"""

import os
import streamlit as st

st.set_page_config(
    page_title="HR Churn Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&family=DM+Mono:wght@400;500&display=swap');

:root {
    --bg:      #0d0f14;
    --bg2:     #13151c;
    --bg3:     #1a1d27;
    --border:  #252836;
    --accent:  #e94560;
    --teal:    #06b6d4;
    --green:   #22c55e;
    --orange:  #f97316;
    --yellow:  #eab308;
    --muted:   #6b7280;
    --text:    #e2e8f0;
    --text2:   #94a3b8;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background: var(--bg) !important;
    color: var(--text) !important;
}
h1,h2,h3,h4 { font-family: 'Syne', sans-serif !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
header[data-testid="stHeader"] { background: var(--bg) !important;
    border-bottom: 1px solid var(--border); }

/* ── HERO ── */
.hero {
    background: linear-gradient(135deg, #0d0f14 0%, #13151c 50%, #0d0f14 100%);
    border-bottom: 1px solid var(--border);
    padding: 5rem 2rem 4rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse 60% 50% at 50% 0%,
        rgba(233,69,96,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-badge {
    display: inline-block;
    background: rgba(233,69,96,0.1);
    border: 1px solid rgba(233,69,96,0.3);
    color: #e94560;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 5px 16px;
    border-radius: 20px;
    margin-bottom: 1.5rem;
}
.hero-title {
    font-size: clamp(2.5rem, 5vw, 4rem);
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -0.03em;
    margin-bottom: 1.2rem;
    background: linear-gradient(135deg, #e2e8f0 0%, #94a3b8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-title span {
    background: linear-gradient(135deg, #e94560, #f97316);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 1.15rem;
    color: var(--text2);
    max-width: 600px;
    margin: 0 auto 2.5rem;
    line-height: 1.7;
    font-weight: 300;
}
.hero-chips {
    display: flex;
    gap: 0.6rem;
    justify-content: center;
    flex-wrap: wrap;
    margin-bottom: 3rem;
}
.chip {
    background: var(--bg3);
    border: 1px solid var(--border);
    color: var(--text2);
    font-size: 0.78rem;
    font-family: 'DM Mono', monospace;
    padding: 4px 14px;
    border-radius: 20px;
}

/* ── STATS BAR ── */
.stats-bar {
    display: flex;
    justify-content: center;
    gap: 3rem;
    padding: 2rem;
    background: var(--bg2);
    border-top: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
}
.stat-item { text-align: center; }
.stat-num {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: var(--accent);
}
.stat-lbl { font-size: 0.78rem; color: var(--muted); margin-top: 2px; }

/* ── SECTIONS ── */
.section { padding: 4rem 3rem; max-width: 1100px; margin: 0 auto; }
.section-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.6rem;
}
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin-bottom: 0.8rem;
    color: var(--text);
}
.section-sub {
    font-size: 1rem;
    color: var(--text2);
    line-height: 1.7;
    max-width: 620px;
    margin-bottom: 2.5rem;
    font-weight: 300;
}

/* ── FEATURE CARDS ── */
.features-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1.2rem;
    margin-top: 1rem;
}
.feat-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.6rem;
    transition: border-color 0.2s, transform 0.2s;
}
.feat-card:hover { border-color: var(--accent); transform: translateY(-2px); }
.feat-icon { font-size: 1.8rem; margin-bottom: 0.8rem; }
.feat-title {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 0.4rem;
    color: var(--text);
}
.feat-desc { font-size: 0.85rem; color: var(--text2); line-height: 1.6; }

/* ── HOW IT WORKS ── */
.steps-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    position: relative;
    margin-top: 1rem;
}
.step-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.6rem;
    position: relative;
}
.step-card.done { border-color: var(--green); }
.step-card.done .step-num { color: var(--green); }
.step-num {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.step-icon { font-size: 1.6rem; margin-bottom: 0.6rem; }
.step-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.95rem;
    font-weight: 700;
    margin-bottom: 0.4rem;
    color: var(--text);
}
.step-desc { font-size: 0.82rem; color: var(--text2); line-height: 1.5; }
.step-status {
    margin-top: 0.8rem;
    font-size: 0.75rem;
    font-weight: 600;
}
.status-ok  { color: var(--green); }
.status-no  { color: var(--orange); }

/* ── PHILOSOPHY ── */
.philosophy-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1.2rem;
}
.philo-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.8rem;
    border-top: 3px solid transparent;
}
.philo-card.p1 { border-top-color: var(--accent); }
.philo-card.p2 { border-top-color: var(--teal); }
.philo-card.p3 { border-top-color: var(--green); }
.philo-icon { font-size: 2rem; margin-bottom: 1rem; }
.philo-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}
.philo-desc { font-size: 0.85rem; color: var(--text2); line-height: 1.65; }

/* ── TECH STACK ── */
.stack-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.8rem;
    margin-top: 1rem;
}
.stack-badge {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-size: 0.82rem;
    font-family: 'DM Mono', monospace;
    color: var(--text2);
    display: flex;
    align-items: center;
    gap: 6px;
}
.stack-badge span { color: var(--text); font-weight: 500; }

/* ── DIVIDER ── */
.section-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 0;
}

/* ── FOOTER ── */
.footer {
    background: var(--bg2);
    border-top: 1px solid var(--border);
    padding: 2rem 3rem;
    text-align: center;
    color: var(--muted);
    font-size: 0.82rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# STATO PROGETTO
# ─────────────────────────────────────────────
has_dataset = os.path.exists("data/hr_dataset.csv")
has_model   = os.path.exists("models/churn_model.pkl")
has_scored  = os.path.exists("data/hr_scored.csv")


# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <div class='hero-badge'>HR Intelligence Platform</div>
    <div class='hero-title'>
        Scopri chi sta per<br><span>lasciare l'azienda</span><br>prima che accada
    </div>
    <div class='hero-sub'>
        Un sistema di machine learning che analizza i segnali deboli nei dati HR
        e identifica i dipendenti a rischio abbandono — con spiegazioni azionabili
        generate da AI e report pronti per il board.
    </div>
    <div class='hero-chips'>
        <div class='chip'>XGBoost</div>
        <div class='chip'>LiteLLM</div>
        <div class='chip'>Privacy-first</div>
        <div class='chip'>GDPR-ready</div>
        <div class='chip'>100% gratuito</div>
        <div class='chip'>No cloud vendor lock-in</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# STATS BAR
# ─────────────────────────────────────────────
st.markdown("""
<div class='stats-bar'>
    <div class='stat-item'>
        <div class='stat-num'>15+</div>
        <div class='stat-lbl'>Feature analizzate</div>
    </div>
    <div class='stat-item'>
        <div class='stat-num'>100+</div>
        <div class='stat-lbl'>Provider LLM supportati</div>
    </div>
    <div class='stat-item'>
        <div class='stat-num'>4</div>
        <div class='stat-lbl'>Livelli di rischio</div>
    </div>
    <div class='stat-item'>
        <div class='stat-num'>0€</div>
        <div class='stat-lbl'>Costo infrastruttura</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PROBLEMA CHE RISOLVE
# ─────────────────────────────────────────────
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.markdown("""
<div class='section'>
    <div class='section-label'>Il problema</div>
    <div class='section-title'>Il turnover costa più di quanto pensi</div>
    <div class='section-sub'>
        Sostituire un dipendente costa in media tra il 50% e il 200% del suo stipendio annuo.
        Il problema è che quando una persona si dimette, l'HR è già in ritardo di mesi.
        I segnali c'erano — nei dati.
    </div>
    <div class='features-grid'>
        <div class='feat-card'>
            <div class='feat-icon'>📉</div>
            <div class='feat-title'>Costo nascosto</div>
            <div class='feat-desc'>
                Recruiting, onboarding, perdita di know-how e calo di produttività
                del team durante la transizione — spesso ignorati nei budget HR.
            </div>
        </div>
        <div class='feat-card'>
            <div class='feat-icon'>⏳</div>
            <div class='feat-title'>Intervento tardivo</div>
            <div class='feat-desc'>
                L'HR scopre il disagio durante il colloquio di uscita.
                A quel punto il dipendente ha già firmato altrove
                e la decisione è irreversibile.
            </div>
        </div>
        <div class='feat-card'>
            <div class='feat-icon'>🔍</div>
            <div class='feat-title'>Segnali ignorati</div>
            <div class='feat-desc'>
                Overload, mancanza di promozioni, manager score in calo,
                check-in HR sempre più rari — i dati parlano, ma nessuno li ascolta
                in modo sistematico.
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# COME FUNZIONA
# ─────────────────────────────────────────────
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

step1_done = "done" if has_dataset else ""
step2_done = "done" if has_model   else ""
step3_done = "done" if has_scored  else ""

step1_status = "<div class='step-status status-ok'>✓ Dataset pronto</div>" if has_dataset else "<div class='step-status status-no'>◦ In attesa</div>"
step2_status = "<div class='step-status status-ok'>✓ Modello trainato</div>" if has_model else "<div class='step-status status-no'>◦ In attesa</div>"
step3_status = "<div class='step-status status-ok'>✓ Analisi disponibile</div>" if has_scored else "<div class='step-status status-no'>◦ In attesa</div>"

st.markdown(f"""
<div class='section'>
    <div class='section-label'>Come funziona</div>
    <div class='section-title'>Quattro passi per la retention intelligente</div>
    <div class='section-sub'>
        Segui il flusso nell'ordine indicato — ogni step sblocca il successivo.
        Puoi usare dati simulati per esplorare o caricare i tuoi dati reali.
    </div>
    <div class='steps-grid'>
        <div class='step-card {step1_done}'>
            <div class='step-num'>Step 01</div>
            <div class='step-icon'>⚙️</div>
            <div class='step-title'>Configurazione</div>
            <div class='step-desc'>
                Genera un dataset simulato con parametri personalizzabili
                oppure carica il tuo CSV. Aggiungi colonne custom allo schema
                e scarica il template.
            </div>
            {step1_status}
        </div>
        <div class='step-card {step2_done}'>
            <div class='step-num'>Step 02</div>
            <div class='step-icon'>🏋️</div>
            <div class='step-title'>Training</div>
            <div class='step-desc'>
                Imposta i pesi delle feature in base al tuo contesto aziendale.
                Il modello XGBoost si addestra sui tuoi dati e mostra
                feature importance e metriche ROC-AUC.
            </div>
            {step2_status}
        </div>
        <div class='step-card {step3_done}'>
            <div class='step-num'>Step 03</div>
            <div class='step-icon'>🔮</div>
            <div class='step-title'>Analisi</div>
            <div class='step-desc'>
                Ogni dipendente riceve uno score 0-100 e un livello di rischio.
                L'AI genera suggerimenti HR azionabili per i casi più critici.
                Scarica il report in PDF o Excel.
            </div>
            {step3_status}
        </div>
        <div class='step-card'>
            <div class='step-num'>Step 04</div>
            <div class='step-icon'>🎛️</div>
            <div class='step-title'>What-if</div>
            <div class='step-desc'>
                Simula interventi HR in tempo reale — promozione, riduzione ore,
                aumento stipendio — e vedi come cambia lo score prima di
                prendere una decisione.
            </div>
            <div class='step-status' style='color:var(--teal);'>◦ Sempre disponibile dopo l'analisi</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FILOSOFIA
# ─────────────────────────────────────────────
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.markdown("""
<div class='section'>
    <div class='section-label'>La filosofia</div>
    <div class='section-title'>Tre principi che guidano il progetto</div>
    <div class='section-sub'>
        Non è solo un tool di predizione — è un modo diverso di fare HR analytics.
    </div>
    <div class='philosophy-grid'>
        <div class='philo-card p1'>
            <div class='philo-icon'>🎯</div>
            <div class='philo-title'>ML + LLM insieme</div>
            <div class='philo-desc'>
                Il modello di machine learning dà uno <strong>score numerico</strong>
                preciso e riproducibile. Il modello linguistico traduce quel numero
                in <strong>linguaggio umano</strong> — cause, contesto, azioni concrete.
                Nessuno dei due funzionerebbe altrettanto bene da solo.
            </div>
        </div>
        <div class='philo-card p2'>
            <div class='philo-icon'>🔐</div>
            <div class='philo-title'>Privacy by design</div>
            <div class='philo-desc'>
                Nome, cognome e ID vengono <strong>rimossi prima di ogni chiamata LLM</strong>.
                Il modello AI riceve solo dati numerici e categorici anonimi.
                La de-anonimizzazione avviene in locale, dopo aver ricevuto la risposta.
                GDPR-ready by default.
            </div>
        </div>
        <div class='philo-card p3'>
            <div class='philo-icon'>🔓</div>
            <div class='philo-title'>Vendor agnostic</div>
            <div class='philo-desc'>
                Funziona con <strong>qualsiasi LLM</strong> — Anthropic, OpenAI, Gemini,
                Mistral, Ollama locale e altri 100+ provider tramite LiteLLM.
                L'azienda porta la propria API key e sceglie il fornitore
                in base a costi, compliance o preferenze.
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# COSA ANALIZZA
# ─────────────────────────────────────────────
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.markdown("""
<div class='section'>
    <div class='section-label'>Le feature</div>
    <div class='section-title'>Cosa analizza il modello</div>
    <div class='section-sub'>
        15+ segnali organizzati in 4 aree — tutte configurabili e estendibili
        con colonne custom dallo schema.
    </div>
    <div class='features-grid'>
        <div class='feat-card'>
            <div class='feat-icon'>💼</div>
            <div class='feat-title'>Carriera</div>
            <div class='feat-desc'>
                Anni senza promozione · Numero di promozioni ricevute ·
                Crescita stipendio media · RAL attuale
            </div>
        </div>
        <div class='feat-card'>
            <div class='feat-icon'>⏱️</div>
            <div class='feat-title'>Workload</div>
            <div class='feat-desc'>
                Ore settimanali medie · Ferie residue non godute ·
                Giorni di assenza · Numero progetti seguiti
            </div>
        </div>
        <div class='feat-card'>
            <div class='feat-icon'>💬</div>
            <div class='feat-title'>Engagement</div>
            <div class='feat-desc'>
                Manager score · eNPS aziendale ·
                Giorni dall'ultimo check-in HR
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TECH STACK
# ─────────────────────────────────────────────
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.markdown("""
<div class='section'>
    <div class='section-label'>Tech stack</div>
    <div class='section-title'>Tutto open source, tutto gratuito</div>
    <div class='section-sub'>
        Nessun costo di infrastruttura. L'unica spesa è l'API key del provider
        LLM che scegli — e puoi usare Ollama in locale per azzerarla.
    </div>
    <div class='stack-row'>
        <div class='stack-badge'>🤖 <span>XGBoost</span> — modello ML</div>
        <div class='stack-badge'>🔗 <span>LiteLLM</span> — 100+ provider LLM</div>
        <div class='stack-badge'>🎨 <span>Streamlit</span> — dashboard</div>
        <div class='stack-badge'>📄 <span>ReportLab</span> — PDF</div>
        <div class='stack-badge'>📊 <span>openpyxl</span> — Excel</div>
        <div class='stack-badge'>🐍 <span>scikit-learn</span> — pipeline ML</div>
        <div class='stack-badge'>🎭 <span>Faker</span> — dataset simulati</div>
        <div class='stack-badge'>📈 <span>Plotly</span> — visualizzazioni</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CTA FINALE
# ─────────────────────────────────────────────
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

if has_dataset and has_model and has_scored:
    cta_text = "Tutto pronto — vai all'Analisi"
    cta_sub  = "Dataset, modello e scoring già disponibili. Puoi esplorare la dashboard subito."
    cta_icon = "🔮"
elif has_dataset and has_model:
    cta_text = "Modello pronto — genera lo scoring"
    cta_sub  = "Dataset e modello disponibili. Vai su Analisi per generare gli score."
    cta_icon = "🎯"
elif has_dataset:
    cta_text = "Dataset pronto — avvia il training"
    cta_sub  = "Il dataset è caricato. Vai su Training per addestrare il modello."
    cta_icon = "🏋️"
else:
    cta_text = "Inizia da qui"
    cta_sub  = "Nessun dato ancora. Vai su Configurazione per generare o caricare un dataset."
    cta_icon = "⚙️"

st.markdown(f"""
<div style='background: linear-gradient(135deg, #13151c, #1a1d27);
            border-top: 1px solid #252836; border-bottom: 1px solid #252836;
            padding: 4rem 2rem; text-align: center;'>
    <div style='font-size: 2.8rem; margin-bottom: 1rem;'>{cta_icon}</div>
    <div style='font-family: Syne, sans-serif; font-size: 1.8rem; font-weight: 800;
                letter-spacing: -0.02em; margin-bottom: 0.6rem;'>{cta_text}</div>
    <div style='color: #6b7280; font-size: 0.95rem; margin-bottom: 2rem;'>{cta_sub}</div>
    <div style='color: #4b5563; font-size: 0.82rem;'>
        Usa il menu laterale ☰ in alto a sinistra per navigare tra le pagine
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class='footer'>
    HR Churn Predictor · Built with Streamlit · 
    Stack 100% open source · Privacy-first · GDPR-ready
</div>
""", unsafe_allow_html=True)
