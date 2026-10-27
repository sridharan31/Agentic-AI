import re
import time
import streamlit as st
from src.agents.agents import llm, build_search_agent, build_reader_agent, writer_chain, critic_chain

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
# Base colours come from .streamlit/config.toml (dark theme), so native text,
# markdown and widgets stay readable. This CSS only adds the visual polish.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, .stApp, .stMarkdown, p, li { font-family: 'DM Sans', sans-serif; }

.stApp {
    background: #07111f;
    background-image:
        radial-gradient(circle at top left, rgba(0,191,255,0.14), transparent 32%),
        radial-gradient(circle at bottom right, rgba(124,58,237,0.12), transparent 30%),
        linear-gradient(180deg, #07111f 0%, #0a1729 100%);
}

/* ── Hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1rem 3rem 4rem; max-width: 1200px; }

/* ── Hero header ── */
.hero { text-align: center; padding: 2.5rem 0 1.5rem; }
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #38bdf8;
    margin-bottom: 0.8rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.6rem, 6vw, 4.6rem);
    font-weight: 800;
    line-height: 1.0;
    letter-spacing: -0.03em;
    color: #f8fbff;
    margin: 0 0 1rem;
    padding: 0;
}
.hero h1 span {
    background: linear-gradient(135deg, #38bdf8, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    font-size: 1.05rem;
    color: #b5c3d9;
    max-width: 560px;
    margin: 0 auto;
    line-height: 1.65;
}
.model-badge {
    display: inline-block;
    margin-top: 1.1rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #b5c3d9;
    background: rgba(56,189,248,0.08);
    border: 1px solid rgba(56,189,248,0.25);
    border-radius: 999px;
    padding: 0.3rem 0.9rem;
}

.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(56,189,248,0.35), transparent);
    margin: 1.5rem 0 2rem;
}

/* ── Text input ── */
[data-testid="stTextInput"] [data-baseweb="input"] {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(56,189,248,0.3) !important;
    border-radius: 12px !important;
}
[data-testid="stTextInput"] [data-baseweb="input"]:focus-within {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 4px rgba(56,189,248,0.14) !important;
}
[data-testid="stTextInput"] input {
    background: transparent !important;
    color: #f8fbff !important;
    -webkit-text-fill-color: #f8fbff !important;
    caret-color: #38bdf8;
    font-size: 1rem !important;
    padding: 0.8rem 1rem !important;
}
[data-testid="stTextInput"] input::placeholder {
    color: #7f93ad !important;
    -webkit-text-fill-color: #7f93ad !important;
}
[data-testid="stWidgetLabel"] p {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: #38bdf8 !important;
}

/* ── Buttons ── */
[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #38bdf8 0%, #8b5cf6 100%) !important;
    color: #ffffff !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    box-shadow: 0 8px 30px rgba(56,189,248,0.22) !important;
    transition: all 0.18s ease !important;
}
[data-testid="stBaseButton-primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 35px rgba(56,189,248,0.32) !important;
}
[data-testid="stBaseButton-primary"] p { font-weight: 700 !important; font-size: 0.95rem !important; }
[data-testid="stBaseButton-secondary"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
    color: #d8e2f0 !important;
}
[data-testid="stBaseButton-secondary"]:hover {
    border-color: #38bdf8 !important;
    color: #38bdf8 !important;
}

/* ── Pipeline step cards ── */
.step-card {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.08);
    border-left: 4px solid rgba(255,255,255,0.12);
    border-radius: 14px;
    padding: 1rem 1.3rem;
    margin-bottom: 0.8rem;
}
.step-card.active { border-color: rgba(56,189,248,0.45); border-left-color: #38bdf8; background: rgba(56,189,248,0.07); }
.step-card.done   { border-color: rgba(34,197,94,0.28);  border-left-color: #22c55e; background: rgba(34,197,94,0.05); }
.step-card.failed { border-color: rgba(248,113,113,0.35); border-left-color: #f87171; background: rgba(248,113,113,0.06); }
.step-header { display: flex; align-items: center; gap: 0.8rem; }
.step-num { font-family: 'DM Mono', monospace; font-size: 0.7rem; letter-spacing: 0.15em; color: #38bdf8; }
.step-title { font-family: 'Syne', sans-serif; font-size: 1rem; font-weight: 700; color: #f8fbff; }
.step-status { margin-left: auto; font-family: 'DM Mono', monospace; font-size: 0.7rem; letter-spacing: 0.1em; }
.step-desc { font-size: 0.84rem; color: #94a3b8; margin-top: 0.25rem; }
.status-waiting { color: #64748b; }
.status-running { color: #38bdf8; }
.status-done    { color: #22c55e; }
.status-failed  { color: #f87171; }

.section-heading {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #f8fbff;
    margin: 0 0 1rem;
}
.try-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #7f93ad;
    letter-spacing: 0.12em;
    margin: 1.2rem 0 0.5rem;
}

/* ── Results ── */
[data-testid="stTabs"] button p { font-size: 0.95rem; }
[data-testid="stMetricValue"] { font-family: 'Syne', sans-serif; color: #38bdf8; }
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { font-family: 'Syne', sans-serif; color: #f8fbff; }
.stMarkdown a { color: #38bdf8; }

.notice {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #7f93ad;
    text-align: center;
    margin-top: 3rem;
    letter-spacing: 0.08em;
}
</style>
""", unsafe_allow_html=True)


STEPS = [
    ("search", "01", "Search Agent", "Gathers recent web information"),
    ("reader", "02", "Reader Agent", "Scrapes & extracts deep content"),
    ("writer", "03", "Writer Chain", "Drafts the full research report"),
    ("critic", "04", "Critic Chain", "Reviews & scores the report"),
]

EXAMPLES = [
    "Future of LLMs in the tech industry",
    "Latest AI agents in 2026",
    "Roadmap for AGI in the next 5 years",
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def render_pipeline(target, results: dict, running: str | None = None, failed: str | None = None):
    """Draw the four step cards into a placeholder so they can update live."""
    status_map = {
        "waiting": ("WAITING", "status-waiting", ""),
        "running": ("● RUNNING", "status-running", "active"),
        "done":    ("✓ DONE", "status-done", "done"),
        "failed":  ("✕ FAILED", "status-failed", "failed"),
    }
    html = ""
    for key, num, title, desc in STEPS:
        if key == failed:
            state = "failed"
        elif key in results:
            state = "done"
        elif key == running:
            state = "running"
        else:
            state = "waiting"
        label, status_cls, card_cls = status_map[state]
        # No indentation: indented lines would be rendered as a markdown code block
        html += (
            f'<div class="step-card {card_cls}">'
            f'<div class="step-header">'
            f'<span class="step-num">{num}</span>'
            f'<span class="step-title">{title}</span>'
            f'<span class="step-status {status_cls}">{label}</span>'
            f'</div>'
            f'<div class="step-desc">{desc}</div>'
            f'</div>'
        )
    target.markdown(html, unsafe_allow_html=True)


def extract_score(feedback: str) -> str | None:
    match = re.search(r"Score:\s*\**\s*(\d+(?:\.\d+)?)\s*/\s*10", feedback, re.IGNORECASE)
    return match.group(1) if match else None


def use_example(text: str):
    st.session_state.topic_input = text


# ── Session state init ────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = {}

model_name = getattr(llm, "model", None) or getattr(llm, "model_name", "unknown")


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
<div class="hero-eyebrow">Multi-Agent AI System</div>
<h1>Researcher<span>Agent</span></h1>
<p class="hero-sub">Four specialized AI agents collaborate — searching, scraping, writing,
and critiquing — to deliver a polished research report on any topic.</p>
<div class="model-badge">⚙ Model: {model_name}</div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)


# ── Layout: input left, pipeline right ───────────────────────────────────────
col_input, col_spacer, col_pipeline = st.columns([5, 0.4, 4])

with col_input:
    st.markdown('<div class="section-heading">Topic</div>', unsafe_allow_html=True)

    with st.container(border=True):
        topic = st.text_input(
            "Research Topic",
            placeholder="e.g. Roadmap for AGI development in next 5 years",
            key="topic_input",
        )
        run_btn = st.button("⚡ Run Research Pipeline", type="primary", width="stretch")

    st.markdown('<div class="try-label">TRY AN EXAMPLE →</div>', unsafe_allow_html=True)
    for ex in EXAMPLES:
        st.button(ex, key=f"ex_{ex}", on_click=use_example, args=(ex,), width="stretch")

with col_pipeline:
    st.markdown('<div class="section-heading">Pipeline</div>', unsafe_allow_html=True)
    pipeline_ph = st.empty()
    render_pipeline(pipeline_ph, st.session_state.results)


# ── Run pipeline ──────────────────────────────────────────────────────────────
if run_btn:
    topic_val = topic.strip()

    if not topic_val:
        st.warning("Please enter a research topic first.")
    else:
        results = {}
        st.session_state.results = {}
        current = "search"
        started = time.time()

        with st.status("Running research pipeline…", expanded=True) as status:
            try:
                # ── Step 1: Search ──
                render_pipeline(pipeline_ph, results, running="search")
                st.write("🔍 Search Agent is searching the web…")
                sr = build_search_agent().invoke({
                    "messages": [("user", f"Find recent, reliable and detailed information about: {topic_val}")]
                })
                results["search"] = sr["messages"][-1].content

                # ── Step 2: Reader ──
                current = "reader"
                render_pipeline(pipeline_ph, results, running="reader")
                st.write("📄 Reader Agent is scraping the top resource…")
                rr = build_reader_agent().invoke({
                    "messages": [(
                        "user",
                        f"Based on the following search results about '{topic_val}', "
                        f"pick the most relevant URL and scrape it for deeper content.\n\n"
                        f"Search Results:\n{results['search'][:800]}"
                    )]
                })
                results["reader"] = rr["messages"][-1].content

                # ── Step 3: Writer (streamed live) ──
                current = "writer"
                render_pipeline(pipeline_ph, results, running="writer")
                st.write("✍️ Writer is drafting the report…")
                research_combined = (
                    f"SEARCH RESULTS:\n{results['search']}\n\n"
                    f"DETAILED SCRAPED CONTENT:\n{results['reader']}"
                )
                with st.container(border=True, height=320):
                    results["writer"] = st.write_stream(
                        writer_chain.stream({"topic": topic_val, "research": research_combined})
                    )

                # ── Step 4: Critic ──
                current = "critic"
                render_pipeline(pipeline_ph, results, running="critic")
                st.write("🧐 Critic is reviewing the report…")
                results["critic"] = critic_chain.invoke({"report": results["writer"]})

                render_pipeline(pipeline_ph, results)
                status.update(
                    label=f"Done in {time.time() - started:.0f}s",
                    state="complete",
                    expanded=False,
                )
            except Exception as e:
                render_pipeline(pipeline_ph, results, failed=current)
                status.update(label=f"Failed at the {current} step", state="error", expanded=True)
                st.error(f"{type(e).__name__}: {e}")

        results["topic"] = topic_val
        st.session_state.results = results


# ── Results display ───────────────────────────────────────────────────────────
r = st.session_state.results

if r.get("writer"):
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    head_left, head_right = st.columns([3, 1])
    with head_left:
        st.markdown('<div class="section-heading">Results</div>', unsafe_allow_html=True)
        st.caption(f"Topic: {r.get('topic', '')}")
    with head_right:
        score = extract_score(r.get("critic", ""))
        if score:
            st.metric("Critic score", f"{score} / 10")

    tab_report, tab_critic, tab_search, tab_reader = st.tabs(
        ["📝 Report", "🧐 Critic Feedback", "🔍 Search Results", "📄 Scraped Content"]
    )

    with tab_report:
        with st.container(border=True):
            st.markdown(r["writer"])
        st.download_button(
            label="⬇ Download Report (.md)",
            data=r["writer"],
            file_name=f"research_report_{int(time.time())}.md",
            mime="text/markdown",
        )

    with tab_critic:
        with st.container(border=True):
            st.markdown(r.get("critic", "_The critic step did not finish._"))

    with tab_search:
        with st.container(border=True):
            st.markdown(r.get("search", ""))

    with tab_reader:
        with st.container(border=True):
            st.markdown(r.get("reader", ""))


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="notice">ResearchAgent · Powered by LangChain multi-agent pipeline · Built with Streamlit</div>
""", unsafe_allow_html=True)
