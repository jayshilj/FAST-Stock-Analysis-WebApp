from __future__ import annotations

from datetime import datetime
from html import escape
from typing import Optional

APP_BRAND_FULL = "Financial Analysis and Stock Trading Analysis"
APP_BRAND_TAGLINE = "Real-time market insights"

NAV_DEFINITION = [
    ("Dashboard", "✨", "Dashboard"),
    ("About the Project", "🏠", "Overview"),
    ("Agentic Research Bot", "🤖", "Agentic Bots"),
    ("Live News Sentiment", "📰", "News & sentiment"),
    ("Company Basic Details", "📋", "Company profile"),
    ("Company Advanced Details", "📊", "Technicals"),
    ("Google Trends with Forecast", "🔎", "Search trends"),
    ("Social Media Trends", "💬", "Social trends"),
    ("Meeting Summarization", "🎙️", "Meeting notes"),
    ("Stock Future Prediction", "🔮", "Forecast"),
]


def _nav_display(internal, icon, short):
    return f"{icon}  {short}"


def init_theme_state(st):
    if "global_ticker_search" not in st.session_state:
        st.session_state["global_ticker_search"] = "AAPL"


def inject_global_css(st):
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, .stApp, [data-testid="stAppViewContainer"] {
            font-family: 'Inter', system-ui, sans-serif !important;
        }

        .stApp {
            background-color: transparent !important;
        }

        header[data-testid="stHeader"],
        div[data-testid="stToolbar"],
        div[data-testid="stDecoration"],
        section.main > div,
        [data-testid="stAppViewContainer"] > .main {
            background: transparent !important;
            background-image: none !important;
        }

        .block-container {
            max-width: 1320px;
            padding-top: 1.2rem !important;
            padding-bottom: 3rem !important;
        }

        h1, h2, h3 {
            letter-spacing: -0.03em;
            font-weight: 700 !important;
            color: var(--text-color) !important;
        }

        p, label, .stMarkdown, .stCaption {
            color: var(--text-color);
            opacity: 0.85;
        }

        section[data-testid="stSidebar"] > div {
            background-color: var(--secondary-background-color) !important;
            backdrop-filter: blur(16px);
        }

        .sidebar-brand {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 1rem;
            padding: 0.75rem;
            border: 1px solid rgba(128,128,128,0.2);
            border-radius: 18px;
            background: rgba(128,128,128,0.05);
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        }

        .sb-logo {
            width: 46px;
            height: 46px;
            border-radius: 14px;
            background: linear-gradient(135deg, var(--primary-color), #22C55E);
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 1.1rem;
            box-shadow: 0 8px 24px rgba(0,0,0,0.10);
        }

        .sb-title {
            font-weight: 800;
            font-size: 0.95rem;
            color: var(--text-color);
            line-height: 1.2;
        }

        .sb-sub {
            font-size: 0.75rem;
            opacity: 0.7;
            margin-top: 0.2rem;
        }

        .stRadio > div {
            gap: 0.45rem;
        }

        .stRadio label {
            border-radius: 14px !important;
            padding: 0.52rem 0.7rem !important;
            border: 1px solid transparent !important;
            background: transparent !important;
            transition: all 0.18s ease;
        }

        .stRadio label:hover {
            background: rgba(128,128,128,0.1) !important;
        }

        .stTextInput input,
        .stNumberInput input,
        .stDateInput input,
        .stSelectbox div[data-baseweb="select"],
        [data-baseweb="input"] {
            background: rgba(128,128,128,0.05) !important;
            color: var(--text-color) !important;
            border: 1px solid rgba(128,128,128,0.2) !important;
            border-radius: 14px !important;
        }

        .stButton > button {
            border-radius: 14px !important;
            border: 1px solid rgba(128,128,128,0.2) !important;
            background: rgba(128,128,128,0.05) !important;
            color: var(--text-color) !important;
            font-weight: 700 !important;
            padding: 0.55rem 1rem !important;
            transition: all 0.18s ease;
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            border-color: var(--primary-color) !important;
        }

        div[data-testid="stMetric"] {
            background: rgba(128,128,128,0.03);
            border: 1px solid rgba(128,128,128,0.2);
            border-radius: 18px;
            padding: 1rem 1rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            backdrop-filter: blur(10px);
        }

        div[data-testid="stMetricLabel"] {
            opacity: 0.8 !important;
        }

        div[data-testid="stMetricValue"] {
            color: var(--text-color) !important;
            font-weight: 800 !important;
        }

        .topbar-wrap {
            border: 1px solid rgba(128,128,128,0.2);
            border-radius: 22px;
            padding: 1rem 1.2rem;
            margin-bottom: 1.2rem;
            background: var(--background-color);
            backdrop-filter: blur(16px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        }

        .topbar-inner {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
            flex-wrap: wrap;
        }

        .topbar-brand {
            font-size: clamp(1.1rem, 2vw, 1.5rem);
            font-weight: 800;
            margin: 0;
            letter-spacing: -0.03em;
        }

        .topbar-brand span {
            color: var(--primary-color);
        }

        .topbar-meta {
            font-size: 0.82rem;
            opacity: 0.7;
            margin: 0;
        }

        .hero-card {
            border: 1px solid rgba(128,128,128,0.2);
            border-radius: 24px;
            padding: 1.4rem 1.45rem;
            margin-bottom: 1.1rem;
            background: var(--background-color);
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            backdrop-filter: blur(10px);
        }

        .hero-kicker {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.15em;
            color: var(--primary-color);
            font-weight: 700;
            margin-bottom: 0.55rem;
        }

        .hero-title {
            font-size: clamp(1.8rem, 4vw, 3rem);
            font-weight: 800;
            color: var(--text-color);
            line-height: 1.05;
            margin-bottom: 0.45rem;
        }

        .hero-subtitle {
            font-size: 1rem;
            opacity: 0.8;
            line-height: 1.6;
            max-width: 780px;
        }

        .page-card {
            background: var(--background-color);
            border: 1px solid rgba(128,128,128,0.2);
            border-radius: 20px;
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        }

        .section-card {
            background: var(--background-color);
            border: 1px solid rgba(128,128,128,0.2);
            border-radius: 20px;
            padding: 1rem 1rem 0.6rem 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        }

        .section-title {
            font-size: 1.15rem;
            font-weight: 700;
            color: var(--text-color);
            margin-bottom: 0.3rem;
        }

        .section-subtitle {
            font-size: 0.86rem;
            opacity: 0.75;
            margin-bottom: 0.9rem;
        }

        .insight-card, .rail-card {
            border: 1px solid rgba(128,128,128,0.2);
            border-radius: 18px;
            padding: 1rem;
            margin-bottom: 0.8rem;
            background: var(--background-color);
            box-shadow: 0 4px 12px rgba(0,0,0,0.02);
        }

        .insight-label, .rail-title {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            opacity: 0.7;
            margin-bottom: 0.35rem;
            font-weight: 700;
        }

        .insight-value {
            font-size: 1.35rem;
            font-weight: 800;
            color: var(--text-color);
            margin-bottom: 0.2rem;
        }

        .insight-help {
            font-size: 0.88rem;
            opacity: 0.8;
            line-height: 1.5;
        }

        .badge {
            display: inline-block;
            padding: 0.28rem 0.65rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 700;
            margin-right: 0.35rem;
            border: 1px solid rgba(128,128,128,0.2);
        }

        .badge-pos {
            background: rgba(34,197,94,0.15);
            color: #22C55E;
        }

        .badge-neg {
            background: rgba(239,68,68,0.14);
            color: #EF4444;
        }

        .badge-neutral {
            background: rgba(245,158,11,0.13);
            color: #F59E0B;
        }

        .hero-header {
            background: rgba(128,128,128,0.05);
            border: 1px solid rgba(128,128,128,0.2);
            border-radius: 22px;
            padding: 1.3rem 1.35rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        }

        .hero-ticker {
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: -0.04em;
            color: var(--text-color);
        }

        .hero-name {
            font-size: 0.95rem;
            opacity: 0.7;
            margin-top: 0.2rem;
        }

        .hero-price {
            font-size: 1.9rem;
            font-weight: 800;
            margin-top: 0.55rem;
            color: var(--text-color);
        }

        .hero-chg-pos {
            color: #22C55E;
            font-weight: 700;
        }

        .hero-chg-neg {
            color: #EF4444;
            font-weight: 700;
        }

        .rail-card {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(128,128,128,0.2);
            border-radius: 18px;
            padding: 1rem;
            margin-bottom: 0.8rem;
            color: var(--text-color);
            box-shadow: 0 4px 12px rgba(0,0,0,0.02);
        }

        .rail-title {
            font-weight: 700;
            color: var(--text-color);
            margin-bottom: 0.45rem;
            font-size: 0.95rem;
        }

        .news-item {
            padding: 1rem 0;
            border-bottom: 1px solid rgba(128,128,128,0.15);
            display: flex;
            flex-direction: column;
            gap: 0.6rem;
            text-align: left;
        }

        .news-item:last-child {
            border-bottom: none;
        }

        .news-header-meta {
            display: flex;
            justify-content: space-between;
            align-items: center;
            width: 100%;
        }

        .news-headline {
            font-size: 0.95rem;
            color: var(--text-color);
            font-weight: 600;
            line-height: 1.4;
            margin-top: 0.2rem;
        }

        .news-meta {
            font-size: 0.78rem;
            opacity: 0.65;
        }

        [data-testid="stDataFrame"] {
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid rgba(128,128,128,0.2);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_top_bar(st):
    now = datetime.now().strftime("%Y-%m-%d · %H:%M")
    first, _, rest = APP_BRAND_FULL.partition(" ")
    brand_html = '<p class="topbar-brand"><span>{}</span> {}</p>'.format(
        escape(first), escape(rest)
    )

    st.markdown(
        """
        <div class="topbar-wrap">
          <div class="topbar-inner">
            {brand}
            <p class="topbar-meta">Live dashboard · {now}</p>
          </div>
        </div>
        """.format(brand=brand_html, now=now),
        unsafe_allow_html=True,
    )


def render_sidebar_navigation(st):
    logo_letter = escape(APP_BRAND_FULL.strip()[0].upper())
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
          <div class="sb-logo">{logo}</div>
          <div>
            <div class="sb-title">{title}</div>
            <div class="sb-sub">{subtitle}</div>
          </div>
        </div>
        """.format(
            logo=logo_letter,
            title=escape(APP_BRAND_FULL),
            subtitle=escape(APP_BRAND_TAGLINE),
        ),
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("##### Navigation")
    options_display = [_nav_display(*row) for row in NAV_DEFINITION]
    internal_by_display = {_nav_display(*row): row[0] for row in NAV_DEFINITION}

    choice = st.sidebar.radio(
        "nav",
        options_display,
        label_visibility="collapsed",
        key="sidebar_nav_radio",
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Smart market workspace")
    st.sidebar.caption("News · Forecast · Technicals · Fundamentals")

    return internal_by_display[choice]


def render_dashboard_hero(st):
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-kicker">AI Finance Dashboard</div>
            <div class="hero-title">Track markets, sentiment, and forecast trends in one place.</div>
            <div class="hero-subtitle">
                A modern stock intelligence dashboard combining price movement, technical indicators,
                live news sentiment, and predictive analytics in a clean institutional-style interface.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_card_start(st, title, subtitle=""):
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">{title}</div>
            <div class="section-subtitle">{subtitle}</div>
        """.format(
            title=escape(str(title)),
            subtitle=escape(str(subtitle)),
        ),
        unsafe_allow_html=True,
    )


def render_section_card_end(st):
    st.markdown("</div>", unsafe_allow_html=True)


def render_insight_card(st, label, value, help_text):
    st.markdown(
        """
        <div class="insight-card">
            <div class="insight-label">{label}</div>
            <div class="insight-value">{value}</div>
            <div class="insight-help">{help_text}</div>
        </div>
        """.format(
            label=escape(str(label)),
            value=escape(str(value)),
            help_text=escape(str(help_text)),
        ),
        unsafe_allow_html=True,
    )


def render_sentiment_badge(sentiment):
    s = str(sentiment or "").strip().lower()
    if s == "positive":
        return '<span class="badge badge-pos">Positive</span>'
    if s == "negative":
        return '<span class="badge badge-neg">Negative</span>'
    return '<span class="badge badge-neutral">Neutral</span>'


def render_company_hero(st, ticker, info):
    name = info.get("longName") or info.get("shortName") or ticker
    price = info.get("regularMarketPrice") or info.get("currentPrice")
    prev = info.get("previousClose")
    cur = info.get("currency") or "USD"

    chg_pct = None
    if price is not None and prev not in (None, 0):
        try:
            chg_pct = (float(price) - float(prev)) / float(prev) * 100.0
        except Exception:
            chg_pct = None

    chg_cls = "hero-chg-pos" if (chg_pct is not None and chg_pct >= 0) else "hero-chg-neg"
    chg_txt = ""
    if chg_pct is not None:
        sign = "+" if chg_pct >= 0 else ""
        chg_txt = '<span class="{cls}">{sign}{pct:.2f}%</span> vs prior close'.format(
            cls=chg_cls, sign=sign, pct=chg_pct
        )

    if isinstance(price, (int, float)):
        price_txt = "{:,.2f}".format(price)
    else:
        price_txt = str(price) if price else "—"

    st.markdown(
        """
        <div class="hero-header">
          <div class="hero-ticker">{ticker}</div>
          <div class="hero-name">{name}</div>
          <div class="hero-price">{price} <span style="font-size:0.95rem;font-weight:500;">{currency}</span></div>
          <div style="margin-top:0.35rem;font-size:0.92rem;">{chg}</div>
        </div>
        """.format(
            ticker=escape(str(ticker)),
            name=escape(str(name)),
            price=escape(str(price_txt)),
            currency=escape(str(cur)),
            chg=chg_txt,
        ),
        unsafe_allow_html=True,
    )


def _fmt_metric(v):
    if v is None or v == "N/A":
        return "—"
    try:
        if isinstance(v, (int, float)):
            if abs(v) >= 1e12:
                return "{:.2f}T".format(v / 1e12)
            if abs(v) >= 1e9:
                return "{:.2f}B".format(v / 1e9)
            if abs(v) >= 1e6:
                return "{:.2f}M".format(v / 1e6)
            if abs(v) >= 1000:
                return "{:,.0f}".format(v)
            return "{:,.4g}".format(v).rstrip("0").rstrip(".")
    except Exception:
        pass
    return str(v)


def render_metric_strip(st, info):
    mcap = info.get("marketCap")
    pe = info.get("forwardPE") or info.get("trailingPE")
    divy = info.get("dividendYield")

    if isinstance(divy, float) and 0 < divy < 1:
        divy_disp = "{:.2f}%".format(divy * 100)
    else:
        divy_disp = _fmt_metric(divy) if divy not in (None, "N/A") else "—"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Market cap", _fmt_metric(mcap))
    with c2:
        st.metric("P/E", _fmt_metric(pe))
    with c3:
        st.metric("Dividend yield", divy_disp)
    with c4:
        st.metric("Beta", _fmt_metric(info.get("beta")))


def render_right_rail_placeholder(st):
    st.markdown('<div class="rail-title">Market pulse</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="rail-card">Add top gainers, watchlist alerts, or sector heatmaps here.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="rail-card">This side rail works best for summaries, signals, and quick stats.</div>',
        unsafe_allow_html=True,
    )


def page_title(st, title, subtitle=None):
    st.markdown(
        '<div class="page-card"><h2 style="margin:0;">{}</h2>'.format(escape(str(title))),
        unsafe_allow_html=True,
    )
    if subtitle:
        st.markdown(
            '<p style="margin:0.35rem 0 0 0;opacity:0.82;font-size:0.95rem;">{}</p>'.format(
                escape(str(subtitle))
            ),
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)