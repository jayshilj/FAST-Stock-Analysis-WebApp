"""ui_theme.py â€“ FAST Financial Dashboard Design System.

This module provides the centralised CSS injection, navigation layout, and
HTML component renderers for the FAST Stock Analysis WebApp.

Usage::

    from ui_theme import (
        init_theme_state, inject_global_css, render_top_bar,
        render_sidebar_navigation, render_company_hero, render_metric_strip,
        render_section_card_start, render_section_card_end,
        render_insight_card, render_sentiment_badge, page_title,
    )

Design tokens:
    - Primary colour: #6366F1 (indigo)
    - Positive sentiment: #22C55E (green)
    - Negative sentiment: #EF4444 (red)
    - Neutral sentiment: #F59E0B (amber)
    - Typography: Google Fonts â€“ Inter (400/500/600/700/800)

All render_* functions accept a ``st`` parameter (the ``streamlit`` module
reference) to avoid a top-level import of Streamlit, which keeps this
module importable in testing environments without a running Streamlit server.
"""
from __future__ import annotations

from datetime import datetime
from html import escape
from typing import Optional

APP_BRAND_FULL = "Financial Analysis and Stock Trading Analysis"
APP_BRAND_TAGLINE = "Real-time market insights"

NAV_DEFINITION = [
    ("Dashboard", "âœ¨", "Dashboard"),
    ("About the Project", "ðŸ ", "Overview"),
    ("Agentic Research Bot", "ðŸ¤–", "Agentic Bots"),
    ("Live News Sentiment", "ðŸ“°", "News & sentiment"),
    ("Company Basic Details", "ðŸ“‹", "Company profile"),
    ("Company Advanced Details", "ðŸ“Š", "Technicals"),
    ("Google Trends with Forecast", "ðŸ”Ž", "Search trends"),
    ("Social Media Trends", "ðŸ’¬", "Social trends"),
    ("Meeting Summarization", "ðŸŽ™ï¸", "Meeting notes"),
    ("Stock Future Prediction", "ðŸ”®", "Forecast"),
]


def _nav_display(internal, icon, short):
    return f"{icon}  {short}"


def init_theme_state(st):
    """Initialise Streamlit session-state keys required by the theme.

    Must be called once before any other render function. Currently seeds
    the global ticker search session key used to persist state across page
    navigations.
    """
    if "global_ticker_search" not in st.session_state:
        st.session_state["global_ticker_search"] = "AAPL"


def inject_global_css(st):
    """Inject the global glassmorphic CSS design system into the Streamlit page.

    Imports the Inter typeface from Google Fonts and applies CSS custom
    properties for layout tokens, glassmorphic section cards, metric strips,
    badge styles, top-bar, hero headers, news items, and the developer card.
    Should be called once, immediately after ``st.set_page_config()``.
    """
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

        .sidebar-dev-card {
            margin-top: 1.5rem;
            padding: 1rem;
            border-top: 1px solid rgba(128,128,128,0.2);
            background: rgba(128,128,128,0.03);
            border-radius: 16px;
        }
        .dev-name {
            font-size: 0.88rem;
            font-weight: 700;
            color: var(--text-color);
            margin-bottom: 0.6rem;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }
        .dev-links {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }
        .dev-link {
            font-size: 0.78rem;
            color: var(--primary-color) !important;
            text-decoration: none !important;
            display: flex;
            align-items: center;
            gap: 0.4rem;
            transition: opacity 0.2s;
            opacity: 0.8;
        }
        .dev-link:hover {
            opacity: 1;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_top_bar(st):
    """Render the branded top navigation bar with live timestamp.

    Displays the APP_BRAND_FULL name (with the first word highlighted in the
    primary colour) and the current date/time in ``YYYY-MM-DD Â· HH:MM`` format.
    """
    now = datetime.now().strftime("%Y-%m-%d Â· %H:%M")
    first, _, rest = APP_BRAND_FULL.partition(" ")
    brand_html = '<p class="topbar-brand"><span>{}</span> {}</p>'.format(
        escape(first), escape(rest)
    )

    st.markdown(
        """
        <div class="topbar-wrap">
          <div class="topbar-inner">
            {brand}
            <p class="topbar-meta">Live dashboard Â· {now}</p>
          </div>
        </div>
        """.format(brand=brand_html, now=now),
        unsafe_allow_html=True,
    )


def render_sidebar_navigation(st):
    """Render the branded sidebar including logo, nav radio buttons, and dev card.

    Returns:
        str: The internal page name selected by the user (e.g. ``'Dashboard'``,
             ``'Agentic Research Bot'``, etc.) as defined in NAV_DEFINITION.
    """
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
    st.sidebar.caption("News Â· Forecast Â· Technicals Â· Fundamentals")

    st.sidebar.markdown(
        """
        <div class="sidebar-dev-card">
            <div class="dev-name">ðŸ‘¨â€ðŸ’» Developed by Jayshil Jain</div>
            <div class="dev-links">
                <a class="dev-link" href="https://github.com/jayshilj/GeoPulseWebApp" target="_blank">ðŸ“‚ GitHub Repository</a>
                <a class="dev-link" href="https://www.linkedin.com/in/jayshiljain/" target="_blank">ðŸ”— LinkedIn Profile</a>
                <a class="dev-link" href="https://www.jayshil.com/" target="_blank">ðŸŒ Personal Website</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    return internal_by_display[choice]


def render_dashboard_hero(st):
    """Render the hero banner displayed at the top of the Dashboard page."""
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
    """Open a glassmorphic section card with a title and optional subtitle.

    Must always be paired with a call to :func:`render_section_card_end` to
    close the ``<div>`` element.
    """
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
    """Close a section card opened by :func:`render_section_card_start`."""
    st.markdown("</div>", unsafe_allow_html=True)


def render_insight_card(st, label, value, help_text):
    """Render a compact KPI card with a label, prominent value, and help text.

    Args:
        st: The ``streamlit`` module reference.
        label (str): Short uppercase label (e.g. ``'Live price'``).
        value (str): Formatted metric value (e.g. ``'$182.63'``).
        help_text (str): One-line explanatory text shown below the value.
    """
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
        price_txt = str(price) if price else "â€”"

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
        return "â€”"
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
    trailing = info.get("trailingAnnualDividendYield")
    
    divy_disp = "â€”"
    if isinstance(divy, (int, float)):
        # Heuristic to detect percentage vs decimal
        if isinstance(trailing, (int, float)) and trailing > 0:
            ratio = divy / trailing
            if 80 < ratio < 120:
                # divy is likely a percentage number, trailing is decimal
                divy_disp = "{:.2f}%".format(divy)
            elif 0.8 < ratio < 1.2:
                # both are in same format
                divy_disp = "{:.2f}%".format(divy * 100 if divy < 0.2 else divy)
            else:
                divy_disp = "{:.2f}%".format(divy if divy > 0.1 else divy * 100)
        else:
            # Fallback if no trailing info
            divy_disp = "{:.2f}%".format(divy if divy > 0.1 else divy * 100)
    elif isinstance(trailing, (int, float)):
        divy_disp = "{:.2f}%".format(trailing * 100)

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

def render_alert_banner(st, message, alert_type="info"):
    """Render a styled alert banner with icon and contextual colour.

    Args:
        st:         The ``streamlit`` module reference.
        message:    The alert message text (will be HTML-escaped).
        alert_type: One of ``'info'`` (blue), ``'success'`` (green),
                    ``'warning'`` (amber), or ``'danger'`` (red).
                    Defaults to ``'info'``.

    Example::

        render_alert_banner(st, "RSI crossed above 70 � overbought signal.", "warning")
    """
    _ALERT_STYLES = {
        "info":    {"bg": "rgba(99,102,241,0.15)",  "border": "#6366F1", "icon": "??"},
        "success": {"bg": "rgba(34,197,94,0.15)",   "border": "#22C55E", "icon": "?"},
        "warning": {"bg": "rgba(245,158,11,0.15)",  "border": "#F59E0B", "icon": "??"},
        "danger":  {"bg": "rgba(239,68,68,0.15)",   "border": "#EF4444", "icon": "??"},
    }
    style = _ALERT_STYLES.get(alert_type, _ALERT_STYLES["info"])
    st.markdown(
        """
        <div style="
            background:{bg};
            border-left:4px solid {border};
            border-radius:8px;
            padding:0.75rem 1rem;
            margin:0.5rem 0;
            display:flex;
            align-items:center;
            gap:0.6rem;
            font-size:0.95rem;
        ">
            <span style="font-size:1.2rem;">{icon}</span>
            <span>{message}</span>
        </div>
        """.format(
            bg=style["bg"],
            border=style["border"],
            icon=style["icon"],
            message=escape(str(message)),
        ),
        unsafe_allow_html=True,
    )


def render_metric_delta_card(st, label, value, delta, delta_label="vs prev. close"):
    """Render a metric card with a value and a coloured delta indicator.

    Args:
        st:          The ``streamlit`` module reference.
        label:       Short metric name (e.g. ``'Daily Return'``).
        value:       Primary formatted value string (e.g. ``'+1.23%'``).
        delta:       Numeric delta used to determine colour direction.
        delta_label: Descriptive label for the delta (default ``'vs prev. close'``).
    """
    colour = "#22C55E" if float(delta) >= 0 else "#EF4444"
    arrow = "?" if float(delta) >= 0 else "?"
    st.markdown(
        """
        <div style="
            background:rgba(255,255,255,0.04);
            border:1px solid rgba(255,255,255,0.08);
            border-radius:12px;
            padding:1rem 1.2rem;
            text-align:center;
        ">
            <p style="margin:0;font-size:0.78rem;opacity:0.6;text-transform:uppercase;letter-spacing:0.06em;">{label}</p>
            <p style="margin:0.4rem 0 0;font-size:1.6rem;font-weight:700;">{value}</p>
            <p style="margin:0.2rem 0 0;font-size:0.82rem;color:{colour};">{arrow} {delta_label}</p>
        </div>
        """.format(
            label=escape(str(label)),
            value=escape(str(value)),
            colour=colour,
            arrow=arrow,
            delta_label=escape(str(delta_label)),
        ),
        unsafe_allow_html=True,
    )
