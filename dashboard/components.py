"""Shared Streamlit components and page styling."""

import streamlit as st


def configure_page() -> None:
    st.set_page_config(
        page_title="Agent Observatory",
        page_icon="◉",
        layout="wide",
        initial_sidebar_state="auto",
    )
    st.markdown(
        """
        <style>
        :root { --ink: #172320; --muted: #61706b; --line: #dce5df; --accent: #176b58; }
        .stApp { background: #f5f8f5; color: var(--ink); }
        [data-testid="stSidebar"] { background: #edf3ee; border-right: 1px solid var(--line); }
        [data-testid="stMetric"] { background: #ffffff; border: 1px solid var(--line); padding: 14px 16px; border-radius: 6px; }
        [data-testid="stMetricLabel"] { color: var(--muted); }
        h1, h2, h3 { color: var(--ink); }
        .eyebrow { color: var(--accent); font-size: .75rem; font-weight: 700; text-transform: uppercase; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str) -> None:
    st.markdown('<div class="eyebrow">AGENT OBSERVATORY</div>', unsafe_allow_html=True)
    st.title(title)
    st.caption(subtitle)


def empty_state(message: str) -> None:
    st.info(message, icon="ℹ️")