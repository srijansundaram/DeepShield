"""
DeepShield — Session History & Audit Log
Tracks all analyses within a Streamlit session
"""

import streamlit as st
import datetime
from typing import Dict, List, Optional
import json


def init_history():
    """Initialize session history if not present."""
    if "analysis_history" not in st.session_state:
        st.session_state.analysis_history = []


def add_to_history(
    filename: str,
    analysis_type: str,
    result: Dict,
    thumbnail=None,
):
    """Add an analysis result to session history."""
    init_history()
    entry = {
        "id": len(st.session_state.analysis_history) + 1,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": filename,
        "type": analysis_type,
        "verdict": result.get("verdict", "Unknown"),
        "confidence": round(result.get("confidence", 0) * 100, 1),
        "fake_probability": round(result.get("fake_probability", 0) * 100, 1),
        "thumbnail": thumbnail,
    }
    st.session_state.analysis_history.insert(0, entry)


def get_history() -> List[Dict]:
    """Return session history list."""
    init_history()
    return st.session_state.analysis_history


def clear_history():
    """Clear all session history."""
    st.session_state.analysis_history = []


def render_history_panel():
    """Render the history log as a Streamlit component."""
    init_history()
    history = st.session_state.analysis_history

    if not history:
        st.info("No analyses yet in this session.")
        return

    st.markdown(f"**{len(history)} analysis record(s) this session**")

    if st.button("Clear history", key="clear_history_btn"):
        clear_history()
        st.rerun()

    for entry in history:
        is_fake = entry["verdict"] == "Deepfake"
        badge_color = "#E24B4A" if is_fake else "#1D9E75"
        badge_bg = "#FCEBEB" if is_fake else "#E1F5EE"

        with st.container():
            cols = st.columns([0.5, 3, 1.5, 1.5])
            cols[0].markdown(f"**#{entry['id']}**")
            cols[1].markdown(f"**{entry['filename'][:30]}**  \n"
                             f"<small style='color:#888'>🕐 {entry['timestamp']} · {entry['type'].capitalize()}</small>",
                             unsafe_allow_html=True)
            cols[2].markdown(
                f"<span style='background:{badge_bg};color:{badge_color};"
                f"padding:3px 10px;border-radius:99px;font-size:12px;font-weight:500'>"
                f"{entry['verdict']}</span>",
                unsafe_allow_html=True,
            )
            cols[3].markdown(f"**{entry['confidence']}%** conf.")
            st.divider()
