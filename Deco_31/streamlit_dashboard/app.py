
import streamlit as st

st.set_page_config(page_title="QMP GOD MODE Dashboard", layout="wide")

st.title("QMP GOD MODE v2.5 — Market Intelligence Interface")

st.markdown("### Live Modules")
st.success("SPIRIT — Active")
st.success("MM_EXPLOIT — Active")
st.success("DNA_HEART — Online")

st.markdown("### Current Signals")
st.metric(label="Pain Index", value="0.84", delta="+0.02")
st.metric(label="Greed Pulse", value="0.91", delta="-0.01")
