"""
Minimal ops dashboard: calls the /predict endpoint and lists flagged accounts.
Run: streamlit run dashboard/app.py
"""
import requests
import streamlit as st

API_URL = "http://localhost:8000/predict"

st.title("Electricity Theft Detection — Ops Dashboard")
st.caption("Every row here came from a model score, not a field visit.")

consumer_id = st.text_input("Consumer ID")
client_type = st.selectbox("Client type", ["residential", "industrial"])
daily_kwh_raw = st.text_area("Daily kWh values (comma-separated, must match training sequence length)")

if st.button("Score this account"):
    try:
        daily_kwh = [float(v.strip()) for v in daily_kwh_raw.split(",") if v.strip()]
        resp = requests.post(API_URL, json={
            "consumer_id": consumer_id,
            "client_type": client_type,
            "daily_kwh": daily_kwh,
        })
        resp.raise_for_status()
        data = resp.json()
        st.metric("Theft probability", f"{data['theft_probability']:.3f}")
        st.write("Flagged for inspection:" if data["is_theft_suspect"] else "Not flagged.")
        if data["reasons"]:
            st.write("Reasons:")
            for r in data["reasons"]:
                st.write(f"- {r}")
    except Exception as e:
        st.error(f"Request failed: {e}")
