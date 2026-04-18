"""Streamlit UI for viva-friendly prediction demo."""

from __future__ import annotations

import streamlit as st

from project.prediction.predictor import predict_failure


st.set_page_config(page_title="Log Failure Prediction Demo", layout="centered")

st.title("Log Failure Prediction using LSTM")
st.caption("Enter exactly 5 EventIds, then click Predict.")

sequence_input = st.text_input("Event Sequence", placeholder="E1 E2 E3 E4 E5")

if st.button("Predict"):
    sequence = [token.strip() for token in sequence_input.replace(",", " ").split() if token.strip()]

    try:
        result = predict_failure(sequence)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
    else:
        probability = result["anomaly_probability"]
        threshold = result["decision_threshold"]
        prediction = "Failure/Anomaly" if result["predicted_failure"] else "Normal/Success"

        st.metric("Anomaly Probability", f"{probability:.4f}")
        st.metric("Decision Threshold", f"{threshold:.4f}")
        st.subheader(f"Final Prediction: {prediction}")

        if result["unknown_event_ids"]:
            st.warning(f"Unknown EventIds detected: {result['unknown_event_ids']}")
        else:
            st.success("All EventIds are known to the model vocabulary.")
