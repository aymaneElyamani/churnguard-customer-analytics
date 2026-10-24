import os
import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="ChurnGuard | Customer Analytics",
    page_icon="📊",
    layout="wide",
)

st.title("ChurnGuard")
st.caption("Customer Analytics · Return Prediction")
st.divider()

st.subheader("Predict customer return")
st.write(
    "Enter customer behavior observed during the previous 180 days."
)

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        frequency = st.number_input(
            "Orders in 180 days",
            min_value=1,
            max_value=100,
            value=1,
            step=1,
        )

    with col2:
        amount = st.number_input(
            "Total item amount",
            min_value=0.0,
            max_value=1_000_000.0,
            value=100.0,
            step=10.0,
        )

    with col3:
        recency = st.number_input(
            "Days since last order",
            min_value=0,
            max_value=180,
            value=45,
            step=1,
        )

    submitted = st.form_submit_button(
        "Predict customer return",
        use_container_width=True,
    )

if submitted:
    payload = {
        "frequence_180j": int(frequency),
        "montant_articles_180j": float(amount),
        "recence_jours": int(recency),
    }

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=10,
        )
        response.raise_for_status()
        result = response.json()

        st.divider()
        st.subheader("Prediction result")

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "Predicted class",
            result["prediction_label"].replace("_", " ").title(),
        )
        metric2.metric(
            "Model score",
            f'{result["return_probability"]:.2%}',
        )
        metric3.metric(
            "Decision threshold",
            f'{result["threshold"]:.0%}',
        )

        if result["prediction"] == 1:
            st.success(
                "The model assigns this customer a score at or above "
                "the current threshold for an observed return."
            )
        else:
            st.info(
                "The model assigns this customer a score below "
                "the current threshold for an observed return."
            )

        st.warning(
            "Prototype limitation: this score is not calibrated and "
            "does not guarantee future behavior. Predictions outside "
            "the training data distribution may be unreliable."
        )

    except requests.RequestException as error:
        st.error(
            "Could not reach the prediction API. "
            "Make sure FastAPI is running at http://127.0.0.1:8000."
        )
        st.caption(str(error))

st.divider()
st.caption(
    "ChurnGuard · Olist dataset · Research and portfolio prototype"
)
