import os
import requests
import streamlit as st

st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide",
)

# =========================
# Custom CSS
# =========================
st.markdown("""
<style>
.main-title{
    text-align:center;
    font-size:40px;
    font-weight:bold;
    color:#ff4b4b;
}

.sub-title{
    text-align:center;
    color:gray;
    margin-bottom:20px;
}

.result-box{
    padding:20px;
    border-radius:12px;
    background-color:#f7f7f7;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Header
# =========================
st.markdown('<p class="main-title">🩺 Breast Cancer Prediction</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Support Vector Machine (SVM) Classification</p>',
    unsafe_allow_html=True
)

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This application predicts whether a breast tumor is **Benign** or **Malignant**
    using a trained Support Vector Machine model.

    **Instructions**
    - Fill all 30 features.
    - Click **Predict**.
    """)

# =========================
# Input Form
# =========================
st.header("Input Features")

inputs = []

cols = st.columns(3)

for i in range(30):
    with cols[i % 3]:
        value = st.number_input(
            f"Feature {i+1}",
            value=0.0,
            format="%.4f"
        )
        inputs.append(value)

st.divider()

# =========================
# Predict Button
# =========================
col1, col2, col3 = st.columns([1,1,1])

with col2:
    predict = st.button(
        "🔍 Predict",
        use_container_width=True,
        type="primary"
    )

# =========================
# Prediction
# =========================
if predict:

    url = os.environ.get(
        "API_URL",
        "http://localhost:8000/predict"
    )

    try:

        with st.spinner("Predicting..."):

            response = requests.post(
                url,
                json={"features": inputs},
                timeout=10
            )

            response.raise_for_status()

            hasil = response.json()

        prediction = hasil["prediction"]
        probability = hasil["probability"]

        st.divider()
        st.header("Prediction Result")

        with st.container(border=True):

            if prediction == 1 or str(prediction).lower() == "malignant":
                st.error("🔴 Malignant")
            else:
                st.success("🟢 Benign")

            st.metric(
                "Confidence",
                f"{probability:.2%}"
            )

            st.progress(float(probability))

            st.write("### Raw Output")
            st.json(hasil)

    except Exception as e:
        st.error(f"Failed to connect to API\n\n{e}")