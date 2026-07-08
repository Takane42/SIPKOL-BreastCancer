import os
import requests
import streamlit as st
from sklearn.datasets import load_breast_cancer

# =====================================================
# Page Configuration
# =====================================================
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide",
)

# =====================================================
# Load Feature Names
# =====================================================
dataset = load_breast_cancer()
feature_names = [name.title() for name in dataset.feature_names]

# =====================================================
# Custom CSS
# =====================================================
st.markdown(
    """
    <style>
    .main-title{
        text-align:center;
        font-size:42px;
        font-weight:bold;
        color:#ff4b4b;
    }

    .subtitle{
        text-align:center;
        color:#888888;
        margin-bottom:30px;
    }

    .stButton > button{
        height:3em;
        font-size:18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =====================================================
# Header
# =====================================================
st.markdown(
    '<p class="main-title">🩺 Breast Cancer Prediction</p>',
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="subtitle">Support Vector Machine (SVM) Classifier</p>',
    unsafe_allow_html=True,
)

# =====================================================
# Sidebar
# =====================================================
with st.sidebar:
    st.header("About")

    st.write(
        """
This application predicts whether a breast tumor is **Benign** or
**Malignant** using a trained Support Vector Machine (SVM) model.
"""
    )

    st.info(
        """
**Instructions**

1. Enter all 30 feature values.
2. Click **Predict**.
3. View the prediction result and confidence score.
"""
    )

# =====================================================
# Input Features
# =====================================================
st.header("Patient Features")

st.caption("Enter the diagnostic measurements below.")

inputs = []

cols = st.columns(3)

for i, feature in enumerate(feature_names):
    with cols[i % 3]:
        value = st.number_input(
            label=feature,
            value=0.0,
            format="%.5f",
            help=f"Input value for {feature}",
        )
        inputs.append(value)

st.divider()

# =====================================================
# Predict Button
# =====================================================
left, center, right = st.columns([1, 1, 1])

with center:
    predict = st.button(
        "🔍 Predict",
        type="primary",
        use_container_width=True,
    )

# =====================================================
# Prediction
# =====================================================
if predict:

    api_url = os.environ.get(
        "API_URL",
        "http://localhost:8000/predict",
    )

    try:

        with st.spinner("Predicting..."):

            response = requests.post(
                api_url,
                json={
                    "features": inputs
                },
                timeout=30,
            )

            response.raise_for_status()

            result = response.json()

        prediction = result["prediction"]
        probability = float(result["probability"])

        st.divider()

        st.header("Prediction Result")

        col1, col2 = st.columns([2, 1])

        with col1:

            with st.container(border=True):

                if prediction in [1, "Malignant", "malignant"]:

                    st.error("### 🔴 Malignant")

                    st.write(
                        """
The model predicts that the tumor is **Malignant**.
Further clinical evaluation is recommended.
"""
                    )

                else:

                    st.success("### 🟢 Benign")

                    st.write(
                        """
The model predicts that the tumor is **Benign**.
"""
                    )

        with col2:

            with st.container(border=True):

                st.metric(
                    label="Confidence",
                    value=f"{probability:.2%}",
                )

                st.progress(min(max(probability, 0.0), 1.0))

        with st.expander("View Raw API Response"):

            st.json(result)

    except requests.exceptions.ConnectionError:

        st.error(
            "Unable to connect to the FastAPI server.\n\n"
            "Make sure the API is running and API_URL is correct."
        )

    except requests.exceptions.Timeout:

        st.error("The request timed out.")

    except requests.exceptions.HTTPError:

        try:
            st.error(response.json())
        except Exception:
            st.error(response.text)

    except Exception as e:

        st.exception(e)