import os
import requests
import streamlit as st

# =====================================================
# Page Configuration
# =====================================================
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide",
)

# =====================================================
# Feature Information
# =====================================================
feature_info = [
    ("Mean Radius", "mm"),
    ("Mean Texture", ""),
    ("Mean Perimeter", "mm"),
    ("Mean Area", "mm²"),
    ("Mean Smoothness", ""),
    ("Mean Compactness", ""),
    ("Mean Concavity", ""),
    ("Mean Concave Points", ""),
    ("Mean Symmetry", ""),
    ("Mean Fractal Dimension", ""),

    ("Radius Error", "mm"),
    ("Texture Error", ""),
    ("Perimeter Error", "mm"),
    ("Area Error", "mm²"),
    ("Smoothness Error", ""),
    ("Compactness Error", ""),
    ("Concavity Error", ""),
    ("Concave Points Error", ""),
    ("Symmetry Error", ""),
    ("Fractal Dimension Error", ""),

    ("Worst Radius", "mm"),
    ("Worst Texture", ""),
    ("Worst Perimeter", "mm"),
    ("Worst Area", "mm²"),
    ("Worst Smoothness", ""),
    ("Worst Compactness", ""),
    ("Worst Concavity", ""),
    ("Worst Concave Points", ""),
    ("Worst Symmetry", ""),
    ("Worst Fractal Dimension", ""),
]

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
        color:#E53935;
        margin-bottom:0;
    }

    .subtitle{
        text-align:center;
        color:gray;
        margin-top:0;
        margin-bottom:30px;
    }

    .stButton>button{
        height:55px;
        font-size:18px;
        font-weight:bold;
        border-radius:10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# =====================================================
# Header
# =====================================================
st.markdown(
    "<h1 class='main-title'>🩺 Breast Cancer Prediction</h1>",
    unsafe_allow_html=True,
)

st.markdown(
    "<p class='subtitle'>Support Vector Machine (SVM) Classifier</p>",
    unsafe_allow_html=True,
)

# =====================================================
# Sidebar
# =====================================================
with st.sidebar:

    st.header("About")

    st.write("""
This application predicts whether a breast tumor is **Benign** or **Malignant**
using a trained Support Vector Machine (SVM) model.
""")

    st.info("""
### Instructions

1. Enter all patient features.
2. Click **Predict**.
3. Review the prediction and confidence score.
""")

# =====================================================
# Input Features
# =====================================================
st.header("Patient Diagnostic Features")
st.caption("Enter the measurements obtained from the patient's diagnostic examination.")

inputs = []


def render_inputs(start, end):
    cols = st.columns(2)

    for i in range(start, end):

        feature, unit = feature_info[i]

        with cols[(i - start) % 2]:

            label = feature if unit == "" else f"{feature} ({unit})"

            value = st.number_input(
                label=label,
                value=0.0,
                format="%.5f",
                help=f"Input value for {feature}",
                key=feature,
            )

            inputs.append(value)


with st.expander("📊 Mean Features", expanded=True):
    render_inputs(0, 10)

with st.expander("📈 Error Features"):
    render_inputs(10, 20)

with st.expander("📉 Worst Features"):
    render_inputs(20, 30)

st.divider()

# =====================================================
# Predict Button
# =====================================================
_, center, _ = st.columns([1, 1, 1])

with center:

    predict = st.button(
        "🔍 Predict",
        use_container_width=True,
        type="primary",
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

        with st.spinner("Running prediction..."):

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

        left, right = st.columns([2, 1])

        with left:

            with st.container(border=True):

                if prediction in [1, "Malignant", "malignant"]:

                    st.error("## 🔴 Malignant")

                    st.write("""
The model predicts that the breast tumor is **Malignant**.

Clinical evaluation by a qualified healthcare professional is recommended.
""")

                else:

                    st.success("## 🟢 Benign")

                    st.write("""
The model predicts that the breast tumor is **Benign**.
""")

        with right:

            with st.container(border=True):

                st.metric(
                    "Confidence",
                    f"{probability:.2%}",
                )

                st.progress(
                    min(max(probability, 0), 1)
                )

        with st.expander("Raw API Response"):

            st.json(result)

    except requests.exceptions.ConnectionError:

        st.error(
            "Unable to connect to the FastAPI server.\n\n"
            "Please ensure the API server is running."
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