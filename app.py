"""
Titanic Survival Prediction - Streamlit App
============================================
A web application to predict passenger survival on the Titanic.

Usage: streamlit run app.py
"""

import streamlit as st
import joblib
import pandas as pd

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# ==============================================================================
# LOAD MODEL
# ==============================================================================
@st.cache_resource
def load_model():
    """Load the trained model pipeline."""
    return joblib.load("model/titanic_survival_model.pkl")

model = load_model()

# ==============================================================================
# APP HEADER
# ==============================================================================
st.title("🚢 Titanic Survival Predictor")
st.markdown("""
This app predicts whether a passenger would have survived the Titanic disaster 
based on their characteristics.

**Enter the passenger details below:**
""")

st.divider()

# ==============================================================================
# USER INPUT FORM
# ==============================================================================
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox(
        "Passenger Class",
        options=[1, 2, 3],
        format_func=lambda x: f"{x}st Class" if x == 1 else f"{x}nd Class" if x == 2 else f"{x}rd Class",
        help="1st = Upper, 2nd = Middle, 3rd = Lower"
    )
    
    sex = st.selectbox(
        "Sex",
        options=["male", "female"],
        format_func=lambda x: x.capitalize()
    )
    
    age = st.number_input(
        "Age",
        min_value=0.0,
        max_value=100.0,
        value=30.0,
        step=1.0,
        help="Age in years"
    )

with col2:
    fare = st.number_input(
        "Fare (£)",
        min_value=0.0,
        max_value=600.0,
        value=32.0,
        step=1.0,
        help="Passenger fare in British Pounds"
    )
    
    embarked = st.selectbox(
        "Port of Embarkation",
        options=["S", "C", "Q"],
        format_func=lambda x: {
            "S": "Southampton",
            "C": "Cherbourg",
            "Q": "Queenstown"
        }[x]
    )

st.divider()

# ==============================================================================
# PREDICTION
# ==============================================================================
if st.button("🔮 Predict Survival", type="primary", use_container_width=True):
    
    # Create input DataFrame with exact column names expected by the model
    input_df = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "Fare": fare,
        "Embarked": embarked
    }])
    
    # Make prediction (pipeline handles all preprocessing automatically)
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]
    
    # Display result
    st.markdown("### Prediction Result")
    
    if prediction == 1:
        st.success("✅ **SURVIVED**")
        st.balloons()
    else:
        st.error("❌ **DID NOT SURVIVE**")
    
    # Display probabilities
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label="Survival Probability",
            value=f"{prediction_proba[1] * 100:.1f}%"
        )
    with col2:
        st.metric(
            label="Non-Survival Probability",
            value=f"{prediction_proba[0] * 100:.1f}%"
        )
    
    # Show input summary
    with st.expander("📋 View Input Summary"):
        st.dataframe(input_df, use_container_width=True)

# ==============================================================================
# FOOTER
# ==============================================================================
st.divider()
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.8em;">
    <p>Built with Streamlit | Model: Logistic Regression with Pipeline</p>
    <p>CSC334 - Python Programming Language II</p>
</div>
""", unsafe_allow_html=True)