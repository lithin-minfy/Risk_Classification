import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.title("🏦 Personal Loan Prediction")

# Load trained model
model_path = 'trained_pipeline.pkl'
if not os.path.exists(model_path):
    st.error("🚫 trained_pipeline.pkl not found.")
    st.stop()

model = joblib.load(model_path)

# Upload section
uploaded_file = st.file_uploader("📁 Upload a CSV file (raw input format)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("📄 Uploaded Data")
        st.dataframe(df.head())

        # ✅ Manual preprocessing to match training steps

        # Drop unused columns
        df.drop(['ID', 'ZIP Code', 'Experience'], axis=1, inplace=True)

        # Log-transform skewed columns
        for col in ['Income', 'CCAvg', 'Mortgage']:
            df[col] = np.log1p(df[col])

        # Create HasMortgage
        df['HasMortgage'] = df['Mortgage'].apply(lambda x: 1 if x > 0 else 0)

        # Drop original Mortgage
        df.drop('Mortgage', axis=1, inplace=True)

        # Make prediction
        df['Prediction'] = model.predict(df)

        st.subheader("📊 Prediction Results")
        st.dataframe(df)

        # Download option
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download CSV with Predictions", csv, "predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"⚠️ Error while processing file: {str(e)}")
else:
    st.info("Upload a raw data CSV file to see predictions.")
