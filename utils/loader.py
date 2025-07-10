import pandas as pd
import streamlit as st


def load_and_preprocess_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return None

    if df.shape[1] < 2:
        st.error("Dataset must have at least 2 columns (1 feature + 1 target)")
        return None

    if df.shape[1] < 4:
        st.warning(
            "Dataset has less than 3 features. PCA will use all available dimensions."
        )

    if df.isnull().sum().sum() > 0:
        st.warning(
            "⚠️ Dataset contains missing values. They will be filled with column means."
        )
        df = df.fillna(df.mean(numeric_only=True))

    return df
