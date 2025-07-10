import streamlit as st
import pandas as pd


def render_landing_page():
    st.markdown(
        """
    ## 🚀 Getting Started

    Upload your CSV file to begin exploring your data with interactive PCA visualization!

    ### 📋 Requirements:
    - **CSV format** with numerical features
    - **Any number of input features** (minimum 1)
    - **One target/output column** (last column)
    - **Numerical data only** (categorical data should be encoded)

    ### ✨ Features:
    - **🎯 Interactive 3D PCA Visualization** - Explore your data in reduced dimensional space
    - **🎛️ Real-time Feature Tuning** - Adjust any feature and see immediate effects
    - **🎨 Enhanced Visuals** - Beautiful, professional-grade plots
    - **📊 Comprehensive Analysis** - Feature importance, loadings, and statistics
    - **🔍 Smart Scaling** - Automatic handling of different feature scales
    - **📈 Variance Analysis** - Understand how much information each component captures

    ### 🎨 Visualization Features:
    - Different colors for each class
    - Existing data points in semi-transparent colors
    - Interactive tunable point highlighted in red
    - 3D rotation and zoom capabilities
    - Detailed hover information
    - Professional styling and layout

    ### 💡 Use Cases:
    - **Classification Analysis** - Understand class separability
    - **Feature Engineering** - Identify important features
    - **Data Exploration** - Discover patterns and clusters
    - **Model Validation** - Assess classification difficulty
    - **Educational** - Learn about PCA and dimensionality reduction
    """
    )

    st.markdown("### 📝 Sample Data Format:")
    sample_data = {
        "Feature_1": [1.2, 2.3, 1.8, 3.1],
        "Feature_2": [0.8, 1.5, 2.2, 0.9],
        "Feature_3": [2.1, 1.7, 3.0, 2.8],
        "Target": ["Class_A", "Class_B", "Class_A", "Class_B"],
    }
    st.dataframe(pd.DataFrame(sample_data), use_container_width=True)
