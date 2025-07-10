import streamlit as st
import pandas as pd
import numpy as np

# Import from config
from config import CUSTOM_CSS, MODERN_COLORS

# Import from utils
from utils.loader import load_and_preprocess_data
from utils.pca_utils import apply_pca_and_scaling
from utils.plot_utils import (
    create_pca_3d_plot,
    create_feature_loadings_heatmap,
    create_pca_variance_chart,
)

# Import from components
from components.layout import render_header, render_metrics, render_sidebar_sliders
from components.instructions import render_landing_page

st.set_page_config(page_title="PCA Classification Visualizer", layout="wide")

# Apply custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

render_header()

uploaded_file = st.file_uploader(
    "📁 Choose a CSV file",
    type="csv",
    help="Upload a CSV file with numerical features and one target column",
)

if uploaded_file is not None:
    df = load_and_preprocess_data(uploaded_file)

    if df is None:
        st.stop()

    render_metrics(df)

    # Show data preview
    with st.expander("🔍 View Data Preview", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)
        st.write("**Data Types:**")
        st.write(df.dtypes)

    feature_columns = df.columns[:-1].tolist()
    target_column = df.columns[-1]

    X = df[feature_columns].values
    y = df[target_column].values

    # Convert target to string for better handling
    y_str = y.astype(str)
    unique_classes = np.unique(y_str)
    n_classes = len(unique_classes)

    try:
        X = X.astype(float)
    except:
        st.error("All feature columns must contain numerical values only.")
        st.stop()

    # Get the pipeline and X_pca from apply_pca_and_scaling
    X_pca, pipeline, _ = apply_pca_and_scaling(
        X, feature_columns
    )  # Corrected to receive pipeline object

    # Access the pca and scaler objects from the pipeline
    pca_model = pipeline.named_steps["pca"]
    scaler_model = pipeline.named_steps["scaler"]

    # Create PCA DataFrame
    n_components = min(3, len(feature_columns))
    pca_columns = [f"PC{i+1}" for i in range(n_components)]
    pca_df = pd.DataFrame(X_pca, columns=pca_columns)
    pca_df["Class"] = y_str

    # PCA Information
    st.subheader("🔬 PCA Analysis")
    explained_variance = pca_model.explained_variance_ratio_  # Use pca_model here
    cumulative_variance = np.cumsum(explained_variance)

    col1, col2 = st.columns(2)

    with col1:
        # No need for st.session_state here, as the chart is created on each run
        fig_var = create_pca_variance_chart(
            pca_columns, explained_variance, cumulative_variance
        )
        st.plotly_chart(fig_var, use_container_width=True)

    with col2:
        st.write("**📋 PCA Summary:**")
        for i, (pc, var) in enumerate(zip(pca_columns, explained_variance)):
            st.write(f"• **{pc}**: {var:.1%} variance")
        st.write(f"• **Total Explained**: {sum(explained_variance):.1%}")

        if sum(explained_variance) < 0.8:
            st.warning(
                "⚠️ First 3 components explain less than 80% of variance. Consider the limitations of 3D visualization."
            )

    # Sidebar for tunable data point
    tunable_features = render_sidebar_sliders(df, feature_columns)

    # Transform the tunable point
    tunable_point = np.array(
        [tunable_features[feature] for feature in feature_columns]
    ).reshape(1, -1)
    tunable_point_scaled = scaler_model.transform(
        tunable_point
    )  # Use scaler_model here
    tunable_point_pca = pca_model.transform(tunable_point_scaled)  # Use pca_model here

    # Determine colors based on number of classes
    if n_classes <= len(MODERN_COLORS):
        colors = MODERN_COLORS[:n_classes]
    else:
        import plotly.express as px

        colors = px.colors.sample_colorscale("plasma", n_classes)

    # color_map = {class_val: colors[i] for i, class_val in enumerate(unique_classes)} # This variable is not used

    fig = create_pca_3d_plot(
        pca_df,
        tunable_point_pca,
        n_components,
        unique_classes,
        colors,
        explained_variance,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Current point information
    st.subheader("🎯 Current Interactive Point")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🔧 Original Feature Values:**")
        feature_stats = df[feature_columns].describe()
        for feature, value in tunable_features.items():
            mean_val = feature_stats.loc["mean", feature]
            diff = value - mean_val
            direction = "↑" if diff > 0 else "↓" if diff < 0 else "="
            st.write(f"• **{feature}**: {value:.3f} {direction}")

    with col2:
        st.markdown("**📍 PCA Coordinates:**")
        for i, coord in enumerate(tunable_point_pca[0]):
            st.write(f"• **PC{i+1}**: {coord:.3f}")

    # Feature importance analysis
    st.subheader("🔍 Feature Analysis")

    loadings_df = pd.DataFrame(
        pca_model.components_.T,  # Use pca_model here
        columns=pca_columns,
        index=feature_columns,
    )

    fig_heatmap = create_feature_loadings_heatmap(loadings_df)
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # Top contributing features
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🏆 Top Features for Each Component:**")
        for i, pc in enumerate(pca_columns):
            abs_loadings = np.abs(loadings_df[pc])
            top_features = abs_loadings.nlargest(3)
            st.write(f"**{pc}:**")
            for feature, loading in top_features.items():
                st.write(f"  • {feature}: {loading:.3f}")

    with col2:
        # Feature importance summary
        feature_importance = (
            np.abs(loadings_df).sum(axis=1).sort_values(ascending=False)
        )
        st.markdown("**📊 Overall Feature Importance:**")
        for feature, importance in feature_importance.head(5).items():
            st.write(f"• **{feature}**: {importance:.3f}")

    # Additional statistics
    with st.expander("📈 Detailed Statistics", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Feature Loadings Table:**")
            st.dataframe(loadings_df.round(4))

        with col2:
            st.write("**Class Distribution:**")
            class_counts = pd.Series(y_str).value_counts()
            st.dataframe(class_counts.to_frame("Count"))

else:
    render_landing_page()
