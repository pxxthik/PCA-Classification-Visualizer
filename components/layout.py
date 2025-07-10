import streamlit as st


def render_header():
    st.markdown(
        """
    <div class="main-header">
        <h1>🎯 Interactive PCA Classification Visualizer</h1>
        <p style="color: white; text-align: center; margin: 0;">
            Upload your dataset and explore how features affect classification in 3D space
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_metrics(df):
    st.subheader("📊 Data Overview")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📈 Samples", f"{len(df):,}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🔢 Features", f"{df.shape[1] - 1}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🎯 Classes", f"{df.iloc[:, -1].nunique()}")
        st.markdown("</div>", unsafe_allow_html=True)


def render_sidebar_sliders(df, feature_columns):
    st.sidebar.markdown("## 🎛️ Interactive Data Point")
    st.sidebar.markdown("Adjust feature values to explore the classification space")

    feature_stats = df[feature_columns].describe()
    tunable_features = {}

    if len(feature_columns) > 6:
        features_per_group = 6
        feature_groups = [
            feature_columns[i : i + features_per_group]
            for i in range(0, len(feature_columns), features_per_group)
        ]

        for group_idx, feature_group in enumerate(feature_groups):
            if len(feature_groups) > 1:
                st.sidebar.markdown(
                    f"### 📊 Features {group_idx * features_per_group + 1}-{min((group_idx + 1) * features_per_group, len(feature_columns))}"
                )

            for feature in feature_group:
                min_val = float(feature_stats.loc["min", feature])
                max_val = float(feature_stats.loc["max", feature])
                mean_val = float(feature_stats.loc["mean", feature])
                std_val = float(feature_stats.loc["std", feature])

                slider_min = min_val - std_val
                slider_max = max_val + std_val

                tunable_features[feature] = st.sidebar.slider(
                    f"🔹 {feature}",
                    min_value=slider_min,
                    max_value=slider_max,
                    value=mean_val,
                    step=(slider_max - slider_min) / 100,
                    help=f"Range: {min_val:.2f} to {max_val:.2f}, Mean: {mean_val:.2f}",
                )
    else:
        for feature in feature_columns:
            min_val = float(feature_stats.loc["min", feature])
            max_val = float(feature_stats.loc["max", feature])
            mean_val = float(feature_stats.loc["mean", feature])
            std_val = float(feature_stats.loc["std", feature])

            slider_min = min_val - std_val
            slider_max = max_val + std_val

            tunable_features[feature] = st.sidebar.slider(
                f"🔹 {feature}",
                min_value=slider_min,
                max_value=slider_max,
                value=mean_val,
                step=(slider_max - slider_min) / 100,
                help=f"Range: {min_val:.2f} to {max_val:.2f}, Mean: {mean_val:.2f}",
            )

    if st.sidebar.button("🔄 Reset to Mean Values"):
        st.rerun()

    return tunable_features
