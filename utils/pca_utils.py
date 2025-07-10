from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import streamlit as st


def apply_pca_and_scaling(X, feature_columns):
    n_components = min(3, len(feature_columns))

    # Create a pipeline for scaling and PCA
    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("pca", PCA(n_components=n_components))]
    )

    X_pca = pipeline.fit_transform(X)

    return X_pca, pipeline, X_pca
