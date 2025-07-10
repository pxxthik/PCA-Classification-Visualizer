import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import streamlit as st
from config import MODERN_COLORS


def create_pca_variance_chart(pca_columns, explained_variance, cumulative_variance):
    fig_var = go.Figure()

    fig_var.add_trace(
        go.Bar(
            x=pca_columns,
            y=explained_variance,
            name="Individual",
            marker_color="lightblue",
            text=[f"{v:.1%}" for v in explained_variance],
            textposition="auto",
        )
    )

    fig_var.add_trace(
        go.Scatter(
            x=pca_columns,
            y=cumulative_variance,
            mode="lines+markers",
            name="Cumulative",
            line=dict(color="red", width=3),
            marker=dict(size=8),
            yaxis="y2",
        )
    )

    fig_var.update_layout(
        title="📊 Explained Variance by Principal Components",
        xaxis_title="Principal Components",
        yaxis_title="Explained Variance Ratio",
        yaxis2=dict(title="Cumulative Variance", overlaying="y", side="right"),
        showlegend=True,
        height=400,
    )
    return fig_var


def create_pca_3d_plot(
    pca_df, tunable_point_pca, n_components, unique_classes, colors, explained_variance
):
    fig = go.Figure()

    # Add existing data points with beautiful styling
    for i, class_val in enumerate(unique_classes):
        class_data = pca_df[pca_df["Class"] == class_val]
        class_count = len(class_data)

        # Create gradient effect for each class
        base_color = colors[i]

        fig.add_trace(
            go.Scatter3d(
                x=class_data["PC1"],
                y=class_data["PC2"],
                z=class_data["PC3"] if n_components > 2 else [0] * len(class_data),
                mode="markers",
                marker=dict(
                    size=6,
                    color=base_color,
                    opacity=0.7,
                    symbol="circle",
                    line=dict(width=0.5, color="rgba(255,255,255,0.8)"),
                    # Add subtle gradient effect
                    colorscale=[[0, base_color], [1, base_color]],
                    showscale=False,
                ),
                name=f"✨ {class_val} ({class_count})",
                hovertemplate=f'<b style="color:{base_color}">🎯 Class {class_val}</b><br>'
                + "<b>PC1:</b> %{x:.3f}<br>"
                + "<b>PC2:</b> %{y:.3f}<br>"
                + ("<b>PC3:</b> %{z:.3f}<br>" if n_components > 2 else "")
                + f"<b>📊 Samples:</b> {class_count}<br>"
                + "<extra></extra>",
                hoverlabel=dict(
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor=base_color,
                    font=dict(color="black", size=12),
                ),
            )
        )

    # Add interactive tunable point with stunning effects
    fig.add_trace(
        go.Scatter3d(
            x=tunable_point_pca[:, 0],
            y=tunable_point_pca[:, 1],
            z=tunable_point_pca[:, 2] if n_components > 2 else [0],
            mode="markers",
            marker=dict(
                size=16,
                color="#FF1744",  # Bright red
                opacity=1.0,
                symbol="diamond",
                line=dict(width=4, color="#D50000"),
                # Add glowing effect
                colorscale=[[0, "#FF1744"], [0.5, "#FF5722"], [1, "#FF8A65"]],
                showscale=False,
            ),
            name="🎯 Interactive Point",
            hovertemplate='<b style="color:#FF1744">🎯 Interactive Point</b><br>'
            + "<b>PC1:</b> %{x:.3f}<br>"
            + "<b>PC2:</b> %{y:.3f}<br>"
            + ("<b>PC3:</b> %{z:.3f}<br>" if n_components > 2 else "")
            + "🔧 <i>Adjust sliders to move</i><br>"
            + "<extra></extra>",
            hoverlabel=dict(
                bgcolor="rgba(255,255,255,0.95)",
                bordercolor="#FF1744",
                font=dict(color="black", size=13, family="Arial Black"),
            ),
        )
    )

    # Add animated orbit traces around the interactive point for visual appeal
    if n_components >= 2:
        theta = np.linspace(0, 2 * np.pi, 20)
        radius = 0.3
        orbit_x = tunable_point_pca[0, 0] + radius * np.cos(theta)
        orbit_y = tunable_point_pca[0, 1] + radius * np.sin(theta)
        orbit_z = [tunable_point_pca[0, 2] if n_components > 2 else 0] * 20

        fig.add_trace(
            go.Scatter3d(
                x=orbit_x,
                y=orbit_y,
                z=orbit_z,
                mode="lines",
                line=dict(color="rgba(255, 23, 68, 0.3)", width=2, dash="dot"),
                name="🌟 Focus Ring",
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Stunning layout with modern aesthetics and gradients
    fig.update_layout(
        title={
            "text": f"<b>🌟 Interactive 3D PCA Visualization</b><br><sup>({len(pca_df.columns) - 1} features → {n_components} components)</sup>",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 22, "color": "#2C3E50", "family": "Arial Black"},
        },
        scene=dict(
            xaxis_title=f"<b>PC1</b> ({explained_variance[0]:.1%} variance)",
            yaxis_title=f"<b>PC2</b> ({explained_variance[1]:.1%} variance)",
            zaxis_title=(
                f"<b>PC3</b> ({explained_variance[2]:.1%} variance)"
                if n_components > 2
                else "<b>PC3</b> (0% variance)"
            ),
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=1.8),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1),
            ),
            # Modern gradient background
            bgcolor="rgba(248, 249, 250, 0.95)",
            xaxis=dict(
                backgroundcolor="rgba(255, 255, 255, 0.8)",
                gridcolor="rgba(100, 149, 237, 0.3)",
                showbackground=True,
                zerolinecolor="rgba(100, 149, 237, 0.5)",
                showspikes=True,
                spikecolor="rgba(100, 149, 237, 0.6)",
                spikesides=True,
                spikethickness=2,
                titlefont=dict(color="#34495E", size=14, family="Arial Black"),
                tickfont=dict(color="#34495E", size=12),
            ),
            yaxis=dict(
                backgroundcolor="rgba(255, 255, 255, 0.8)",
                gridcolor="rgba(46, 204, 113, 0.3)",
                showbackground=True,
                zerolinecolor="rgba(46, 204, 113, 0.5)",
                showspikes=True,
                spikecolor="rgba(46, 204, 113, 0.6)",
                spikesides=True,
                spikethickness=2,
                titlefont=dict(color="#34495E", size=14, family="Arial Black"),
                tickfont=dict(color="#34495E", size=12),
            ),
            zaxis=dict(
                backgroundcolor="rgba(255, 255, 255, 0.8)",
                gridcolor="rgba(231, 76, 60, 0.3)",
                showbackground=True,
                zerolinecolor="rgba(231, 76, 60, 0.5)",
                showspikes=True,
                spikecolor="rgba(231, 76, 60, 0.6)",
                spikesides=True,
                spikethickness=2,
                titlefont=dict(color="#34495E", size=14, family="Arial Black"),
                tickfont=dict(color="#34495E", size=12),
            ),
            # Add subtle ambient lighting
            aspectmode="cube",
            aspectratio=dict(x=1, y=1, z=1),
        ),
        width=1000,
        height=750,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.02,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="rgba(52, 73, 94, 0.3)",
            borderwidth=2,
            font=dict(color="#2C3E50", size=12, family="Arial"),
            # Add subtle shadow effect
            itemsizing="constant",
            itemwidth=30,
        ),
        # Modern color scheme
        plot_bgcolor="rgba(248, 249, 250, 0.95)",
        paper_bgcolor="rgba(255, 255, 255, 0.95)",
        # Add subtle animation
        transition=dict(duration=300, easing="cubic-in-out"),
    )
    return fig


def create_feature_loadings_heatmap(loadings_df):
    fig_heatmap = go.Figure(
        data=go.Heatmap(
            z=loadings_df.T.values,
            x=loadings_df.index,
            y=loadings_df.columns,
            colorscale="RdYlBu_r",
            hoverongaps=False,
            hovertemplate="<b>%{y}</b><br>"
            + "<b>Feature:</b> %{x}<br>"
            + "<b>Loading:</b> %{z:.3f}<br>"
            + "<extra></extra>",
            colorbar=dict(
                title="Loading Value",
                titleside="right",
                tickmode="linear",
                tick0=-1,
                dtick=0.5,
                thickness=15,
                len=0.7,
            ),
        )
    )

    # Add text annotations for better readability
    for i, pc in enumerate(loadings_df.columns):
        for j, feature in enumerate(loadings_df.index):
            fig_heatmap.add_annotation(
                x=feature,
                y=pc,
                text=f"{loadings_df.loc[feature, pc]:.2f}",
                showarrow=False,
                font=dict(
                    color=(
                        "white" if abs(loadings_df.loc[feature, pc]) > 0.5 else "black"
                    ),
                    size=10,
                    family="Arial",
                ),
            )

    fig_heatmap.update_layout(
        title={
            "text": "<b>🎨 Feature Loadings Heatmap</b><br><sup>Contribution of each feature to principal components</sup>",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 18, "color": "#2C3E50", "family": "Arial Black"},
        },
        height=400,
        xaxis_title="<b>Features</b>",
        yaxis_title="<b>Principal Components</b>",
        xaxis=dict(
            tickangle=45,
            titlefont=dict(color="#34495E", size=14),
            tickfont=dict(color="#34495E", size=11),
        ),
        yaxis=dict(
            titlefont=dict(color="#34495E", size=14),
            tickfont=dict(color="#34495E", size=12),
        ),
        plot_bgcolor="rgba(255, 255, 255, 0.95)",
        paper_bgcolor="rgba(248, 249, 250, 0.95)",
    )
    return fig_heatmap
