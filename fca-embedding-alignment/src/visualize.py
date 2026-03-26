import numpy as np
import plotly.graph_objects as go


from .evaluate import project_word


def plot_curves(history):
    fig = go.Figure()
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig.add_trace(go.Scatter(
        x=epochs, y=history["train_loss"], name="train loss"))
    fig.add_trace(go.Scatter(
        x=epochs, y=history["val_loss"],   name="val loss"))
    if "ortho" in history:
        fig.add_trace(go.Scatter(
            x=epochs, y=history["ortho"], name="ortho loss"))
    fig.update_layout(title="Training curves",
                      xaxis_title="Epoch", yaxis_title="Loss")
    fig.show()
    return fig


def visualize_3d_projection(gender_pairs, royalty_pairs, embeddings, projector,
                            condition=None, device="cpu", output_html="results/graph_3d.html",
                            title="Semantic Projection 3D"):
    all_words = {w for pair in gender_pairs + royalty_pairs for w in pair}
    word_to_3d = {
        w: project_word(w, embeddings, projector,
                        condition=condition, device=device)
        for w in all_words
    }

    gender_ws = {w for pair in gender_pairs for w in pair}
    royalty_ws = {w for pair in royalty_pairs for w in pair}

    fig = go.Figure()

    # edges
    for pairs, color in [(gender_pairs, "rgba(0,0,255,0.15)"), (royalty_pairs, "rgba(255,100,0,0.15)")]:
        for w1, w2 in pairs:
            p1, p2 = word_to_3d[w1], word_to_3d[w2]
            fig.add_trace(go.Scatter3d(
                x=[p1[0], p2[0], None], y=[p1[1], p2[1], None], z=[p1[2], p2[2], None],
                mode="lines", line=dict(color=color, width=2),
                showlegend=False, hoverinfo="skip",
            ))

    # nodes
    categories = {
        "Gender only":  (gender_ws - royalty_ws, "blue"),
        "Royalty only": (royalty_ws - gender_ws, "orange"),
        "Both":         (gender_ws & royalty_ws, "green"),
    }
    for name, (words, color) in categories.items():
        if not words:
            continue
        wlist = sorted(words)
        coords = np.array([word_to_3d[w] for w in wlist])
        fig.add_trace(go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            mode="markers+text",
            marker=dict(size=5, color=color, opacity=0.8),
            text=wlist, textposition="top center", textfont=dict(size=7),
            name=name,
            hovertemplate="<b>%{text}</b><br>X:%{x:.3f} Y:%{y:.3f} Z:%{z:.3f}<extra></extra>",
        ))

    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="Axis 1 (gender)",
                   yaxis_title="Axis 2 (status)", zaxis_title="Axis 3"),
        width=1100, height=750,
    )
    fig.write_html(output_html)
    print(f"Saved: {output_html}")
    fig.show()
    return fig
