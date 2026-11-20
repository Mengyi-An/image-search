"""
Draw the figures for the README.

1. figures/recall.png: Recall@1/5/10 of text -> image search (from data/metrics.json,
   written by src/evaluate.py)
2. figures/search_examples.png: a few text queries, each with its top 5 images

Usage:
    python -m src.plot
"""

import os
import json
import matplotlib.pyplot as plt
from PIL import Image
from src.data import load_captions
from src.model import load_model
from src.embed import load_embeds
from src.evaluate import metrics_path
from src.search import search_by_text

figures_path = os.path.join(os.path.dirname(__file__), "..", "figures")

example_queries = [
    "a dog on the beach",
    "a man riding a wave on a surfboard",
    "a plate of food with broccoli",
    "a red double decker bus",
]
n_results = 5

bar_color = "#2a78d6"
text_color = "#52514e"


def load_metrics():
    with open(metrics_path) as f:
        return json.load(f)


def plot_recall(metrics):
    recalls = metrics["text_to_image"]

    names = []
    values = []
    for name in recalls:
        names.append(name.replace("recall", "Recall"))
        values.append(recalls[name])

    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(names, values, width=0.5, color=bar_color)

    # write the value above each bar
    for i in range(len(bars)):
        bar = bars[i]
        value = values[i]
        x = bar.get_x() + bar.get_width() / 2
        label = f"{value:.1%}"
        ax.text(x, value + 0.02, label, ha="center", color=text_color)

    ax.set_ylim(0, 1)
    ax.set_ylabel("share of captions", color=text_color)
    n_queries = metrics["n_queries"]
    n_images = metrics["n_images"]
    ax.set_title(f"Text -> image search on COCO val2017\n({n_queries} captions, {n_images} images)")

    # keep the axes quiet: no top/right frame, light grid behind the bars
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="#e5e5e5")
    ax.set_axisbelow(True)

    out_path = os.path.join(figures_path, "recall.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"saved to {out_path}")


def plot_search_examples(items, model, processor, embeds):
    n_rows = len(example_queries)
    fig, axes = plt.subplots(n_rows, n_results, figsize=(2.2 * n_results, 2.4 * n_rows))

    for row in range(n_rows):
        query = example_queries[row]
        indices, scores = search_by_text(query, model, processor, embeds, k=n_results)

        for col in range(n_results):
            ax = axes[row][col]
            index = indices[col]
            score = scores[col]
            path = items[index]["path"]
            image = Image.open(path).convert("RGB")

            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(f"#{col + 1}  score {score:.2f}", fontsize=8, color=text_color)

        # the query text on the left of each row
        axes[row][0].set_ylabel(query, fontsize=9, rotation=0, ha="right", va="center")

    fig.suptitle("Text -> image search examples (top 5 of 5000 images)")
    fig.tight_layout()

    out_path = os.path.join(figures_path, "search_examples.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved to {out_path}")


if __name__ == "__main__":
    os.makedirs(figures_path, exist_ok=True)

    metrics = load_metrics()
    plot_recall(metrics)

    items = load_captions()
    embeds = load_embeds()
    model, processor = load_model()
    plot_search_examples(items, model, processor, embeds)
