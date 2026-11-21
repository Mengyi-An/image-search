"""
Evaluate text -> image search with Recall@K on COCO val2017.

COCO gives every image ~5 human-written captions, which are a free test set:
    use each caption (25014 in total) as a search query,
    and check whether its own image is ranked in the top K of the 5000 images.
    Recall@K = share of captions whose image is in the top K.

Comparison experiment: plain CLIP vectors vs centered vectors (see src/embed.py).
Besides Recall@K, it counts how often each image appears in the top 10 of all searches:
a few "hub" images that appear in almost every result list make the search worse.

Output: data/metrics.json

Usage:
    python -m src.evaluate
"""

import os
import json
import numpy as np
from src.data import data_path, load_captions
from src.model import model_name
from src.embed import load_image_embeds, load_caption_embeds, load_means, flatten_captions, center

metrics_path = os.path.join(data_path, "metrics.json")
ks = [1, 5, 10]
hub_k = 10
batch_size = 1000


def compute_ranks(queries, targets, gallery):
    """
    function: for every caption, find the rank of its correct image among all images
    rank 0 = the correct image is the best match

    also counts how often each image is in the top hub_k of a search
    computed batch by batch, so we never hold the full 25014 x 5000 score matrix
    """
    ranks = []
    top_counts = np.zeros(len(gallery), dtype=int)
    n_queries = len(queries)

    for start in range(0, n_queries, batch_size):
        end = min(start + batch_size, n_queries)

        # (batch, 512) @ (512, 5000) -> (batch, 5000): score of every caption with every image
        scores = queries[start:end] @ gallery.T

        for row in range(end - start):
            row_scores = scores[row]
            target = targets[start + row]
            correct_score = row_scores[target]

            # rank = how many images score higher than the correct one
            n_better = np.sum(row_scores > correct_score)
            ranks.append(n_better)

            # the hub_k best images of this search
            best = np.argsort(-row_scores)[:hub_k]
            for index in best:
                top_counts[index] += 1

    return np.array(ranks), top_counts


def summarize(ranks, top_counts):
    """
    function: Recall@K, plus how uneven the top-10 appearances are
    """
    result = {}
    for k in ks:
        in_top_k = ranks < k
        result[f"recall@{k}"] = float(in_top_k.mean())

    # if every image appeared equally often, each would appear this many times
    fair_share = top_counts.mean()
    result["max_top10_count"] = int(top_counts.max())
    result["fair_top10_count"] = float(fair_share)
    result["never_in_top10"] = int(np.sum(top_counts == 0))
    return result


if __name__ == "__main__":
    items = load_captions()
    texts, targets = flatten_captions(items)
    image_embeds = load_image_embeds()
    caption_embeds = load_caption_embeds()
    image_mean, text_mean = load_means()

    print(f"Searching with {len(texts)} captions over {len(items)} images ...")

    ranks, top_counts = compute_ranks(caption_embeds, targets, image_embeds)
    plain = summarize(ranks, top_counts)

    centered_queries = center(caption_embeds, text_mean)
    centered_gallery = center(image_embeds, image_mean)
    ranks, top_counts = compute_ranks(centered_queries, targets, centered_gallery)
    centered = summarize(ranks, top_counts)

    print(f"{'':28s} {'plain':>8s} {'centered':>9s}")
    for name in plain:
        print(f"{name:28s} {plain[name]:8.4g} {centered[name]:9.4g}")

    metrics = {
        "model": model_name,
        "n_images": len(items),
        "n_queries": len(texts),
        "plain": plain,
        "centered": centered,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"saved to {metrics_path}")
