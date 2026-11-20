"""
Evaluate text -> image search with Recall@K on COCO val2017.

COCO gives every image ~5 human-written captions, which are a free test set:
    use each caption (25014 in total) as a search query,
    and check whether its own image is ranked in the top K of the 5000 images.
    Recall@K = share of captions whose image is in the top K.

Output: data/metrics.json

Usage:
    python -m src.evaluate
"""

import os
import json
import numpy as np
from src.data import data_path, load_captions
from src.model import load_model, encode_texts, model_name
from src.embed import load_embeds

metrics_path = os.path.join(data_path, "metrics.json")
ks = [1, 5, 10]
batch_size = 256


def caption_queries(items):
    """
    function: flatten all captions into one list, and remember which image each one belongs to

    output:
        texts: ["A man is in a kitchen making pizzas.", ...]   25014 captions
        targets: [0, 0, 0, 0, 0, 1, 1, ...]                     index of the correct image
    """
    texts = []
    targets = []
    for index in range(len(items)):
        captions = items[index]["captions"]
        for caption in captions:
            texts.append(caption)
            targets.append(index)
    return texts, np.array(targets)


def compute_ranks(model, processor, texts, targets, embeds):
    """
    function: for every caption, find the rank of its correct image among all images
    rank 0 = the correct image is the best match

    computed batch by batch, so we never hold the full 25014 x 5000 score matrix
    """
    ranks = []
    n_texts = len(texts)

    for start in range(0, n_texts, batch_size):
        end = min(start + batch_size, n_texts)
        batch_texts = texts[start:end]
        batch_targets = targets[start:end]

        text_vectors = encode_texts(model, processor, batch_texts)

        # (batch, 512) @ (512, 5000) -> (batch, 5000): score of every caption with every image
        scores = text_vectors @ embeds.T

        for row in range(len(batch_texts)):
            row_scores = scores[row]
            target = batch_targets[row]
            correct_score = row_scores[target]

            # rank = how many images score higher than the correct one
            n_better = np.sum(row_scores > correct_score)
            ranks.append(n_better)

        if end % 5120 == 0 or end == n_texts:
            print(f"  {end} / {n_texts} captions")

    return np.array(ranks)


def recall_at_k(ranks):
    """
    function: Recall@K = share of queries whose correct image is in the top K
    """
    recalls = {}
    for k in ks:
        in_top_k = ranks < k
        recall = in_top_k.mean()
        recalls[f"recall@{k}"] = float(recall)
    return recalls


if __name__ == "__main__":
    items = load_captions()
    embeds = load_embeds()
    model, processor = load_model()

    texts, targets = caption_queries(items)
    print(f"Searching with {len(texts)} captions over {len(items)} images ...")
    ranks = compute_ranks(model, processor, texts, targets, embeds)
    recalls = recall_at_k(ranks)

    for name in recalls:
        value = recalls[name]
        print(f"{name}: {value:.4f}")

    metrics = {
        "model": model_name,
        "n_images": len(items),
        "n_queries": len(texts),
        "text_to_image": recalls,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"saved to {metrics_path}")
