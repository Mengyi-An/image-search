"""
Evaluate text -> image search with Recall@K on COCO val2017.

COCO gives every image ~5 human-written captions, which are a free test set:
    use each caption (25014 in total) as a search query,
    and check whether its own image is ranked in the top K of the 5000 images.
    Recall@K = propotion of captions whose image is in the top K.

Comparison experiment: plain CLIP vectors vs centered vectors (see src/embed.py).

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
batch_size = 1000


def compute_ranks(queries, targets, gallery):
    """
    function: for every caption, find the rank of its correct image among all images
    rank 0 = the correct image is the best match

    computed batch by batch, so we never hold the full 25014 x 5000 score matrix
    """
    ranks = []
    n_queries = len(queries)

    for i in range(0, n_queries, batch_size):
        batch_queries = queries[i: i+batch_size]

        batch_scores = batch_queries @ gallery.T

        for row in range(len(batch_queries)):
            row_scores = batch_scores[row]

            # targets 是一个一维数组，targets[k] 表示第 k 句 caption 对应的图片下标。
            # i 是这一批的起始位置（0、1000、2000……），row 是这句 caption 在这一批里的第几行，所以 i + row 就是它在全部 caption 中的位置，targets[i + row] 就是它对应的正确图片。
            target = targets[i + row] # targets is imported from embed.py
            correct_score = row_scores[target]

            # rank = how many images score higher than the correct one
            n_better = np.sum(row_scores > correct_score)
            ranks.append(n_better)

    return np.array(ranks)


def summarize(ranks):
    """
    function: Recall@K
    对单句 caption 来说，排名越小越好
    """
    result = {}
    for k in ks:
        in_top_k = ranks < k
        result[f"recall@{k}"] = float(in_top_k.mean())
    return result


if __name__ == "__main__":
    items = load_captions()
    texts, targets = flatten_captions(items)
    image_embeds = load_image_embeds()
    caption_embeds = load_caption_embeds()
    image_mean, text_mean = load_means()

    print(f"Searching with {len(texts)} captions over {len(items)} images ...")

    ranks = compute_ranks(caption_embeds, targets, image_embeds)
    plain = summarize(ranks)

    centered_queries = center(caption_embeds, text_mean)
    centered_gallery = center(image_embeds, image_mean)
    ranks = compute_ranks(centered_queries, targets, centered_gallery)
    centered = summarize(ranks)

    print(f"{'':12s} {'plain':>8s} {'centered':>9s}")
    for name in plain:
        print(f"{name:12s} {plain[name]:8.4f} {centered[name]:9.4f}")

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
