"""
Search the COCO images with a text or with an image.

    query (text or image) -> CLIP -> query vector
    query vector . every image vector -> 5000 similarity scores -> top k images

Needs data/image_embeds.npy from `python -m src.embed`.

Usage:
    python -m src.search "a dog on the beach"        # text -> images
    python -m src.search path/to/photo.jpg           # image -> similar images
"""

import sys
import os
import numpy as np
from PIL import Image
from src.data import load_captions
from src.model import load_model, encode_images, encode_texts
from src.embed import load_embeds

top_k = 10


def top_matches(query_vector, embeds, k):
    """
    function: find the k images most similar to the query

    all vectors have length 1, so the dot product is the cosine similarity
    output: indices of the k best images (best first), and their scores
    """
    # (5000, 512) @ (512,) -> (5000,): one score per image
    scores = embeds @ query_vector

    # argsort sorts from small to large, so sort the negative scores to get large first
    order = np.argsort(-scores)
    best_indices = order[:k]
    best_scores = scores[best_indices]
    return best_indices, best_scores


def search_by_text(text, model, processor, embeds, k=top_k):
    text_vectors = encode_texts(model, processor, [text])
    query_vector = text_vectors[0]
    return top_matches(query_vector, embeds, k)


def search_by_image(image_path, model, processor, embeds, k=top_k):
    image = Image.open(image_path).convert("RGB")
    image_vectors = encode_images(model, processor, [image])
    query_vector = image_vectors[0]
    return top_matches(query_vector, embeds, k)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = sys.argv[1]
    else:
        query = "a dog on the beach"

    items = load_captions()
    embeds = load_embeds()
    model, processor = load_model()

    # a path to an existing file means image search, anything else is a text
    if os.path.isfile(query):
        print(f"Images similar to {query}:")
        indices, scores = search_by_image(query, model, processor, embeds)
    else:
        print(f"Images matching '{query}':")
        indices, scores = search_by_text(query, model, processor, embeds)

    for rank in range(len(indices)):
        index = indices[rank]
        score = scores[rank]
        path = items[index]["path"]
        first_caption = items[index]["captions"][0]
        print(f"{rank + 1:2d}. {score:.3f}  {os.path.basename(path)}  ({first_caption})")
