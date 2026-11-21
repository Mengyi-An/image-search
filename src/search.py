"""
Search the COCO images with a text or with an image.

    query (text or image) -> CLIP -> query vector -> centered
    query vector . every centered image vector -> 5000 similarity scores -> top k images

Centering (see src/embed.py): image vectors and image queries subtract the image mean,
text queries subtract the text mean, then all are normalized again.

Needs data/image_embeds.npy and data/caption_embeds.npy from `python -m src.embed`.

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
from src.embed import load_image_embeds, load_means, center

top_k = 10


def load_gallery():
    """
    function: everything a search needs besides the model

    output: centered image vectors (5000, 512), image mean, text mean
    """
    image_mean, text_mean = load_means()
    embeds = load_image_embeds()
    gallery = center(embeds, image_mean)
    return gallery, image_mean, text_mean


def top_matches(query_vector, gallery, k):
    """
    function: find the k images most similar to the query

    all vectors have length 1, so the dot product is the cosine similarity
    output: indices of the k best images (best first), and their scores
    """
    # (5000, 512) @ (512,) -> (5000,): one score per image
    scores = gallery @ query_vector

    # argsort sorts from small to large, so sort the negative scores to get large first
    order = np.argsort(-scores)
    best_indices = order[:k]
    best_scores = scores[best_indices]
    return best_indices, best_scores


def search_by_text(text, model, processor, gallery, text_mean, k=top_k):
    text_vectors = encode_texts(model, processor, [text])
    query_vector = center(text_vectors[0], text_mean)
    return top_matches(query_vector, gallery, k)


def search_by_image(image_path, model, processor, gallery, image_mean, k=top_k):
    image = Image.open(image_path).convert("RGB")
    image_vectors = encode_images(model, processor, [image])
    query_vector = center(image_vectors[0], image_mean)
    return top_matches(query_vector, gallery, k)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = sys.argv[1]
    else:
        query = "a dog on the beach"

    items = load_captions()
    gallery, image_mean, text_mean = load_gallery()
    model, processor = load_model()

    # a path to an existing file means image search, anything else is a text
    if os.path.isfile(query):
        print(f"Images similar to {query}:")
        indices, scores = search_by_image(query, model, processor, gallery, image_mean)
    else:
        print(f"Images matching '{query}':")
        indices, scores = search_by_text(query, model, processor, gallery, text_mean)

    for rank in range(len(indices)):
        index = indices[rank]
        score = scores[rank]
        path = items[index]["path"]
        first_caption = items[index]["captions"][0]
        print(f"{rank + 1:2d}. {score:.3f}  {os.path.basename(path)}  ({first_caption})")
