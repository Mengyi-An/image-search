"""
Search the COCO images with a text or with an image.

    text  -> CLIP text encoder  -> subtract text mean  -> normalize ┐
    image -> CLIP image encoder -> subtract image mean -> normalize ┴-> query vector

    scores = gallery @ query vector     (gallery: 5000 centered image vectors)
    top k images = the k highest scores

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
    function: centered image vectors

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
    # query_vector就是做 search 时候，input 的 image/text(以文搜图/以图搜图)
    scores = gallery @ query_vector     # cosine similarity

    # argsort sorts from small to large, then [::-1] reverse the list
    # we get indices from argsort instead of list elements
    # 这里取的 indices 是排序前的 indices
    order = np.argsort(scores)[::-1] 
    best_indices = order[:k]
    best_scores = scores[best_indices]
    return best_indices, best_scores


def search_by_text(text, model, processor, gallery, text_mean, k=top_k):
    """
    function: search with text
        把用户输入的那一句话转成向量
        减去文字均值，再归一化（中心化）
        和图库里 5000 个中心化后的图片向量算点积，取分数最高的 k 张
    """

    text_vectors = encode_texts(model, processor, [text])
    query_vector = center(text_vectors[0], text_mean)
    return top_matches(query_vector, gallery, k)


def search_by_image(image_path, model, processor, gallery, image_mean, k=top_k):
    # function: search with image
    
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
