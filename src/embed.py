"""
Compute the CLIP vector of every image and every caption in COCO val2017 (run once).

    5000 images    -> CLIP image encoder -> 5000 x 512 vectors  -> data/image_embeds.npy
    25014 captions -> CLIP text encoder  -> 25014 x 512 vectors -> data/caption_embeds.npy

Row i of image_embeds.npy belongs to items[i] from src.data.load_captions().
The captions are in the order of flatten_captions(items).

Usage:
    python -m src.embed
"""

import os
import time
import numpy as np
from PIL import Image
from src.data import data_path, load_captions
from src.model import load_model, encode_images, encode_texts, device

image_embeds_path = os.path.join(data_path, "image_embeds.npy")
caption_embeds_path = os.path.join(data_path, "caption_embeds.npy")
image_batch_size = 64
text_batch_size = 256


def flatten_captions(items):
    """
    function: put all captions into one list, and remember which image each one belongs to

    output:
        texts: ["A man is in a kitchen making pizzas.", ...]   25014 captions
        targets: [0, 0, 0, 0, 0, 1, 1, ...]                     index of the image of each caption
    """
    texts = []
    targets = []
    for index in range(len(items)):
        captions = items[index]["captions"]
        for caption in captions:
            texts.append(caption)
            targets.append(index)
    return texts, np.array(targets)


def embed_images(model, processor, items):
    """
    function: encode all images batch by batch

    output: numpy array, shape (number of images, 512)
    """
    all_vectors = []
    n_images = len(items)

    for i in range(0, n_images, image_batch_size):
        batch_items = items[i : i + image_batch_size]

        # open the images of this batch
        # convert("RGB"): a few COCO images are grayscale, CLIP needs 3 channels
        images = []
        for item in batch_items:
            path = item["path"]
            image = Image.open(path).convert("RGB")
            images.append(image)

        vectors = encode_images(model, processor, images)
        all_vectors.append(vectors)

    # stack the list of (batch, 512) arrays into one (5000, 512) array
    embeds = np.concatenate(all_vectors)
    return embeds


def embed_texts(model, processor, texts):
    """
    function: encode all texts batch by batch
              the type of texts must be a list

    output: numpy array, shape (number of texts, 512)
    """
    all_vectors = []
    n_texts = len(texts)

    for i in range(0, n_texts, text_batch_size):
        batch_texts = texts[i : i + text_batch_size]
        vectors = encode_texts(model, processor, batch_texts)
        all_vectors.append(vectors)

    embeds = np.concatenate(all_vectors)
    return embeds


def load_image_embeds():
    """
    function: load file which is converted images into vectors
    """
    return np.load(image_embeds_path)


def load_caption_embeds():
    """
    function: load file which is converted texts into vectors
    """
    return np.load(caption_embeds_path)


def center(vectors, mean):
    """
    function: subtract the mean vector, then scale every vector back to length 1
    works for one vector (512,) or many vectors (n, 512), type: numpy
    """
    shifted = vectors - mean
    lengths = np.linalg.norm(shifted, axis=-1, keepdims=True)
    return shifted / lengths


def load_means():
    """
    function: mean of all image vectors and mean of all caption vectors
    """
    image_mean = load_image_embeds().mean(axis=0)   # according row direction
    text_mean = load_caption_embeds().mean(axis=0)
    return image_mean, text_mean


if __name__ == "__main__":
    items = load_captions()
    texts, targets = flatten_captions(items)
    model, processor = load_model()

    print(f"Encoding {len(items)} images on {device} ...")
    start_time = time.time()
    image_embeds = embed_images(model, processor, items)
    image_seconds = time.time() - start_time

    print(f"Encoding {len(texts)} captions on {device} ...")
    start_time = time.time()
    caption_embeds = embed_texts(model, processor, texts)
    text_seconds = time.time() - start_time

    # float32 is enough precision and half the size of float64
    image_embeds = image_embeds.astype(np.float32)
    caption_embeds = caption_embeds.astype(np.float32)
    np.save(image_embeds_path, image_embeds)
    np.save(caption_embeds_path, caption_embeds)

    print(f"images: {image_embeds.shape}, took {image_seconds:.0f}s")
    print(f"captions: {caption_embeds.shape}, took {text_seconds:.0f}s")
    print(f"saved to {image_embeds_path} and {caption_embeds_path}")
