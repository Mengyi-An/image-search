"""
Compute the CLIP vector of every image in COCO val2017 (run once).

This project has no training step; this file takes its place: it is the slow
step that runs once and saves the result, so searching later is fast.

    5000 images -> CLIP image encoder -> 5000 x 512 vectors -> data/image_embeds.npy

Row i of image_embeds.npy belongs to items[i] from src.data.load_captions().

Usage:
    python -m src.embed
"""

import os
import time
import numpy as np
from PIL import Image
from src.data import data_path, load_captions
from src.model import load_model, encode_images, device

embeds_path = os.path.join(data_path, "image_embeds.npy")
batch_size = 64


def embed_images(model, processor, items):
    """
    function: encode all images batch by batch

    output: numpy array, shape (number of images, 512)
    """
    all_vectors = []
    n_images = len(items)

    for start in range(0, n_images, batch_size):
        end = min(start + batch_size, n_images)

        # open the images of this batch
        # convert("RGB"): a few COCO images are grayscale, CLIP needs 3 channels
        images = []
        for i in range(start, end):
            path = items[i]["path"]
            image = Image.open(path).convert("RGB")
            images.append(image)

        vectors = encode_images(model, processor, images)
        all_vectors.append(vectors)

        if end % 640 == 0 or end == n_images:
            print(f"  {end} / {n_images} images")

    # stack the list of (batch, 512) arrays into one (5000, 512) array
    embeds = np.concatenate(all_vectors)
    return embeds


def load_embeds():
    """
    function: read the saved image vectors
    """
    return np.load(embeds_path)


if __name__ == "__main__":
    items = load_captions()
    model, processor = load_model()

    print(f"Encoding {len(items)} images on {device} ...")
    start_time = time.time()
    embeds = embed_images(model, processor, items)
    seconds = time.time() - start_time

    # float32 is enough precision and half the size of float64
    embeds = embeds.astype(np.float32)
    np.save(embeds_path, embeds)

    print(f"shape: {embeds.shape}, took {seconds:.0f}s")
    print(f"saved to {embeds_path}")
