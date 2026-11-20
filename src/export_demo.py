"""
Export the files the browser demo needs (the "image search" tab of image-classifier).

The full 5000 images are too big for a web page, so the demo uses the first 1000:
    thumbs/0.jpg ... thumbs/999.jpg   small copies of the images (longest side 320px)
    embeds.bin                        their CLIP vectors, 1000 x 512 float32, row by row
    meta.json                         model name, vector size, and the first caption of each image

In the browser, Transformers.js runs "Xenova/clip-vit-base-patch32" (the same CLIP model)
to turn a text or an uploaded image into a vector, then compares it with embeds.bin.

Output: ../image-classifier/docs/search/ (the image-classifier repo next to this one)

Usage:
    python -m src.export_demo
"""

import os
import json
import numpy as np
from PIL import Image
from src.data import load_captions
from src.embed import load_embeds

demo_path = os.path.join(os.path.dirname(__file__), "..", "..", "image-classifier", "docs", "search")
browser_model_name = "Xenova/clip-vit-base-patch32"
n_demo_images = 1000
thumb_size = 320
jpeg_quality = 80


def save_thumbnails(items, out_dir):
    """
    function: save a small jpg of each image as thumbs/<index>.jpg
    """
    thumbs_dir = os.path.join(out_dir, "thumbs")
    os.makedirs(thumbs_dir, exist_ok=True)

    for index in range(len(items)):
        path = items[index]["path"]
        image = Image.open(path).convert("RGB")

        # thumbnail() shrinks the image in place and keeps its aspect ratio
        image.thumbnail((thumb_size, thumb_size))

        thumb_path = os.path.join(thumbs_dir, f"{index}.jpg")
        image.save(thumb_path, quality=jpeg_quality)


def save_embeds(embeds, out_dir):
    """
    function: save the vectors as raw float32 bytes
    the browser reads them with: new Float32Array(await response.arrayBuffer())
    """
    embeds = embeds.astype("<f4")  # float32, little-endian (the byte order browsers use)
    embeds_path = os.path.join(out_dir, "embeds.bin")
    embeds.tofile(embeds_path)


def save_meta(items, embeds, out_dir):
    """
    function: save what the browser needs to read embeds.bin and show the results
    """
    first_captions = []
    for item in items:
        first_caption = item["captions"][0].strip()
        first_captions.append(first_caption)

    n_images = embeds.shape[0]
    dim = embeds.shape[1]
    meta = {
        "model": browser_model_name,
        "count": n_images,
        "dim": dim,
        "captions": first_captions,
    }
    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f)


def export_demo(items, embeds, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    save_thumbnails(items, out_dir)
    save_embeds(embeds, out_dir)
    save_meta(items, embeds, out_dir)


def folder_size_mb(folder):
    total_bytes = 0
    for root, dirs, files in os.walk(folder):
        for name in files:
            file_path = os.path.join(root, name)
            total_bytes += os.path.getsize(file_path)
    return total_bytes / 1024 / 1024


if __name__ == "__main__":
    items = load_captions()
    embeds = load_embeds()

    # keep only the first n_demo_images, row i of embeds still belongs to items[i]
    demo_items = items[:n_demo_images]
    demo_embeds = embeds[:n_demo_images]

    print(f"Exporting {len(demo_items)} images to {demo_path} ...")
    export_demo(demo_items, demo_embeds, demo_path)

    size_mb = folder_size_mb(demo_path)
    print(f"done, {size_mb:.1f} MB")
