"""
Export the files the browser demo needs (the "image search" tab of image-classifier).

All 5000 images, made small enough for a web page:
    thumbs/0.jpg ... thumbs/4999.jpg  small copies of the images (longest side 256px)
    embeds.bin                        their centered CLIP vectors, 5000 x 512 float32, row by row
    meta.json                         model name, vector size, image mean, text mean,
                                      and the first caption of each image

In the browser, Transformers.js runs "Xenova/clip-vit-base-patch32" (the same CLIP model)
to turn a text or an uploaded image into a vector, centers it with the mean from meta.json
(text mean for texts, image mean for images), then compares it with embeds.bin.
Thumbnails are only downloaded when they are shown, so the page itself stays light.

Output: ../image-classifier/docs/search/ (the image-classifier repo next to this one)

Usage:
    python -m src.export_demo
"""

import os
import json
from PIL import Image
from src.data import load_captions
from src.search import load_gallery

demo_path = os.path.join(os.path.dirname(__file__), "..", "..", "image-classifier", "docs", "search")
browser_model_name = "Xenova/clip-vit-base-patch32"
thumb_size = 256
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

        if (index + 1) % 1000 == 0:
            print(f"  {index + 1} / {len(items)} thumbnails")


def save_embeds(gallery, out_dir):
    """
    function: save the vectors as raw float32 bytes
    the browser reads them with: new Float32Array(await response.arrayBuffer())
    """
    gallery = gallery.astype("<f4")  # float32, little-endian (the byte order browsers use)
    embeds_path = os.path.join(out_dir, "embeds.bin")
    gallery.tofile(embeds_path)


def save_meta(items, gallery, image_mean, text_mean, out_dir):
    """
    function: save what the browser needs to read embeds.bin, center queries, and show the results
    """
    first_captions = []
    for item in items:
        first_caption = item["captions"][0].strip()
        first_captions.append(first_caption)

    # round the means to 6 decimals to keep meta.json small
    image_mean_list = []
    for value in image_mean:
        image_mean_list.append(round(float(value), 6))
    text_mean_list = []
    for value in text_mean:
        text_mean_list.append(round(float(value), 6))

    n_images = gallery.shape[0]
    dim = gallery.shape[1]
    meta = {
        "model": browser_model_name,
        "count": n_images,
        "dim": dim,
        "image_mean": image_mean_list,
        "text_mean": text_mean_list,
        "captions": first_captions,
    }
    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f)


def export_demo(items, gallery, image_mean, text_mean, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    save_thumbnails(items, out_dir)
    save_embeds(gallery, out_dir)
    save_meta(items, gallery, image_mean, text_mean, out_dir)


def folder_size_mb(folder):
    total_bytes = 0
    for root, dirs, files in os.walk(folder):
        for name in files:
            file_path = os.path.join(root, name)
            total_bytes += os.path.getsize(file_path)
    return total_bytes / 1024 / 1024


if __name__ == "__main__":
    items = load_captions()
    gallery, image_mean, text_mean = load_gallery()

    print(f"Exporting {len(items)} images to {demo_path} ...")
    export_demo(items, gallery, image_mean, text_mean, demo_path)

    size_mb = folder_size_mb(demo_path)
    print(f"done, {size_mb:.1f} MB")
