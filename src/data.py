"""
Load COCO val2017: 5000 images, each with ~5 human-written captions.
Downloaded from cocodataset.org on first run (~1GB).

Usage (download data and print dataset statistics):
    python -m src.data
"""

import os
import json
import urllib.request
import zipfile


data_path = os.path.join(os.path.dirname(__file__), "..", "data")
images_url = "http://images.cocodataset.org/zips/val2017.zip"
captions_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# zip files are saved here while downloading
images_zip_path = os.path.join(data_path, "val2017.zip")
captions_zip_path = os.path.join(data_path, "annotations_trainval2017.zip")

# where the files end up after unzipping
images_dir = os.path.join(data_path, "val2017")
captions_file = os.path.join(data_path, "annotations", "captions_val2017.json")


def download_and_unzip(url, zip_path, check_path):
    """
    function: download a zip file, unzip it into data/, then delete the zip
    skip if check_path (the unzipped file or folder) already exists
    """
    if os.path.exists(check_path):
        print(f"Already downloaded: {check_path}")
        return

    os.makedirs(data_path, exist_ok=True)

    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as f:
        f.extractall(data_path)
    os.remove(zip_path)


def load_captions():
    """
    function: read image paths and captions from captions_val2017.json

    output: a list of 5000 dicts, one per image
        [{"path": ".../val2017/000000397133.jpg",
          "captions": ["A man is in a kitchen making pizzas.", ...]},
         ...]
    """
    # coco is a dict, keys: info, licenses, images, annotations
    with open(captions_file) as f:
        coco = json.load(f)

    # coco["images"] is a list of dicts: {"id": 397133, "file_name": "000000397133.jpg", ...}
    # use the image id as key, because each caption only knows its image_id
    images = {}
    for image in coco["images"]:
        image_id = image["id"]
        image_name = image["file_name"]
        path = os.path.join(images_dir, image_name)
        images[image_id] = {"path": path, "captions": []}

    # coco["annotations"] is a list of dicts: {"image_id": 397133, "caption": "...", ...}
    for annotation in coco["annotations"]:
        image_id = annotation["image_id"]
        caption = annotation["caption"]
        entry = images[image_id]
        entry["captions"].append(caption)

    # turn into a list, so each image has a fixed index 0, 1, 2, ...
    items = list(images.values())
    return items


def describe(items):
    """
    function: print dataset statistics

    output:
        images: 5000
        captions per image: min 5, max 7, total 25014
        words per caption: mean 10.4, max 50
    """
    n_captions = []
    n_words = []
    for item in items:
        captions = item["captions"]
        n_captions.append(len(captions))
        for caption in captions:
            words = caption.split()
            n_words.append(len(words))

    n_images = len(items)
    min_captions = min(n_captions)
    max_captions = max(n_captions)
    total_captions = sum(n_captions)
    mean_words = sum(n_words) / len(n_words)
    max_words = max(n_words)

    print(f"images: {n_images}")
    print(f"captions per image: min {min_captions}, max {max_captions}, total {total_captions}")
    print(f"words per caption: mean {mean_words:.1f}, max {max_words}")

    # show the first image, to check that path and captions match
    first_item = items[0]
    first_path = first_item["path"]
    first_captions = first_item["captions"]
    print(f"\nfirst image: {first_path}")
    for caption in first_captions:
        print(f"  - {caption}")


if __name__ == "__main__":
    download_and_unzip(images_url, images_zip_path, images_dir)
    download_and_unzip(captions_url, captions_zip_path, captions_file)

    items = load_captions()
    describe(items)
