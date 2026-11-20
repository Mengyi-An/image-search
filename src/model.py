"""
Load the pretrained CLIP model and turn images / texts into vectors.

CLIP has two encoders that map images and texts into the same 512-dim space:
    image -> image encoder -> 512-dim vector
    text  -> text encoder  -> 512-dim vector
Every vector is normalized to length 1, so the dot product of two vectors
is their cosine similarity (higher = more similar).

No training: the weights are downloaded from the Hugging Face Hub on first run (~600MB).
The browser demo uses "Xenova/clip-vit-base-patch32", the same model converted to ONNX,
so vectors computed here and in the browser can be compared.

Usage (similarity check on one image):
    python -m src.model
"""

import torch
from transformers import CLIPModel, CLIPProcessor

model_name = "openai/clip-vit-base-patch32"

# use the Apple GPU (mps) on a Mac, the NVIDIA GPU (cuda) on Colab, otherwise the CPU
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def load_model():
    """
    function: load CLIP and its processor
    the processor resizes / normalizes images and tokenizes texts
    """
    model = CLIPModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor


def normalize(vectors):
    """
    function: scale every vector to length 1
    """
    lengths = vectors.norm(dim=-1, keepdim=True)
    return vectors / lengths


def encode_images(model, processor, images):
    """
    function: turn a list of PIL images into normalized vectors

    output: numpy array, shape (number of images, 512)
    """
    inputs = processor(images=images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        outputs = model.get_image_features(pixel_values=pixel_values)

    # pooler_output holds the 512-dim vector of each image
    vectors = normalize(outputs.pooler_output)
    return vectors.cpu().numpy()


def encode_texts(model, processor, texts):
    """
    function: turn a list of strings into normalized vectors

    output: numpy array, shape (number of texts, 512)
    """
    # padding: make all texts in the batch the same length
    # truncation: CLIP reads at most 77 tokens
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)

    vectors = normalize(outputs.pooler_output)
    return vectors.cpu().numpy()


if __name__ == "__main__":
    from PIL import Image
    from src.data import load_captions

    items = load_captions()
    first_path = items[0]["path"]
    image = Image.open(first_path).convert("RGB")

    model, processor = load_model()
    image_vectors = encode_images(model, processor, [image])

    texts = ["a man making pizza in a kitchen", "a dog on the beach"]
    text_vectors = encode_texts(model, processor, texts)

    print(f"device: {device}")
    print(f"image vector shape: {image_vectors.shape}")
    print(f"text vector shape: {text_vectors.shape}")

    # the pizza text should be more similar to the kitchen image than the dog text
    image_vector = image_vectors[0]
    for i in range(len(texts)):
        text = texts[i]
        similarity = image_vector @ text_vectors[i]
        print(f"similarity with '{text}': {similarity:.3f}")
