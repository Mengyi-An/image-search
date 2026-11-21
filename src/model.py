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

Usage:
    python -m src.model
"""

import torch
from transformers import CLIPModel, CLIPProcessor

model_name = "openai/clip-vit-base-patch32"

# if GPU is available, otherwise CPU
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

    # Assume we have a vector[3, 4]
    lengths = vectors.norm(dim=-1, keepdim=True)    # compute length: √(3² + 4²) = √25 = 5
    return vectors / lengths    # Normalize: [3/5, 4/5] = [0.6, 0.8]


def encode_images(model, processor, images):
    """
    function: turn a list of PIL images into normalized vectors
    64 images of different sizes
    → processor       (64, 3, 224, 224)   resize, center crop, normalize pixels
    → .to(device)     move to GPU
    → image encoder   (64, 512)           one vector per image
    → normalize       (64, 512)           scale each vector to length 1
    → .cpu().numpy()  numpy array, used by the later steps


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
    input_ids = inputs["input_ids"].to(device)  # 每个 token 的 id
    attention_mask = inputs["attention_mask"].to(device)    # 1 = 真实的 token，0 = 补上去的

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
