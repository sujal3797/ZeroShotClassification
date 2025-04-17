import torch
import clip
from PIL import Image

# Load fine-tuned CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.load_state_dict(torch.load("../models/clip_finetuned.pth"))  # Load trained model
model.eval()

# Load new image for classification
image_path = "../data/input/input_image.jpg"
image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

# Define 1000+ animal classes for zero-shot prediction
animal_classes = ["crocodile", "armadillo", "cheetah", "walrus", "meerkat", "octopus", "dog", "human", "pig"]
text_inputs = clip.tokenize(animal_classes).to(device)

# Compute similarity
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text_inputs)
    similarity = torch.cosine_similarity(image_features, text_features)

# Get best match
best_match = animal_classes[similarity.argmax().item()]
print(f"Predicted Animal: {best_match}")
