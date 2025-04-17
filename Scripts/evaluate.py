import torch
import clip
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Load fine-tuned CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.load_state_dict(torch.load("../models/clip_finetuned.pth"))  # Load trained model
model.eval()

# Load test dataset (unseen animals)
dataset_path = "../data/test"
dataset = datasets.ImageFolder(root=dataset_path, transform=preprocess)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# Get class names
class_names = dataset.classes  # Unseen animal names
text_inputs = clip.tokenize(class_names).to(device)
text_features = model.encode_text(text_inputs)

correct = 0
total = 0

# Evaluate accuracy
with torch.no_grad():
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        image_features = model.encode_image(images)
        similarity = torch.cosine_similarity(image_features.unsqueeze(1), text_features.unsqueeze(0), dim=-1)

        predictions = similarity.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Zero-Shot Accuracy: {accuracy * 100:.2f}%")
