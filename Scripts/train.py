import torch
import clip
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm  # Import progress bar

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load dataset (90 animals)
dataset_path = "../data/train"
dataset = datasets.ImageFolder(root=dataset_path, transform=preprocess)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=5e-6)

# Get class names
class_names = dataset.classes  # 90 animal names
text_inputs = clip.tokenize(class_names).to(device)
text_features = model.encode_text(text_inputs).detach()  # Detach from computation graph

# Training loop with progress bar
epochs = 5
for epoch in range(epochs):
    # Wrap the dataloader with tqdm for a progress bar
    loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        image_features = model.encode_image(images)

        # Compute cosine similarity loss
        loss = -torch.mean(F.cosine_similarity(image_features, text_features[labels], dim=-1))

        optimizer.zero_grad()
        loss.backward()  # Removed retain_graph=True if not necessary
        optimizer.step()

        # Update progress bar with the current loss
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch + 1}: Loss = {loss.item()}")

# Save fine-tuned model
torch.save(model.state_dict(), "../models/clip_finetuned.pth")
print("Model saved!")
