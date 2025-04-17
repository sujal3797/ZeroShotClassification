import os
from PIL import Image
from torchvision import transforms

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to CLIP input size
    transforms.ToTensor()
])

# Convert tensor back to image
to_pil = transforms.ToPILImage()

# Process dataset images
def preprocess_images(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(root, file)
                try:
                    img = Image.open(img_path).convert("RGB")  # Open image
                    img = transform(img)  # Apply transformations
                    img = to_pil(img)  # Convert tensor back to PIL image
                    img.save(img_path)  # Overwrite with resized version
                    print(f"Processed: {img_path}")
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

# Apply preprocessing to train & test sets
preprocess_images("../data/train")
preprocess_images("../data/test")

print("Preprocessing complete! ✅")
