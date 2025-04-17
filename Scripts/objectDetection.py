import torch
import clip
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from torchvision import transforms
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.load_state_dict(torch.load("../models/clip_finetuned.pth"))
clip_model.eval()

# Load Faster R-CNN with new 'weights' parameter
proposal_model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(device)
proposal_model.eval()

# Zero-shot labels
labels = ["dog", "cat", "elephant", "giraffe", "crocodile", "cheetah", "zebra", "lion", "deer", "cow", "donkey", "pig", "human"]
text_inputs = clip.tokenize(labels).to(device)

# Precompute and normalize text embeddings
with torch.no_grad():
    text_features = clip_model.encode_text(text_inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# Detection transform
detection_transform = transforms.Compose([
    transforms.ToTensor()
])


def zero_shot_object_detection(image_path, score_threshold=0.5):
    """
    1) Generate region proposals using Faster R-CNN.
    2) For each proposal, crop and classify with CLIP (zero-shot).
    3) Return bounding boxes, predicted labels, and annotated PIL image.
    """
    # A) Load and preprocess the image
    original_image = Image.open(image_path).convert("RGB")
    input_tensor = detection_transform(original_image).unsqueeze(0).to(device)

    # B) Get region proposals
    with torch.no_grad():
        outputs = proposal_model(input_tensor)

    # Extract boxes & scores
    boxes = outputs[0]["boxes"]
    scores = outputs[0]["scores"]

    # Filter low-confidence proposals
    keep = scores > score_threshold
    boxes = boxes[keep]
    scores = scores[keep]

    # Convert to CPU
    boxes = boxes.cpu().numpy().astype(int)
    scores = scores.cpu().numpy()

    # Draw bounding boxes on a copy of the original image
    annotated_image = original_image.copy()
    draw = ImageDraw.Draw(annotated_image)

    predicted_labels = []

    # C) Classify each bounding box with CLIP
    for box in boxes:
        x1, y1, x2, y2 = box
        region = original_image.crop((x1, y1, x2, y2))

        # Preprocess region for CLIP
        region_tensor = clip_preprocess(region).unsqueeze(0).to(device)
        with torch.no_grad():
            region_features = clip_model.encode_image(region_tensor)
            region_features = region_features / region_features.norm(dim=-1, keepdim=True)

            # Compute similarity to text embeddings
            similarity = (region_features @ text_features.T).squeeze(0)
            best_idx = similarity.argmax().item()
            best_label = labels[best_idx]
            predicted_labels.append(best_label)

        # Draw the bounding box and label on the image
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        # Optional: Draw label text
        draw.text((x1, y1), best_label, fill="white")

    return boxes, predicted_labels, annotated_image


if __name__ == "__main__":
    test_image_path = "../data/input/input_image.jpg"  # Replace with your image path
    boxes, predicted_labels, annotated_img = zero_shot_object_detection(
        test_image_path, score_threshold=0.7
    )

    for box, label in zip(boxes, predicted_labels):
        print(f"Box {box}, Label: {label}")

    # Display the annotated image
    annotated_img.show()
    # Or save it to a file
    annotated_img.save("../data/output/annotated_output2.jpg")
