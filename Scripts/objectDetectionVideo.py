import cv2
import torch
import clip
from PIL import Image, ImageDraw
import numpy as np
from torchvision import transforms
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights
)

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.load_state_dict(torch.load("../models/clip_finetuned.pth"))
clip_model.eval()

# Load Faster R-CNN for region proposals with updated weights API
proposal_model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(device)
proposal_model.eval()

# Define your zero-shot class labels
labels = ["dog", "cat", "elephant", "giraffe", "crocodile", "cheetah", "zebra", "lion", "dolphin", "bird", "rabbit", "walrus"]
text_inputs = clip.tokenize(labels).to(device)
with torch.no_grad():
    text_features = clip_model.encode_text(text_inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# Detection transform (Faster R-CNN expects [0,1] tensors)
detection_transform = transforms.Compose([
    transforms.ToTensor()
])


def process_frame(pil_image, score_threshold=0.7):
    """
    Process a single PIL image:
      1. Generate region proposals.
      2. For each proposal, crop and classify using CLIP.
      3. Draw bounding boxes and labels.
    Returns the annotated PIL image.
    """
    # Convert image to tensor and get proposals
    input_tensor = detection_transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = proposal_model(input_tensor)

    boxes = outputs[0]["boxes"]
    scores = outputs[0]["scores"]
    keep = scores > score_threshold
    boxes = boxes[keep].cpu().numpy().astype(int)

    # Annotate image
    annotated_image = pil_image.copy()
    draw = ImageDraw.Draw(annotated_image)

    # Process each detected region
    for box in boxes:
        x1, y1, x2, y2 = box
        region = pil_image.crop((x1, y1, x2, y2))
        region_tensor = clip_preprocess(region).unsqueeze(0).to(device)
        with torch.no_grad():
            region_features = clip_model.encode_image(region_tensor)
            region_features = region_features / region_features.norm(dim=-1, keepdim=True)
            similarity = (region_features @ text_features.T).squeeze(0)
            best_idx = similarity.argmax().item()
            best_label = labels[best_idx]

        # Draw bounding box and label
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1), best_label, fill="white")

    return annotated_image


def process_video(video_path, output_path=None):
    """
    Process a video file frame by frame, applying zero-shot object detection
    and drawing annotated bounding boxes.
    Optionally, save the annotated video.
    """
    cap = cv2.VideoCapture(video_path)
    writer = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame (BGR) to a PIL Image (RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # Process the frame with our detection pipeline
        annotated_pil = process_frame(pil_image, score_threshold=0.7)

        # Convert annotated image back to OpenCV (BGR) format
        annotated_frame = cv2.cvtColor(np.array(annotated_pil), cv2.COLOR_RGB2BGR)

        # Initialize VideoWriter if needed
        if output_path and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            height, width, _ = annotated_frame.shape
            writer = cv2.VideoWriter(output_path, fourcc, 20, (width, height))

        if writer:
            writer.write(annotated_frame)

        # Display the annotated frame
        cv2.imshow("Annotated Video", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Provide the path to your input video (or use 0 for webcam)
    input_video = "../data/input/input_video3.mp4"  # Change this to your video file path
    output_video = "../data/output/annotated_video3.mp4"  # Output video path (optional)

    process_video(input_video, output_path=output_video)
