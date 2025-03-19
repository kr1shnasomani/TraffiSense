# Import the required libraries
import os
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

def segment_image(image_path, output_path):
    # Load the DeepLab model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    model.eval()

    # Define the transformation for preprocessing
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Define the color palette for visualization
    palette = torch.tensor([3 ** 25 - 1, 3 ** 15 - 1, 3 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    # Load and preprocess the image
    input_image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  

    # Perform inference and get segmentation predictions
    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)

    # Convert the output predictions to a NumPy array
    output_predictions_np = output_predictions.byte().cpu().numpy()

    # Apply the color palette to the segmentation mask
    output_mask = Image.fromarray(output_predictions_np)
    output_mask.putpalette(colors)

    # Save the segmented image
    segmented_image_path = os.path.join(output_path, "both-resultant-image.png")
    output_mask.save(segmented_image_path)

# Define the path of the image to segment
image_path = r"C:\Users\krish\OneDrive\Desktop\Projects\Pedestrian & Vehicle Detection\image.jpeg"
output_path = r"C:\Users\krish\OneDrive\Desktop\Projects\Pedestrian & Vehicle Detection"
segment_image(image_path, output_path)