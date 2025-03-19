# Import the required libraries
import os
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

def segment_pedestrians(image_path, output_path):
    # Load the DeepLab model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    model.eval()

    # Define the transformation for preprocessing
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load and preprocess the image
    input_image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) 

    # Perform inference and get segmentation predictions
    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)

    # Extract pedestrian class (COCO class index 15 for "person")
    pedestrian_mask = (output_predictions == 15).byte().cpu().numpy()

    # Create a blank RGB image
    segmented_image = np.zeros((pedestrian_mask.shape[0], pedestrian_mask.shape[1], 3), dtype=np.uint8)
    segmented_image[pedestrian_mask == 1] = [181, 83, 67]

    # Convert the NumPy array to an image
    segmented_image = Image.fromarray(segmented_image)

    # Save the segmented image
    segmented_image_path = os.path.join(output_path, "pedestrian-resultantimage.png")
    segmented_image.save(segmented_image_path)

# Define the path of the image to segment
image_path = '/Users/krishnasomani/Desktop/Projects/StreetScanner/dataset/image.jpeg'
output_path = '/Users/krishnasomani/Desktop/Projects/StreetScanner/result'
segment_pedestrians(image_path, output_path)
