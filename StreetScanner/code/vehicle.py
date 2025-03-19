# Import the required libraries
import os
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

def segment_vehicles(image_path, output_path):
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

    # Convert the output predictions to a NumPy array
    output_predictions_np = output_predictions.byte().cpu().numpy()

    # Create a mask for vehicles (class IDs 2 and 7 represent car and truck in COCO dataset)
    vehicle_mask = np.isin(output_predictions_np, [2, 7]).astype(np.uint8)

    # Create a color image with vehicles in RGB format (89, 135, 161)
    color_image = np.zeros((vehicle_mask.shape[0], vehicle_mask.shape[1], 3), dtype=np.uint8)
    color_image[:, :, 0] = vehicle_mask * 89  
    color_image[:, :, 1] = vehicle_mask * 135 
    color_image[:, :, 2] = vehicle_mask * 161 

    # Save the segmented image 
    vehicle_image = Image.fromarray(color_image)
    segmented_image_path = os.path.join(output_path, "vehicle-resultant-image.png")
    vehicle_image.save(segmented_image_path)

# Define the path of the image to segment
image_path = '/Users/krishnasomani/Desktop/Projects/StreetScanner/dataset/image.jpeg'
output_path = '/Users/krishnasomani/Desktop/Projects/StreetScanner/result'
segment_vehicles(image_path, output_path)
