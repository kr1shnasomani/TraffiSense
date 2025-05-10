<h1 align="center">StreetScanner</h1>
The system leverages DeepLabv3 ResNet-50 to detect and segment pedestrians and vehicles, with color-coded masks for easy visualization. This solution is designed for applications in traffic monitoring, pedestrian safety and smart city solutions. Built using PyTorch, Semantic Segmentation and Computer Vision techniques.

## Execution Guide:
1. Clone the repoistory:
   ```
   git clone https://github.com/kr1shnasomani/TraffiSense.git
   cd TraffiSense/StreetScanner
   ```

2. Download the dependencies:
   ```
   pip install -r requirements.txt
   ```

3. On running the code it will save the results in the file - `both-resultant-image.png`, `pedestrian-resultant-image.png` and `vehicle-resultant-image.png`

## Model Prediction:

   Input Image:

   ![image](https://github.com/user-attachments/assets/23ae6bf3-7f30-4389-95f5-e3772be3d3f7)

   Output Image:

   a. `both-resultant-image.png`

   ![image](https://github.com/user-attachments/assets/4e1fa906-e2f4-4018-8e81-bf3415d715c4)

   b. `pedestrian-resultant-image.png`

   ![image](https://github.com/user-attachments/assets/ff1745c7-b5b6-4021-bb9c-3d0b5c8e3577)

   c. `vehicle-resultant-image.png`

   ![image](https://github.com/user-attachments/assets/1c0b6d28-ce89-4d31-b83d-1ceaa5f1c2c4)

- **`vehicle.py`**: For applications requiring vehicle detection (e.g., traffic monitoring, autonomous vehicle systems).

These scripts demonstrate a modular approach, where specific objects of interest (pedestrians, vehicles) can be segmented independently or collectively, depending on the application.
