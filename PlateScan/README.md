<h1 align="center">PlateScan</h1>
The project detects and extracts text from vehicle number plates using OpenCV for image processing, EasyOCR for text recognition, and imutils for contour detection. It identifies the number plate region, crops it, extracts the text, and saves the result without displaying the images.

## Execution Guide:
1. Run the following command in the terminal:
   ```
   pip install opencv-python numpy matplotlib easyocr imutils
   ```

2. Enter the path of the base directory (a folder which contains all the files)

3. Put the image of your choice in the `dataset` folder

4. Enter the name of image file name into the code (Optional: Enter the name of the file by which you would like to save the output, if not done it would be saved by the default name - `number-plate-1.jpg`)

5. Now upon running the code it would save the cropped image of the vehicle plate number in the result folder and will output the plate number in the terminal

## Result:

Input:

![image](https://github.com/user-attachments/assets/c81cd6d9-f7a7-4b98-820c-26f6603eafeb)

Output:

![image](https://github.com/user-attachments/assets/5dbda361-7c9b-45b5-933b-40701f8d16ae)

`Number plate: HR26BR9044`

## Overview:
The code is a **License Plate Recognition System** that processes an image to extract and recognize a vehicle’s number plate using **OpenCV** and **OCR (EasyOCR)**. The following if the detailed explaination of the code:

1. **Import Required Libraries**  
   - `cv2` (OpenCV) for image processing  
   - `os` for file handling  
   - `matplotlib.pyplot` for visualization (though not used in the script)  
   - `numpy` for numerical operations  
   - `easyocr` for Optical Character Recognition (OCR)  
   - `imutils` for contour manipulation  

2. **Load and Preprocess the Image**  
   - Reads the input image from the dataset  
   - Converts it to grayscale for better processing  
   - Applies **bilateral filtering** to reduce noise while preserving edges  

3. **Edge Detection**  
   - Uses **Canny Edge Detection** to highlight strong edges, helping in number plate detection  

4. **Find and Identify Contours**  
   - Finds the largest contours in the image  
   - Selects the contour that forms a **quadrilateral** (assuming the number plate is rectangular)  

5. **Masking and Cropping the Number Plate**  
   - Creates a mask to isolate the number plate region  
   - Extracts the relevant section of the image  

6. **OCR for Text Extraction**  
   - Uses **EasyOCR** to recognize and extract text from the cropped number plate image  

7. **Display and Save Results**  
   - Prints the recognized number plate text in the terminal  
   - Saves the cropped number plate image in the `result` folder  
