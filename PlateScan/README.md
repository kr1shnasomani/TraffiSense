<h1 align="center">PlateScan</h1>
The project detects and extracts text from vehicle number plates using OpenCV for image processing, EasyOCR for text recognition, and imutils for contour detection. It identifies the number plate region, crops it, extracts the text, and saves the result without displaying the images.

## Execution Guide:
1. Clone the repoistory:
   ```
   git clone https://github.com/kr1shnasomani/TraffiSense.git
   cd TraffiSense/PlateScan
   ```

2. Download the dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Enter the path of the base directory (a folder which contains all the files)

3. Put the image of your choice in the `dataset` folder

4. Enter the name of image file name into the code (Optional: Enter the name of the file by which you would like to save the output, if not done it would be saved by the default name - `number-plate-1.jpg`)

5. Now upon running the code it would save the cropped image of the vehicle plate number in the result folder and will output the plate number in the terminal

## Result:

Input:

![image](https://github.com/user-attachments/assets/c81cd6d9-f7a7-4b98-820c-26f6603eafeb)

Output:

![image](https://github.com/user-attachments/assets/5dbda361-7c9b-45b5-933b-40701f8d16ae)

`Number plate: HR26BR9044`
