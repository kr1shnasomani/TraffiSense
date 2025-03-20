# Import the required libraries
import cv2
import os
from matplotlib import pyplot as plt
import numpy as np
import easyocr
import imutils

# Image Preprocessing
base_dir = "/Users/krishnasomani/Desktop/Projects/PlateScan"
img_path = os.path.join(base_dir, 'dataset', 'image1.jpg')
img = cv2.imread(img_path)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
bfilter = cv2.bilateralFilter(gray, 11, 17, 17)

# Edge detection
edged = cv2.Canny(bfilter, 30, 200)

# Find contours and apply mark to separate out actual number plate
keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break

# Masking out number plate
mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [location], 0, 255, -1)
new_image = cv2.bitwise_and(img, img, mask=mask)

# Crop the number plate region
(x, y) = np.where(mask == 255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2+1, y1:y2+1]

# Extract text from images using OCR
reader = easyocr.Reader(['en'])
result = reader.readtext(cropped_image)

# Display the extracted text in the terminal
if result:
    text = result[0][-2]
    print(f'Number Plate: {text}')
else:
    print('No text found')

# Save the cropped image
result_path = os.path.join(base_dir, 'result', 'number-plate-1.jpg')
cv2.imwrite(result_path, cropped_image)