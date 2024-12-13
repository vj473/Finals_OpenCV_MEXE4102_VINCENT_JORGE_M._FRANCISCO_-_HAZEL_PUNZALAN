# Finals_OpenCV_MEXE4102_VINCENT_JORGE_M._FRANCISCO_-_JHEN_HAZEL_PUNZALAN

# INTRODUCTION
* Edge detection is a fundamental process in computer vision and image processing, aimed at identifying significant transitions or boundaries within an image. These boundaries often correspond to changes in intensity, color, texture, or depth, representing the edges of objects or features in a scene. The ability to detect edges is critical for various applications, including object recognition, image segmentation, and scene understanding.

* In essence, edge detection transforms raw image data into a structured representation by highlighting areas of interest. This simplification reduces the computational complexity of subsequent processing tasks while preserving essential geometric information.
# ABSTRACT
  Edge detection is a critical technique in image processing, used to identify the boundaries of objects within an image. In this project, we apply two common edge detection methods—Sobel and Canny—to visualize the edges of a wolf in a given image. The Sobel operator computes the gradient of pixel intensities to highlight areas with rapid intensity changes, while the Canny edge detector performs a multi-step process that refines edge detection by reducing noise and ensuring more precise boundaries. This study aims to compare the effectiveness of both methods in extracting the wolf’s shape and features, ultimately demonstrating how edge detection can enhance the clarity of an object’s outline. The results of this visualization can be useful in various applications, including wildlife monitoring, object recognition, and automated image analysis.
# PROJECT METHODS
1. IMAGE ACQUISITION:
   - A high quality image of a wolf chosen for edge detection.
2. PREPROCESSING:
   - Noise Reduction - apply Gaussian blur or other smoothing techniques to reduce noise in the image, which could interfere with edge detection.
3. EDGE DETECTION METHODS:
  - Sobel Edge Detection - the Sobel operator uses two filters horizontal and vertical to compute the gradients of pixel intensities in both directions. By applying these filters to the image, the gradients along the horizontal and vertical axes are obtained.
  - Canny Edge Detection - starts with Gaussian blur to reduce noise, followed by gradient calculation to identify intensity changes. Non-maximum suppression thins the edges by keeping only prominent ones. Double thresholding classifies edges as strong, weak, or non-edges, and edge tracing by hysteresis ensures weak edges are retained only if connected to strong edges, finalizing the process.
4. COMPARISON OF RESULTS:
  - Visualize the edges detected using Sobel and Canny side by side.
  - Analyze the effectiveness of each method in terms of; Edge clarity and sharpness, handling of noise, accuracy of boundary detection arount the wolf.
5. VISUALIZATION:
  - Use tools like Matplotlib or OpenCV to display the original image alongside the edge-detected images.
  - Optionally, overlay the detected edges on the original image to better highlight the wolf's shape.
6. EVALUATION:
  - Assess the performance of both edge detection methods qualitatively.
# CONCLUSION
* In conclusion, this project successfully explored the principles and implementation of visual edge detection, a critical technique in image processing and computer vision. Using [insert techniques/tools, e.g., Sobel, Canny, or advanced neural networks], the study demonstrated how edge detection enhances the ability to identify object boundaries, shapes, and structural information in images.

* The results highlighted the effectiveness of edge detection algorithms in isolating key features while maintaining computational efficiency. However, challenges such as handling noisy images and optimizing performance in complex scenarios were noted. These limitations provide avenues for future work, including the integration of advanced denoising techniques and adaptive algorithms to improve robustness.

* Overall, this project underscores the importance of edge detection in diverse applications, from medical imaging to autonomous navigation, and paves the way for further exploration in this vital area of computer vision.
  
# ADDITIONAL MATERIALS

 # 16 Basic OpenCV projects

 ## Part 1: OpenCV Basics
 ```!git clone https://github.com/vj473/Finals_OpenCV_MEXE4102_VINCENT_JORGE_M._FRANCISCO_-_HAZEL_PUNZALAN.git
%cd Finals_OpenCV_MEXE4102_VINCENT_JORGE_M._FRANCISCO
from IPython.display import clear_output
clear_output()
```
#1 Converting Images to Grayscale
* Use the color space conversion code to convert RGB images to grayscale for basic image preprocessing.

```import cv2
from google.colab.patches import cv2_imshow

#colorful image - 3 channels
image = cv2.imread("/content/Finals_OpenCV_MEXE4102_VINCENT_JORGE_M._FRANCISCO_-_HAZEL_PUNZALAN/OPENCV IMAGES/asong lubo.jpg")
print(image.shape)

#grayscale image
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2_imshow(gray)
```
![Screenshot 2024-12-13 183414](https://github.com/user-attachments/assets/dcd1b3b6-57c6-4502-abc3-93a4541dfadc)

#2 Visualizing Edge Detection
* Apply the edge detection code to detect and visualize edges in a collection of object images.

```import cv2
from google.colab.patches import cv2_imshow
import numpy as np

image = cv2.imread("/content/Finals_OpenCV_MEXE4102_VINCENT_JORGE_M._FRANCISCO_-_HAZEL_PUNZALAN/OPENCV IMAGES/tanggol.jpg")
# cv2_imshow(image)

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
canny_image = cv2.Canny(gray,150, 200)
cv2_imshow(canny_image)
```
![Screenshot 2024-12-13 183424](https://github.com/user-attachments/assets/d45d35d2-836b-4833-956c-3505a408847e)

#3 Demonstrating Morphological Erosion
* Use the erosion code to show how an image's features shrink under different kernel sizes.

```import cv2
from google.colab.patches import cv2_imshow
import numpy as np

image = cv2.imread("/content/Finals_OpenCV_MEXE4102_VINCENT_JORGE_M._FRANCISCO_-_HAZEL_PUNZALAN/OPENCV IMAGES/tanggol.jpg")
# cv2_imshow(image)

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
canny_image = cv2.Canny(gray,150, 200)
cv2_imshow(canny_image)

# EROSION

kernel = np.ones((1,1),np.uint8)
eroded_image = cv2.erode(canny_image,kernel,iterations=1)
cv2_imshow(eroded_image)
```
![Screenshot 2024-12-13 184214](https://github.com/user-attachments/assets/8eba9647-3478-45ce-85a9-50a94fb054f3)


#4 Demonstrating Morphological Dilation
* Apply the dilation code to illustrate how small gaps in features are filled.

```import cv2
from google.colab.patches import cv2_imshow
import numpy as np

image = cv2.imread("/content/Finals_OpenCV_MEXE4102_VINCENT_JORGE_M._FRANCISCO_-_HAZEL_PUNZALAN/OPENCV IMAGES/tanggol.jpg")
# cv2_imshow(image)

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
canny_image = cv2.Canny(gray,150, 200)
cv2_imshow(canny_image)

# DILATED

kernel = np.ones((5,5),np.uint8)
dilated_image = cv2.dilate(canny_image,kernel,iterations=1)
cv2_imshow(dilated_image)
```
![Screenshot 2024-12-13 183453](https://github.com/user-attachments/assets/2ce6dc1d-a033-4d0d-b488-19975ba1bbab)

#5 Reducing Noise in Photos
* Use the denoising code to clean noisy images and compare the before-and-after effects.

```import cv2
from google.colab.patches import cv2_imshow
import numpy as np

image = cv2.imread("/content/Finals_OpenCV_MEXE4102_VINCENT_JORGE_M._FRANCISCO_-_HAZEL_PUNZALAN/OPENCV IMAGES/HEV PAKYAW.jpg")
# cv2_imshow(image)
dst = cv2.fastNlMeansDenoisingColored(image, None, 20, 20, 7, 15)

display = np.hstack((image, dst))
cv2_imshow(display)
```
![Screenshot 2024-12-13 183508](https://github.com/user-attachments/assets/e089923f-e855-452f-8425-f2ba2e1478ee)

#6 Drawing Geometric Shapes on Images
* Apply the shape-drawing code to overlay circles, rectangles, and lines on sample photos.

```import cv2
import numpy as np
from google.colab.patches import cv2_imshow

img = np.zeros((512, 512, 3), np.uint8)
#uint8: 0 to 255

# Drawing Function
# Draw a Circle
cv2.circle(img, (100,100), 50, (0,255,0),5)
# Draw a Rectangle
cv2.rectangle(img,(200,200),(400,500),(0,0,255),5)
#Draw a Line
cv2.line(img, (160,160),(359,29),(255,0,0),3)
#Write a Text
cv2.putText(img,"OpenCV",(160,160),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,255),2)
# Displaying the Image
cv2_imshow(img)
```
![Screenshot 2024-12-13 183521](https://github.com/user-attachments/assets/fd9dc12f-55e7-4a2b-9cac-386053905b43)

#7 Adding Text to Images
* Use the text overlay code to label images with captions, annotations, or titles.

```import cv2
import numpy as np
from google.colab.patches import cv2_imshow

img = np.zeros((512, 512, 3), np.uint8)
#write a text
cv2.putText(img,"ANUEE?",(80,250),cv2.FONT_HERSHEY_COMPLEX,2,(0,100,300),3)
cv2_imshow(img)
```
![Screenshot 2024-12-13 183530](https://github.com/user-attachments/assets/c05fa081-a009-4fac-9923-48e2a7a49e57)


#8 Isolating Objects by Color
* Apply the HSV thresholding code to extract and display objects of specific colors from an image.
```import cv2
import numpy as np
from google.colab.patches import cv2_imshow
#BGR Image . It is represented in Blue, Green and Red Channels...
image = cv2.imread("/content/Finals_OpenCV_MEXE4102_VINCENT_JORGE_M._FRANCISCO_-_HAZEL_PUNZALAN/OPENCV IMAGES/shapes.png")
hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

# Blue Color Triangle
lower_hue = np.array([65,0,0])
upper_hue = np.array([110, 255,255])

# Red Color
lower_hue = np.array([0,0,0])
upper_hue = np.array([20,255, 255])

# Green Color
lower_hue = np.array([46,0,0])
upper_hue = np.array([91,255,255])

# Yellow Color
lower_hue = np.array([21,0,0])
upper_hue = np.array([45,255,255])

mask = cv2.inRange(hsv,lower_hue,upper_hue)
cv2_imshow(mask)
result = cv2.bitwise_and(image, image, mask = mask)
cv2_imshow(result)
cv2_imshow(image)
```
![Screenshot 2024-12-13 183551](https://github.com/user-attachments/assets/ee9353a4-87f9-4886-b98f-02b12215c50d)
![Screenshot 2024-12-13 183542](https://github.com/user-attachments/assets/9376ba1b-2b7a-4aa0-a987-4cc4d7537381)

#9 Detecting Faces in Group Photos
* Use the face detection code to identify and highlight faces in group pictures.

```import cv2
from google.colab.patches import cv2_imshow

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
img = cv2.imread("OPENCV IMAGES/gang.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# Draw rectangles around faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)

cv2_imshow(img)
```
![Screenshot 2024-12-13 183607](https://github.com/user-attachments/assets/6628e00d-3d76-4d9d-bbd1-9497d57396d6)


#10 Outlining Shapes with Contours
* Apply the contour detection code to outline and highlight shapes in simple object images.

```import cv2
import numpy as np
from google.colab.patches import cv2_imshow

img = cv2.imread("/content/Finals_OpenCV_MEXE4102_VINCENT_JORGE_M._FRANCISCO_-_HAZEL_PUNZALAN/OPENCV IMAGES/hugis.png")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray,50,255,1)
contours,h = cv2.findContours(thresh,1,2)
# cv2_imshow(thresh)
for cnt in contours:
  approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
  n = len(approx)
  if n==6:
    # this is a hexagon
    print("We have a hexagon here")
    cv2.drawContours(img,[cnt],0,255,10)
  elif n==3:
    # this is a triangle
    print("We found a triangle")
    cv2.drawContours(img,[cnt],0,(0,255,0),3)
  elif n>9:
    # this is a circle
    print("We found a circle")
    cv2.drawContours(img,[cnt],0,(0,255,255),3)
  elif n==4:
    # this is a Square
    print("We found a square")
    cv2.drawContours(img,[cnt],0,(255,255,0),3)
cv2_imshow(img)
```
![Screenshot 2024-12-13 183619](https://github.com/user-attachments/assets/8f87aa30-982a-48dd-8c67-18d73b1c05fe)

#11 Tracking a Ball in a Video
* Use the HSV-based object detection code to track a colored ball in a recorded video.
```import cv2
import numpy as np
from google.colab.patches import cv2_imshow
import time  # For adding delays between frames

# Initialize the video and variables
ball = []
cap = cv2.VideoCapture("/content/Finals_OpenCV_MEXE4102_VINCENT_JORGE_M._FRANCISCO_-_HAZEL_PUNZALAN/OPENCV IMAGES/Video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to HSV and create a mask for the ball color
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_hue = np.array([21, 0, 0])  # Adjust for your ball's color
    upper_hue = np.array([45, 255, 255])
    mask = cv2.inRange(hsv, lower_hue, upper_hue)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    center = None

    if len(contours) > 0:
        # Get the largest contour
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)

        try:
            # Calculate the center of the ball
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            # Draw a circle at the center
            cv2.circle(frame, center, 10, (255, 0, 0), -1)
            ball.append(center)
        except ZeroDivisionError:
            pass

        # Draw the tracking path
        if len(ball) > 2:
            for i in range(1, len(ball)):
                cv2.line(frame, ball[i - 1], ball[i], (0, 0, 255), 5)

    # Display the frame in the notebook
    cv2_imshow(frame)

    # Add a small delay to simulate real-time playback
    time.sleep(0.05)

cap.release()
```
![Screenshot 2024-12-13 184558](https://github.com/user-attachments/assets/fd9e8d06-afdd-49ec-937a-b66eeb7a4720)



#12 Highlighting Detected Faces
* Apply the Haar cascade face detection code to identify and highlight multiple faces in family or crowd photos.
```!git clone https://github.com/vj473/Finals_OpenCV_MEXE4102_VINCENT_JORGE_M._FRANCISCO_-_HAZEL_PUNZALAN.git
!pip install face_recognition
%cd Finals_OpenCV_MEXE4102_VINCENT_JORGE_M._FRANCISCO_-_HAZEL_PUNZALAN
```
```import face_recognition
import numpy as np
from google.colab.patches import cv2_imshow
import cv2

# Creating the encoding profiles
face_1 = face_recognition.load_image_file("OPENCV IMAGES/angkol john loyd.jpg")
face_1_encoding = face_recognition.face_encodings(face_1)[0]

face_2 = face_recognition.load_image_file("OPENCV IMAGES/pareng neil.jpg")
face_2_encoding = face_recognition.face_encodings(face_2)[0]

face_3 = face_recognition.load_image_file("OPENCV IMAGES/kuya gelo.jpg")
face_3_encoding = face_recognition.face_encodings(face_3)[0]

known_face_encodings = [
                        face_1_encoding,
                        face_2_encoding,
                        face_3_encoding
]

known_face_names = [
                    "angkol john loyd",
                    "pareng neil",
                    "kuya gelo"
]

```
```file_name = "OPENCV IMAGES/pareng neil 2.jpg"
unknown_image = face_recognition.load_image_file(file_name)
unknown_image_to_draw = cv2.imread(file_name)

face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

for (top,right, bottom, left), face_encoding in zip(face_locations, face_encodings):
  matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

  name = "Unknown"

  face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
  best_match_index = np.argmin(face_distances)
  if matches[best_match_index]:
    name = known_face_names[best_match_index]
  cv2.rectangle(unknown_image_to_draw, (left, top), (right, bottom),(0,255,0),3)
  cv2.putText(unknown_image_to_draw,name, (left, top-20), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2, cv2.LINE_AA)

cv2_imshow(unknown_image_to_draw)
```
![Screenshot 2024-12-13 183736](https://github.com/user-attachments/assets/8595a54d-20d0-4d01-b039-9473cf4935b2)

#13 Extracting Contours for Shape Analysis
Use contour detection to analyze and outline geometric shapes in hand-drawn images.
```import cv2
from google.colab.patches import cv2_imshow
import numpy as np

# Read the input image
image = cv2.imread("/content/Finals_OpenCV_MEXE4102_VINCENT_JORGE_M._FRANCISCO_-_HAZEL_PUNZALAN/OPENCV IMAGES/geo-hand.jpg")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply edge detection
edges = cv2.Canny(gray, 50, 150)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a copy of the original image to draw contours
contour_image = image.copy()

# Draw the contours on the image
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

# Analyze each contour and approximate the shape
for contour in contours:
    # Approximate the contour
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Find the bounding rectangle to label the shape
    x, y, w, h = cv2.boundingRect(approx)

    # Determine the shape based on the number of vertices
    if len(approx) == 3:
        shape = "Triangle"
    elif len(approx) == 4:
        # Check if the shape is square or rectangle
        aspect_ratio = float(w) / h
        shape = "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
    elif len(approx) > 4:
        shape = "Circle"
    else:
        shape = "Polygon"

    # Put the name of the shape on the image
    cv2.putText(contour_image, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Stack the original, edge-detected, and contour images for display
stacked_result = np.hstack((cv2.resize(image, (300, 300)),
                            cv2.resize(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), (300, 300)),
                            cv2.resize(contour_image, (300, 300))))

# Display the results
cv2_imshow(stacked_result)
```
![Screenshot 2024-12-13 183749](https://github.com/user-attachments/assets/31340b9f-e4a3-44cc-8c0b-ede63a041952)

#14 Applying Image Blurring Techniques
* Demonstrate various image blurring methods (Gaussian blur, median blur) to soften details in an image.
```import cv2
from google.colab.patches import cv2_imshow
import numpy as np

image = cv2.imread("/content/Finals_OpenCV_MEXE4102_VINCENT_JORGE_M._FRANCISCO_-_HAZEL_PUNZALAN/OPENCV IMAGES/tiger commando.jpg")
Gaussian = cv2.GaussianBlur(image,(7,7),0)
Median = cv2.medianBlur(image,5)

display = np.hstack((Gaussian,Median))
cv2_imshow(display)
```
![Screenshot 2024-12-13 183804](https://github.com/user-attachments/assets/45b1881b-2a10-4a26-836b-26062a708619)

#15 Segmenting Images Based on Contours
* Use contour detection to separate different sections of an image, like dividing a painting into its distinct elements.
```import cv2
from google.colab.patches import cv2_imshow
import numpy as np

# Read the input image
image = cv2.imread("/content/Finals_OpenCV_MEXE4102_VINCENT_JORGE_M._FRANCISCO_-_HAZEL_PUNZALAN/OPENCV IMAGES/geo-hand.jpg")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply binary thresholding
_, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# Find contours
contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a blank mask for segmentation
segmented_image = np.zeros_like(image)

# Loop through each contour to extract and display segmented areas
for i, contour in enumerate(contours):
    # Create a mask for the current contour
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [contour], -1, 255, -1)

    # Extract the segment by masking the original image
    segmented_part = cv2.bitwise_and(image, image, mask=mask)

    # Add the segment to the segmented image
    segmented_image = cv2.add(segmented_image, segmented_part)

    # Optionally draw bounding boxes for visualization
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)


# Display results
cv2_imshow(image)  # Original image with bounding boxes
cv2_imshow(segmented_image)  # Segmented image
```
![Screenshot 2024-12-13 183820](https://github.com/user-attachments/assets/3698258d-02c8-4937-afaf-0c4b7ad70f87)

#16 Combining Erosion and Dilation for Feature Refinement
* Apply erosion followed by dilation on an image to refine and smooth out small features.
```import cv2
from google.colab.patches import cv2_imshow
import numpy as np

image = cv2.imread("/content/Finals_OpenCV_MEXE4102_VINCENT_JORGE_M._FRANCISCO_-_HAZEL_PUNZALAN/OPENCV IMAGES/car.jpg")
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
canny_image = cv2.Canny(gray,150, 200)
kernel = np.ones((1,1), np.uint8)
erode_image = cv2.erode(canny_image,kernel, iterations=1)
kernel1 = np.ones((3,3), np.uint8)
dilate_image = cv2.dilate(erode_image, kernel1, iterations=1)

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(canny_image, 'Canny Image', (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(erode_image, 'Eroded', (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(dilate_image, 'Feature Refined', (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

display = np.hstack((canny_image,erode_image,dilate_image))
cv2_imshow(display)
```
![Screenshot 2024-12-13 183833](https://github.com/user-attachments/assets/eb300e84-52a7-4435-831c-fd0edaa2fa3c)


## Part 2

```
# Clone the GitHub repository
!git clone https://github.com/vj473/Finals_OpenCV_MEXE4102_VINCENT_JORGE_M._FRANCISCO_-_HAZEL_PUNZALAN.git

# Change directory into the cloned repository (ensure the folder name matches the cloned repo)
%cd Finals_OpenCV_MEXE4102_VINCENT_JORGE_M._FRANCISCO_-_HAZEL_PUNZALAN/

# Clear the output (useful for Jupyter notebooks)
from IPython.display import clear_output
clear_output()

# Print a success message
print("Repository cloned and directory changed successfully!")
```

```
import os
import cv2
import matplotlib.pyplot as plt

# Print the current working directory to ensure the correct folder
print("Current working directory:", os.getcwd())

# Define relative paths to the images (since they are in the same folder as the script)
image1_path = "asong lubo.jpg"
image2_path = "wolf.jpg"

# Check if the images exist at the given paths
if os.path.exists(image1_path):
    print(f"Image 1 exists at: {image1_path}")
    # Load and display the first image
    image1 = cv2.imread(image1_path)
    if image1 is not None:
        print("First image loaded successfully.")
        plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
        plt.title("First Image")
        plt.axis('off')
        plt.show()
    else:
        print(f"Error: Unable to load the first image '{image1_path}'.")
else:
    print(f"Error: '{image1_path}' does not exist.")

if os.path.exists(image2_path):
    print(f"Image 2 exists at: {image2_path}")
    # Load and display the second image
    image2 = cv2.imread(image2_path)
    if image2 is not None:
        print("Second image loaded successfully.")
        plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
        plt.title("Second Image")
        plt.axis('off')
        plt.show()
    else:
        print(f"Error: Unable to load the second image '{image2_path}'.")
else:
    print(f"Error: '{image2_path}' does not exist.")
```
![image](https://github.com/user-attachments/assets/b5aeedfb-654f-4d70-888b-e87da23c164b)
![image](https://github.com/user-attachments/assets/32383776-52d6-49f4-b01b-9753dedbe7c8)


```
# Convert to graycsale
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
```

```
sobelx = cv2.Sobel(src=image, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)

sobely = cv2.Sobel(src=image, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)

sobelxy = cv2.Sobel(src=image, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)

plt.figure(figsize=(18,19))
plt.subplot(221)
plt.imshow(image, cmap='gray')
plt.title('Original')
plt.axis("off")

plt.subplot(222)
plt.imshow(sobelxy, cmap='gray')
plt.title('Sobel X Y')
plt.axis("off")

plt.subplot(223)
plt.imshow(sobelx, cmap='gray')
plt.title('Sobel X')
plt.axis("off")

plt.subplot(224)
plt.imshow(sobely, cmap='gray')
plt.title('Sobel Y')
plt.axis("off")
```
![Screenshot 2024-12-13 204133](https://github.com/user-attachments/assets/7729a36b-f3f4-4bf6-bc44-ec2df6b6bcff)
![Screenshot 2024-12-13 204149](https://github.com/user-attachments/assets/e31e2b93-62f7-415e-813e-de3e676963d7)

```
edges = cv2.Canny(image=image, threshold1=100, threshold2=200)

plt.figure(figsize=(18,19))
plt.subplot(121)
plt.imshow(image, cmap='gray')
plt.title('Original')
plt.axis("off")

plt.subplot(122)
plt.imshow(edges, cmap='gray')
plt.title('Edge image')
plt.axis("off")
```
![Screenshot 2024-12-13 204258](https://github.com/user-attachments/assets/bfe659eb-304f-4ef8-9690-b1781fbc1bfe)



