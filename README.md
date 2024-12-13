# Finals_OpenCV_MEXE4102_VINCENT_JORGE_M._FRANCISCO_-_HAZEL_PUNZALAN

# INTRODUCTION
Edge detection is a fundamental process in computer vision and image processing, aimed at identifying significant transitions or boundaries within an image. These boundaries often correspond to changes in intensity, color, texture, or depth, representing the edges of objects or features in a scene. The ability to detect edges is critical for various applications, including object recognition, image segmentation, and scene understanding.

In essence, edge detection transforms raw image data into a structured representation by highlighting areas of interest. This simplification reduces the computational complexity of subsequent processing tasks while preserving essential geometric information.
# ABSTRACT

# PROJECT METHODS

# CONCLUSION

# ADDITIONAL MATERIALS

 # 16 Basic OpenCV projects

 Part 1: OpenCV Basics
#1 Converting Images to Grayscale
Use the color space conversion code to convert RGB images to grayscale for basic image preprocessing.
import cv2
from google.colab.patches import cv2_imshow

#colorful image - 3 channels
image = cv2.imread("/content/Finals_OpenCV_MEXE4102_VINCENT_JORGE_M._FRANCISCO_-_HAZEL_PUNZALAN/OPENCV IMAGES/asong lubo.jpg")
print(image.shape)

#grayscale image
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2_imshow(gray)


