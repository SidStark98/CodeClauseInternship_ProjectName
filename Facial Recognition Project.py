import cv2
import numpy as np

#read an image
img = cv2.imread("img/fig1.jpg")

#converting to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#cv2.imshow("window", img_gray)
#cv2.waitKey(0)

#Applying Gaussian Blurring
blurred_img = cv2.GaussianBlur(img_gray, (5, 5), 0)
cv2.imshow("Gaussian Blurred Image", blurred_img)
cv2.waitKey(0)

#Applying Canny edge Detection
edges = cv2.Canny(blurred_img, 100, 200) #Adjusted thresholds as needed
cv2.imshow("Canny Edge Detection", edges)
cv2.waitKey(0)

#Applying Thresholding
_, threshold_img = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
cv2.imshow("Threshold Image", threshold_img)
cv2.waitKey(0)

#Closing all windows
cv2.destroyAllWindows()

# Enabling Camera Live Feed with Face Detection
# Loading Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start Video Capture
video_cap = cv2.VideoCapture(0)
while True:
    ret, video_data = video_cap.read()
    if not ret:
        print("Error: Failed to capture video.")
        break

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected faces
    for (x,y,w,h) in faces:
        cv2.rectangle(video_data, (x, y), (x+w, y+h), (255,0,0), 2)

    # Display the video feed with detected faces
    cv2.imshow("Live Video Feed with Face Detection", video_data)

    # press 'a' to exit
    if cv2.waitKey(1) & 0xFF == ord("a"):
        break

video_cap.release()
cv2.destroyAllWindows()