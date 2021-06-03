import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm

# Get the fps
prev_time = 0
curr_time = 0

# Start video 
cap =  cv2.VideoCapture(0)
detector = htm.HandDetector()

while True:
  # Get the current image from video 
  succ, img = cap.read()
  
  # Get image from detector
  img = detector.find_hands(img)
  # Get landmark list
  lm_list = detector.find_position(img)
  if len(lm_list) != 0:
    print(lm_list[4])

  # Set fps
  curr_time = time.time()
  fps = 1 / (curr_time - prev_time)
  prev_time = curr_time

  # Show fps
  cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (252, 165, 3), 3)

  cv2.imshow("Image: ", img)
  cv2.waitKey(1)
