import cv2
import mediapipe as mp
import time

# Start video 
cap =  cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Get the fps
prev_time = 0
curr_time = 0

while True:
  # Get the current image from video 
  succ, img = cap.read()
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  results = hands.process(img_rgb)
  
  # Make sure we have landmarks
  if results.multi_hand_landmarks:
    # Loop through all hands
    for hand_mark in results.multi_hand_landmarks:
      for id, lm in enumerate(hand_mark.landmark):
        # Get the shape of the image(hand)
        h, w, c = img.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        print(id, cx, cy)
      
      # Drawing the landmarks on the hand
      mp_draw.draw_landmarks(img, hand_mark, mp_hands.HAND_CONNECTIONS)

  # Set fps
  curr_time = time.time()
  fps = 1 / (curr_time - prev_time)
  prev_time = curr_time

  # Show fps
  cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (252, 165, 3), 3)

  cv2.imshow("Image: ", img) 
  cv2.waitKey(1)
