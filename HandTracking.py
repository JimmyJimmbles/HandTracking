import cv2
import mediapipe as mp
import time

cap =  cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

while True:
  succ, img = cap.read()
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  results = hands.process(img_rgb)
  
  if results.multi_hand_landmarks:
    for hand_mark in results.multi_hand_landmarks:
      mp_draw.draw_landmarks(img, hand_mark, mp_hands.HAND_CONNECTIONS)

  cv2.imshow("Image: ", img) 
  cv2.waitKey(1)
