# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import cv2
import mediapipe as mp
import sys


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose,mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
            
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    scale_percent = 150 # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
     
    # resize image
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    results_pose = pose.process(image)
    results_hand = hands.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results_hand.multi_hand_landmarks:
        # Get the landmarks of the first hand detected
        landmarks = results_hand.multi_hand_landmarks[0]
        
        # Get the indices of the landmarks representing the left arm
        left_arm_landmark_indices = [11, 13, 15, 17, 19, 5]
        
        # Get the coordinates of the left arm
        left_arm_coordinates = []
        for index in left_arm_landmark_indices:
            if landmarks.landmark[index].visibility < 0 or landmarks.landmark[index].presence < 0:
                # Landmark not detected
                left_arm_coordinates.append(None)
              
            else:
                # Landmark detected
                left_arm_coordinates.append((int(landmarks.landmark[index].x * image.shape[1]), int(landmarks.landmark[index].y * image.shape[0])))
        
        # Print the coordinates of the left arm
        left_arm_coordinates_str = ', '.join([f'({x},{y})' for x, y in left_arm_coordinates])
        print("coordinates:"+left_arm_coordinates_str)
        
        for landmarks in results_hand.multi_hand_landmarks:
           mp_drawing.draw_landmarks(
           image,
           landmarks,
           mp_hands.HAND_CONNECTIONS,
           mp_drawing_styles.get_default_hand_landmarks_style(),
           mp_drawing_styles.get_default_hand_connections_style())
           
           
    if results_pose.pose_landmarks:   
        for landmark in results_pose.pose_landmarks.landmark:
            mp_drawing.draw_landmarks(
                image,
                results_pose.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    
    sys.stdout.flush()
    if cv2.waitKey(5) & 0xFF == ord('q'):
      break
cv2.destroyAllWindows()
cap.release()
