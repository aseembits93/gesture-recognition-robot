from time import sleep
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

import queue
import threading

from tensorflow.keras.models import load_model



class ProcessGesture:

    def __init__(self):

        self.current_command = None

    def run_gesture_recognition(self, q):
        # initialize mediapipe
        mpHands = mp.solutions.hands
        hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        mpDraw = mp.solutions.drawing_utils

        # Load the gesture recognizer model
        model = load_model('../gesture-recognition-robot/mp_hand_gesture')

        classNames = ['okay', 'peace', 'thumbs up', 'thumbs down',
                    'call me', 'stop', 'rock', 'live long', 'fist', 'smile']

        # Initialize the webcam
        cap = cv2.VideoCapture(0)

        while True:
            # Read each frame from the webcam
            _, frame = cap.read()

            x, y, c = frame.shape

            # Flip the frame vertically
            frame = cv2.flip(frame, 1)
            framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Get hand landmark prediction
            result = hands.process(framergb)

            # post process the result
            if result.multi_hand_landmarks:
                landmarks = []
                for handslms in result.multi_hand_landmarks:
                    for lm in handslms.landmark:
                        lmx = int(lm.x * x)
                        lmy = int(lm.y * y)

                        landmarks.append([lmx, lmy])

                    # Drawing landmarks on frames
                    mpDraw.draw_landmarks(
                        frame, handslms, mpHands.HAND_CONNECTIONS)

                    # Predict gesture
                    prediction = model.predict([landmarks])
                    
                    classID = np.argmax(prediction)
                    className = classNames[classID]

                self.current_command = className
                q.put(self.current_command)
                # show the prediction on the frame
                cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2, cv2.LINE_AA)

                
            else:

                self.current_command = "stop"
                q.put(self.current_command)

            # Show the final output
            cv2.imshow("Output", frame)
            #print(prediction)

            if cv2.waitKey(1) == ord('q'):
                break
            
            sleep(0.3)

        # release the webcam and destroy all active windows
        cap.release()

        cv2.destroyAllWindows()
    
    def getCurrentCommand(self):
        #print(self.current_command)
        return self.current_command


# q = queue.Queue()
# myGesture = ProcessGesture()

# #myGesture.run_gesture_recognition()

# t1 = threading.Thread(target=myGesture.run_gesture_recognition, args= (q,))
# t1.start()
# # t2 = threading.Thread(target=myGesture.getCurrentCommand)
# while True:
#     value = q.get()
#     print(value)

#t2.start()

