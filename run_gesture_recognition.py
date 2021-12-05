import cv2
import socket, pickle
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model

class Run_gesture_recognition:

    def __init__(self):
    # initialize mediapipe
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mpDraw = mp.solutions.drawing_utils

        # Load the gesture recognizer model
        self.model = load_model('./mp_hand_gesture')

        self.classNames = ['okay', 'peace', 'thumbs up', 'thumbs down',
                    'call me', 'stop', 'rock', 'live long', 'fist', 'smile']

        # Initialize the webcam
        self.cap = cv2.VideoCapture(0)

        #Initialize connection
        host = '192.168.2.138'
        #host = '127.0.0.1'
        port = 7021


        #Initialize belief array for discrete bayes filter

        self.belief = np.array((1/len(self.classNames) * np.ones(len(self.classNames))))

        print("Starting")

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((host, port))
        s.listen(1) #Allow only 1 connection

        self.c, addr = s.accept()
        print("Started")


        self.current_gesture = ""
        self.run()


    def normalize_belief(self, distribution):

        distribution /= sum(distribution.astype(float))

        distribution = np.clip(distribution, 1e-5, 0.99)

        return distribution



    def update_belief(self, belief, gestureReading, prob_correct): 

        gestureReading_beliefIndex = self.classNames.index(gestureReading)#belief.index(gestureReading)
        #print("lol")
        #print(gestureReading_beliefIndex)
        scale = prob_correct / (1 - prob_correct)
        #print(belief)
        for i, gestureProb in enumerate(belief):

            if gestureReading_beliefIndex == i:

                gestureProb *= scale 
                belief[i] = gestureProb


        updatedBelief = self.normalize_belief(belief)

        self.belief = updatedBelief

    def run(self, n=10):
        print("Running")
        last_n = ['stop' for _ in range(n)]
        while True:
            # Read each frame from the webcam
            _, frame = self.cap.read()

            x, y, c = frame.shape

            # Flip the frame vertically
            frame = cv2.flip(frame, 1)
            framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Get hand landmark prediction
            result = self.hands.process(framergb)

            # post process the result
            post_processed = False
            if result.multi_hand_landmarks:
                landmarks = []
                for handslms in result.multi_hand_landmarks:
                    for lm in handslms.landmark:
                        lmx = int(lm.x * x)
                        lmy = int(lm.y * y)

                        landmarks.append([lmx, lmy])

                    rect = cv2.minAreaRect(np.array([landmarks]))
                    
                    gestureAngle = rect[2]
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)

                    # Drawing landmarks on frames
                    self.mpDraw.draw_landmarks(
                        frame, handslms, self.mpHands.HAND_CONNECTIONS)

                    # Predict gesture
                    prediction = self.model.predict([landmarks])
                    classID = np.argmax(prediction)
                    post_processed = True

                    last_n = last_n[1:] + [self.classNames[classID]]
                    className = max(last_n, key=last_n.count)

                    self.update_belief(self.belief, className, 0.9)

                    most_likely_current_gesture_index = np.argmax(self.belief)
                    className = self.classNames[most_likely_current_gesture_index]
                    self.current_gesture = self.classNames[most_likely_current_gesture_index]
                    #print(f"probability of {self.current_gesture} is {np.amax(self.belief)}")

                cv2.drawContours(frame,[box],0,(0,191,255),1)
                #cv2.rectangle(frame, box[], (255,0,0), 2)
                # show the prediction on the frame
                cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2, cv2.LINE_AA)

            # Show the final output
            cv2.imshow("Output", frame)

            if post_processed:

                msg = [className, gestureAngle]
                msg = pickle.dumps(msg)
                self.c.send(msg)

            if cv2.waitKey(1) == ord('q'):
                break

        # release the webcam and destroy all active windows
        self.cap.release()
        cv2.destroyAllWindows()

Run_gesture_recognition()
