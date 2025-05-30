import cv2
import numpy as np

prev_gray = None

def extract_respiration(frame):
    global prev_gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    signal = 0
    if prev_gray is not None:
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 
                                             0.5, 3, 15, 3, 5, 1.2, 0)
        signal = np.mean(flow[..., 1])  # Gerakan vertikal
    prev_gray = gray
    return signal
