import tkinter as tk
from tkinter import Label
import cv2
from PIL import Image, ImageTk
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import butter, filtfilt


# ==== Filter Utility ====
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


# ==== rPPG Dummy POS ====
class POS:
    def __init__(self):
        self.window = []

    def extract(self, frame):
        roi = frame[100:200, 100:200]
        mean_rgb = np.mean(np.mean(roi, axis=0), axis=0)
        self.window.append(mean_rgb)

        if len(self.window) > 150:
            self.window.pop(0)

        signal = self._apply_pos(np.array(self.window))
        return signal[-1] if signal.size > 0 else 0

    def _apply_pos(self, rgb_window):
        if rgb_window.shape[0] < 32:
            return np.array([])

        rgb_mean = np.mean(rgb_window, axis=0)
        normalized = rgb_window / rgb_mean - 1
        projection = np.array([[0, 1, -1], [-2, 1, 1]])
        S = projection @ normalized.T
        h = S[0] + S[1] * 0.5
        return h


# ==== Respirasi Dummy dengan Optical Flow ====
prev_gray = None
def extract_respiration(frame):
    global prev_gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    signal = 0
    if prev_gray is not None:
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 
                                             0.5, 3, 15, 3, 5, 1.2, 0)
        signal = np.mean(flow[..., 1])
    prev_gray = gray
    return signal


# ==== GUI App ====
class App:
    def __init__(self, window):
        self.window = window
        self.window.title("Real-Time rPPG dan Respirasi Monitor")

        self.video_label = Label(window)
        self.video_label.pack()

        # Plot Matplotlib
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(5, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=window)
        self.canvas.get_tk_widget().pack()

        self.rppg_line, = self.ax1.plot([], [], label="rPPG")
        self.ax1.set_ylim(-0.05, 0.05)
        self.ax1.set_title("rPPG Signal")

        self.resp_line, = self.ax2.plot([], [], label="Respiration")
        self.ax2.set_ylim(-1, 1)
        self.ax2.set_title("Respiration Signal")

        self.rppg_data = [0] * 100
        self.resp_data = [0] * 100

        self.cap = cv2.VideoCapture(0)
        self.rppg_extractor = POS()

        self.running = True
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            small_frame = cv2.resize(frame, (320, 240))

            rppg_val = self.rppg_extractor.extract(small_frame)
            resp_val = extract_respiration(small_frame)

            self.rppg_data.append(rppg_val)
            self.rppg_data.pop(0)

            self.resp_data.append(resp_val)
            self.resp_data.pop(0)

            # Update plots
            self.rppg_line.set_data(range(len(self.rppg_data)), self.rppg_data)
            self.ax1.set_xlim(0, len(self.rppg_data))

            self.resp_line.set_data(range(len(self.resp_data)), self.resp_data)
            self.ax2.set_xlim(0, len(self.resp_data))

            self.canvas.draw()

            # Update video frame
            img = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

    def on_close(self):
        self.running = False
        self.cap.release()
        self.window.destroy()


# ==== Jalankan Program ====
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
