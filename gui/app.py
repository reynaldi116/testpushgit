import tkinter as tk
from threading import Thread
from video_capture import VideoProcessor

def start_gui():
    root = tk.Tk()
    root.title("Real-Time Respirasi dan rPPG Monitor")

    processor = VideoProcessor()

    def start():
        thread = Thread(target=processor.run, daemon=True)
        thread.start()

    btn_start = tk.Button(root, text="Start", command=start)
    btn_start.pack()

    root.mainloop()
