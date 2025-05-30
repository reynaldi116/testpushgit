import matplotlib.pyplot as plt
import numpy as np

class RealTimePlot:
    def __init__(self, title="Signal"):
        self.data = [0] * 300
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot(self.data)
        self.ax.set_title(title)
        plt.ion()
        plt.show()

    def update_plot(self, new_val):
        self.data.append(new_val)
        self.data.pop(0)
        self.line.set_ydata(self.data)
        self.line.set_xdata(np.arange(len(self.data)))
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
