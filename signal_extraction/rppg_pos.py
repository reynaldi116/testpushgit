import numpy as np

class POS:
    def __init__(self):
        self.window = []

    def extract(self, frame):
        # Ekstrak ROI wajah dan ubah ke ruang warna
        roi = frame[100:200, 100:200]  # Sementara hardcoded
        mean_rgb = np.mean(np.mean(roi, axis=0), axis=0)
        self.window.append(mean_rgb)

        if len(self.window) > 150:
            self.window.pop(0)

        signal = self._apply_pos(np.array(self.window))
        return signal[-1] if signal.size > 0 else 0

    def _apply_pos(self, rgb_window):
        # Implementasi metode POS (Wang et al.)
        if rgb_window.shape[0] < 32:
            return np.array([])

        rgb_mean = np.mean(rgb_window, axis=0)
        normalized = rgb_window / rgb_mean - 1

        projection = np.array([[0, 1, -1], [-2, 1, 1]])
        S = projection @ normalized.T
        h = S[0] + S[1] * 0.5
        return h
