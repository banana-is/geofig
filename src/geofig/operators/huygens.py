import numpy as np

class HuygensEvolution:
    def __init__(self, front, wavelet):
        self.front = front
        self.wavelet = wavelet

    def evolved_wavelets(self, t: float, n_front: int = 80, n_wavelet: int = 120):
        front_pts = self.front.sample(n_front)
        wavelets = []
        for p in front_pts:
            W = self.wavelet.sample(n_wavelet, scale=t)
            wavelets.append(W + p)
        return wavelets