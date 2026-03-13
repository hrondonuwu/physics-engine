"""Models internal chronometers with stochastic time fluctuations for clock measurements."""
import random
from engine.config import SENSOR_CONFIG, DT

class TimeSensor:
    def __init__(self):
        config = SENSOR_CONFIG.get("time_sensor", {})
        self.noise_std = config.get("noise_std", 0.005)
        
        self.internal_time = 0.0

    def tick(self):
        self.internal_time += DT

    def read(self):
        # Time shouldn't go completely backwards in reality, but noise can make it fluctuate slightly.
        # Ensure returned time is not less than 0. 
        noisy_time = self.internal_time + random.gauss(0, self.noise_std)
        return max(0.0, noisy_time)
