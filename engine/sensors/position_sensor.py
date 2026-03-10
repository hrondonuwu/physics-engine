"""Models coordinate observation instruments, fusing spatial and lag variance to deter digital leaks."""
import random
from engine.config import SENSOR_CONFIG, DT

class PositionSensor:
    def __init__(self, obj):
        self.obj = obj
        
        config = SENSOR_CONFIG.get("position_sensor", {})
        self.noise_std = config.get("noise_std", 0.05)
        self.lag_std = config.get("lag_std", 0.02)
        
    def tick(self):
        pass

    def read(self):
        noisy_position = []
        for p, v in zip(self.obj.position, self.obj.velocity):
            lag_error = random.gauss(0, self.lag_std) * v
            spatial_error = random.gauss(0, self.noise_std)
            noisy_position.append(p + lag_error + spatial_error)
            
        return noisy_position