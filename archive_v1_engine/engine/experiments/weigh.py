"""Provides an observation interface to indirectly deduce the mass of an object subject to noise profiles."""
import random
from engine.config import SENSOR_CONFIG

class WeighExperiment:
    def __init__(self, precision="medium"):
        # Load precision map from config
        config = SENSOR_CONFIG.get("weigh_sensor", {})
        self.precision_map = config.get("precision_map", {
            "low": 0.5,
            "medium": 0.1,
            "high": 0.01
        })
        
        # Determine standard deviation based on requested precision
        self.noise_std = self.precision_map.get(precision, 0.1)

    def measure(self, obj):
        noisy_mass = obj.mass + random.gauss(0, self.noise_std)
        # Prevent negative mass readings which don't make sense physically
        return max(0.01, noisy_mass)
