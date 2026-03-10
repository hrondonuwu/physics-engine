"""Defines gravity-based drop experiments with initial condition variances."""
import random

class DropExperiment:
    def __init__(self, world, noise_std=1e-3):
        self.world = world
        self.noise_std = noise_std

    def drop(self, obj, drop_height, initial_x=0.0):
        # Gaussian injection corrupts deterministic perfect symmetry states
        actual_x = initial_x + random.gauss(0, self.noise_std)
        actual_y = drop_height + random.gauss(0, self.noise_std)
        
        if len(obj.position) > 1:
            obj.position[0] = actual_x
            obj.position[1] = actual_y
        else:
            obj.position[0] = actual_y
            
        obj.velocity = [random.gauss(0, self.noise_std) for _ in obj.velocity]
        
        if obj not in self.world.objects:
            self.world.add_object(obj)
