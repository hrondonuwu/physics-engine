"""Manages geometric boundary intersections and resolves impact kinematics."""
import random

class Collisions:
    def __init__(self, ground_y=0.0, restitution=0.5, noise_std=0.02):
        self.ground_y = ground_y
        self.restitution = restitution
        self.noise_std = noise_std
        
    def apply(self, objects, old_positions=None):
        for obj in objects:
            y_index = 1 if len(obj.position) > 1 else 0
            
            old_y = obj.position[y_index]
            if old_positions and id(obj) in old_positions:
                old_y = old_positions[id(obj)][y_index]
                
            new_y = obj.position[y_index]
            
            # Intersection test required for cases where v > bound thickness per tick
            if (old_y > self.ground_y and new_y <= self.ground_y) or (new_y <= self.ground_y):
                obj.position[y_index] = self.ground_y
                
                if obj.velocity[y_index] < 0:
                    bounce_factor = self.restitution + random.gauss(0, self.noise_std)
                    bounce_factor = max(0.0, min(1.0, bounce_factor))
                    
                    obj.velocity[y_index] = -obj.velocity[y_index] * bounce_factor
                    
                    # Neutralize v < \epsilon to preempt numerical micro-sinking oscillations
                    if abs(obj.velocity[y_index]) < 0.1:
                        obj.velocity[y_index] = 0.0
