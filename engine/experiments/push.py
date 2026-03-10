"""Defines impulse-based push experiments with applied kinematic variances."""
import random

class PushExperiment:
    STRENGTH_MAP = {
        "soft": 1.0,
        "medium": 5.0,
        "hard": 10.0,
        "very_hard": 20.0
    }

    def __init__(self, world, force_noise_std=0.5, pos_noise_std=1e-3, vel_noise_std=1e-3):
        self.world = world
        self.force_noise_std = force_noise_std
        self.pos_noise_std = pos_noise_std
        self.vel_noise_std = vel_noise_std
        self.active_pushes = []

    def apply(self, obj, direction, strength="medium", duration=1.0):
        if strength not in self.STRENGTH_MAP:
            strength = "medium"
            
        base_force = self.STRENGTH_MAP[strength]
        
        actual_force_magnitude = abs(base_force + random.gauss(0, self.force_noise_std))
        
        dir_magnitude = sum(d**2 for d in direction)**0.5
        if dir_magnitude == 0:
            raise ValueError("Push direction cannot be a zero vector")
            
        force_vector = [(d / dir_magnitude) * actual_force_magnitude for d in direction]
        
        # Micro-variances injected to obfuscate internal bounding volume states from AI observers
        for i in range(len(obj.position)):
            obj.position[i] += random.gauss(0, self.pos_noise_std)
        for i in range(len(obj.velocity)):
            obj.velocity[i] += random.gauss(0, self.vel_noise_std)
        
        push_event = {
            "object": obj,
            "force_vector": force_vector,
            "remaining_time": duration
        }
        
        if obj not in self.world.objects:
            self.world.add_object(obj)
            
        self.world.active_events.append(push_event)