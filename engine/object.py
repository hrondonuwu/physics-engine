"""Defines the base physical entity with mass, position, velocity, and force accumulation."""
class Object:
    def __init__(self, mass, position, velocity=None):
        self.mass = mass
        self.position = list(position)
        self.velocity = list(velocity) if velocity else [0.0] * len(position)
        self.force_accumulator = [0.0] * len(position)

    def apply_force(self, force):
        if len(force) != len(self.force_accumulator):
            raise ValueError(f"Force dimension mismatch. Expected {len(self.force_accumulator)}, got {len(force)}")
            
        for i in range(len(self.force_accumulator)):
            self.force_accumulator[i] += force[i]

    def clear_forces(self):
        self.force_accumulator = [0.0] * len(self.force_accumulator)