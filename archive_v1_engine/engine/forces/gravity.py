"""Applies continuously uniform global acceleration forces to all massive objects."""
from engine.config import GRAVITY

class GravityForce:
    def __init__(self, axis_index=None):
        if isinstance(GRAVITY, dict):
            self.g_strength = GRAVITY.get("strength", -9.81)
        else:
            self.g_strength = float(GRAVITY)
            
        self.axis_index = axis_index
        self._cached_vectors = {}

    def apply(self, objects):
        for obj in objects:
            if obj.mass <= 0:
                 continue
                 
            dim = len(obj.position)
            axis = self.axis_index if self.axis_index is not None else (1 if dim > 1 else 0)
            
            if dim not in self._cached_vectors:
                self._cached_vectors[dim] = [0.0] * dim
                
            f_vec = self._cached_vectors[dim]
            for i in range(dim):
                f_vec[i] = 0.0
                
            f_vec[axis] = obj.mass * self.g_strength
            obj.apply_force(f_vec)