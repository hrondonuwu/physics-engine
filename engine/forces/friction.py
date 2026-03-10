"""Implements macroscopic contact forces (static and kinetic friction) opposing horizontal motion."""
from engine.config import GRAVITY, DT

class Friction:
    def __init__(self, ground_y=0.0, mu_s=0.6, mu_k=0.4):
        self.ground_y = ground_y
        self.mu_s = mu_s
        self.mu_k = mu_k
        if isinstance(GRAVITY, dict):
            self.g_strength = abs(GRAVITY.get("strength", -9.81))
        else:
            self.g_strength = abs(float(GRAVITY))

    def apply(self, objects):
        for obj in objects:
            if obj.mass <= 0:
                continue
                
            if len(obj.position) == 1:
                y_index = None
                on_ground = True
            else:
                y_index = 1
                on_ground = (obj.position[y_index] <= self.ground_y + 1e-5)
            
            if on_ground:
                N = obj.mass * self.g_strength
                
                v_sq = 0.0
                for i in range(len(obj.velocity)):
                    if i != y_index:
                        v_sq += obj.velocity[i]**2
                speed = v_sq**0.5
                
                f_ext_sq = 0.0
                for i in range(len(obj.force_accumulator)):
                    if i != y_index:
                        f_ext_sq += obj.force_accumulator[i]**2
                external_force_mag = f_ext_sq**0.5

                if speed < 1e-4:
                    max_static = self.mu_s * N
                    if external_force_mag <= max_static:
                        for i in range(len(obj.force_accumulator)):
                            if i != y_index:
                                obj.force_accumulator[i] = 0.0
                        for i in range(len(obj.velocity)):
                            if i != y_index:
                                obj.velocity[i] = 0.0
                    else:
                        friction_mag = self.mu_k * N
                        for i in range(len(obj.force_accumulator)):
                            if i != y_index:
                                dir_component = obj.force_accumulator[i] / external_force_mag
                                obj.force_accumulator[i] -= dir_component * friction_mag
                else:
                    friction_mag = self.mu_k * N
                    for i in range(len(obj.force_accumulator)):
                        if i != y_index:
                            dir_component = obj.velocity[i] / speed
                            f_fric = dir_component * friction_mag
                            
                            # Clamps deceleration to exactly 0 m/s relative frame to prevent frictionless oscillation
                            max_fric = (obj.mass * abs(obj.velocity[i])) / DT
                            if abs(f_fric) > max_fric:
                                sign = 1 if f_fric > 0 else -1
                                f_fric = sign * max_fric
                                
                            obj.force_accumulator[i] -= f_fric
