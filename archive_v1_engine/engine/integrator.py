"""Provides numerical integration methods for progressing system states over time."""
from engine.config import DT

def semi_implicit_euler(objects):
    for obj in objects:
        if obj.mass <= 0:
            continue

        for i in range(len(obj.velocity)):
            acceleration = obj.force_accumulator[i] / obj.mass
            obj.velocity[i] += acceleration * DT

        for i in range(len(obj.position)):
            obj.position[i] += obj.velocity[i] * DT

        obj.clear_forces()