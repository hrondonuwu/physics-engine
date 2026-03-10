"""Coordinates the main simulation loop, force application, and integration phases."""
from engine.config import DT
from engine.integrator import semi_implicit_euler

class World:
    def __init__(self):
        self.objects = []
        self.forces = []
        self.collision_handlers = []
        self.sensors = []
        self.active_events = []
        self.step_count = 0
        self.time = 0.0

    def add_object(self, obj):
        if obj not in self.objects:
            self.objects.append(obj)

    def add_force(self, force):
        if type(force).__name__ == "Collisions":
            self.collision_handlers.append(force)
        else:
            self.forces.append(force)

    def add_sensor(self, sensor):
        self.sensors.append(sensor)

    def step(self):
        self.step_count += 1
        self.time = self.step_count * DT
        
        # Raycasting requires tracking motion vectors pre-integration
        old_positions = {id(obj): list(obj.position) for obj in self.objects}

        for force in self.forces:
            force.apply(self.objects)
            
        events_to_keep = []
        for event in self.active_events:
            event["object"].apply_force(event["force_vector"])
            event["remaining_time"] -= DT
            if event["remaining_time"] > 0:
                events_to_keep.append(event)
        self.active_events = events_to_keep

        semi_implicit_euler(self.objects)
        
        # Collisions resolve post-integration to prevent state paradoxes
        for handler in self.collision_handlers:
            handler.apply(self.objects, old_positions)

        for sensor in self.sensors:
            if hasattr(sensor, "tick"):
                sensor.tick()

    def run(self, duration):
        steps = round(duration / DT)
        for _ in range(steps):
            self.step()