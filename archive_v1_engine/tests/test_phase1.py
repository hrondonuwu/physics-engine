import sys
import os

# Ensure the engine is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.world import World
from engine.object import Object
from engine.forces.gravity import GravityForce
from engine.forces.collisions import Collisions
from engine.experiments.drop import DropExperiment
from engine.experiments.push import PushExperiment
from engine.sensors.position_sensor import PositionSensor

def run_drop_test():
    print("--- DROP TEST ---")
    world = World()
    world.add_force(GravityForce())
    world.add_force(Collisions(ground_y=0.0, restitution=0.5))
    
    # AI only knows it creates an object of some mass (hidden here)
    obj = Object(mass=2.0, position=[0.0, 10.0])
    
    sensor = PositionSensor(obj)
    world.add_sensor(sensor)
    
    experiment = DropExperiment(world)
    print("Initiating drop from 10m...")
    experiment.drop(obj, drop_height=10.0)
    
    # Simulate for 2 seconds
    world.run(2.0)
    
    # Read sensor
    reading = sensor.read()
    print(f"Sensor reading after 2s: {reading}")

def run_push_test():
    print("\n--- PUSH TEST ---")
    world = World()
    
    # Assume 1D motion for push on frictionless surface
    obj = Object(mass=1.0, position=[0.0])
    world.add_object(obj)
    
    sensor = PositionSensor(obj)
    world.add_sensor(sensor)
    
    experiment = PushExperiment(world)
    print("Initiating 'hard' push to the right...")
    experiment.apply(obj, direction=[1.0], strength="hard", duration=0.5)
    
    # Simulate for 1 second
    world.run(1.0)
    
    # Read sensor
    reading = sensor.read()
    print(f"Sensor reading after 1s: {reading}")

if __name__ == "__main__":
    run_drop_test()
    run_push_test()
