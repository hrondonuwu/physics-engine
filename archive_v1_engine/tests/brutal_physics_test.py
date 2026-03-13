import sys
import os
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.world import World
from engine.object import Object
from engine.config import DT
from engine.forces.gravity import GravityForce
from engine.forces.collisions import Collisions
from engine.forces.friction import Friction
from engine.experiments.drop import DropExperiment
from engine.experiments.push import PushExperiment
from engine.sensors.position_sensor import PositionSensor

def test_galileo_and_law():
    print("--- GALILEO EXPERIMENT ---")
    world = World()
    world.add_force(GravityForce(axis_index=1))
    world.add_force(Collisions(ground_y=0.0, restitution=0.0, noise_std=0.0))
    
    obj_light = Object(mass=1.0, position=[0.0, 100.0])
    obj_heavy = Object(mass=100.0, position=[5.0, 100.0])
    
    sensor_light = PositionSensor(obj_light)
    sensor_heavy = PositionSensor(obj_heavy)
    world.add_sensor(sensor_light)
    world.add_sensor(sensor_heavy)
    
    exp = DropExperiment(world, noise_std=0.0)
    exp.drop(obj_light, drop_height=100.0, initial_x=0.0)
    exp.drop(obj_heavy, drop_height=100.0, initial_x=5.0)
    
    hit_time_light = None
    hit_time_heavy = None
    
    history_t = []
    history_y = []
    
    while world.time < 10.0:
        world.step()
        l_pos = sensor_light.read()
        h_pos = sensor_heavy.read()
        
        if hit_time_light is None and l_pos[1] <= 0.0:
            hit_time_light = world.time
            
        if hit_time_heavy is None and h_pos[1] <= 0.0:
            hit_time_heavy = world.time
            
        if l_pos[1] > 0.0:
            history_t.append(world.time)
            history_y.append(100.0 - l_pos[1])
            
        if hit_time_light and hit_time_heavy:
            break
            
    print(f"Light object hit at: {hit_time_light:.4f}s")
    print(f"Heavy object hit at: {hit_time_heavy:.4f}s")
    assert hit_time_light is not None and hit_time_heavy is not None
    assert abs(hit_time_light - hit_time_heavy) < 0.05, "Masses fell at different rates!"
    print("Galileo verification passed: Objects fell at same rate.")
    
    g_theoretical = 9.80665
    if len(history_t) > 10:
        mid = len(history_t) // 2
        t = history_t[mid]
        y_obs = history_y[mid]
        
        g_obs = (2 * y_obs) / (t**2)
        error = abs(g_obs - g_theoretical)
        print(f"Theoretical Distance at t={t:.4f}: {0.5 * g_theoretical * t**2:.4f}")
        print(f"Observed Distance at t={t:.4f}: {y_obs:.4f}")
        print(f"Discovered Law Gravity Estimate: {g_obs:.4f}")
        print(f"Discovered Law Error: {error:.4f} m/s^2")
    
def test_inertia():
    print("\n--- INERTIA & FRICTION EXPERIMENT ---")
    
    # Test 1: No friction
    world_no_fric = World()
    obj1 = Object(mass=2.0, position=[0.0, 0.0])
    world_no_fric.add_object(obj1)
    world_no_fric.add_force(GravityForce(axis_index=1))
    world_no_fric.add_force(Collisions(ground_y=0.0))
    world_no_fric.add_force(Friction(mu_s=0.0, mu_k=0.0))
    
    push = PushExperiment(world_no_fric, force_noise_std=0.0, pos_noise_std=0.0, vel_noise_std=0.0)
    push.apply(obj1, direction=[1.0, 0.0], strength="hard", duration=0.1) 
    
    world_no_fric.run(0.1)
    v_after_push = obj1.velocity[0]
    
    world_no_fric.run(2.0)
    v_final = obj1.velocity[0]
    print(f"Frictionless velocity after push: {v_after_push:.4f}, after 2s void: {v_final:.4f}")
    assert abs(v_after_push - v_final) < 1e-4, "Object slowed down without friction!"
    
    # Test 2: Friction
    world_fric = World()
    obj2 = Object(mass=2.0, position=[0.0, 0.0])
    world_fric.add_object(obj2)
    world_fric.add_force(GravityForce(axis_index=1))
    world_fric.add_force(Collisions(ground_y=0.0))
    world_fric.add_force(Friction(mu_s=0.5, mu_k=0.3))
    
    push2 = PushExperiment(world_fric, force_noise_std=0.0, pos_noise_std=0.0, vel_noise_std=0.0)
    push2.apply(obj2, direction=[1.0, 0.0], strength="hard", duration=0.1)
    
    world_fric.run(0.1)
    v2_after_push = obj2.velocity[0]
    
    world_fric.run(5.0)
    v2_final = obj2.velocity[0]
    
    print(f"Friction velocity after push: {v2_after_push:.4f}, after 5s sliding: {v2_final:.4f}")
    assert abs(v2_final) < 1e-4, "Object did not stop despite friction!"
    print("Inertia verification passed.")

def test_tunneling():
    print("\n--- TUNNELING STRESS TEST ---")
    world = World()
    world.add_force(GravityForce(axis_index=1))
    world.add_force(Collisions(ground_y=0.0, restitution=0.5, noise_std=0.0))
    
    obj = Object(mass=10.0, position=[0.0, 1000.0])
    exp = DropExperiment(world, noise_std=0.0)
    exp.drop(obj, drop_height=1000.0, initial_x=0.0)
    
    min_y = float('inf')
    while world.time < 20.0:
        world.step()
        if obj.position[1] < min_y:
            min_y = obj.position[1]
            
    print(f"Lowest Y position recorded: {min_y}")
    assert min_y >= 0.0, f"Tunneling detected! Object fell to {min_y}"
    print("Tunneling prevented successfully.")

if __name__ == "__main__":
    test_galileo_and_law()
    test_inertia()
    test_tunneling()
    print("\nALL BRUTAL PHYSICS TESTS PASSED.")
