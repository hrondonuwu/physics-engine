import sys
import os

# Ensure the engine is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.world import World
from engine.object import Object
from engine.sensors.time_sensor import TimeSensor
from engine.sensors.position_sensor import PositionSensor
from engine.experiments.timer import TimerExperiment
from engine.experiments.measure_position import MeasurePositionExperiment
from engine.experiments.weigh import WeighExperiment

def run_discovery_test():
    print("--- DISCOVERY SENSORS TEST ---")
    world = World()
    
    # Hidden inner object we are testing
    true_mass = 5.0
    obj = Object(mass=true_mass, position=[0.0, 0.0])
    world.add_object(obj)
    
    # 1. Weigh checking
    weigh_exp_low = WeighExperiment(precision="low")
    weigh_exp_high = WeighExperiment(precision="high")
    
    w_low = weigh_exp_low.measure(obj)
    w_high = weigh_exp_high.measure(obj)
    
    print(f"True Mass hidden: {true_mass}")
    print(f"Weigh (Low Precision): {w_low}")
    print(f"Weigh (High Precision): {w_high}")
    
    # 2. Timer checking
    t_sensor = TimeSensor()
    world.add_sensor(t_sensor) # Added to world to tick automatically
    
    timer = TimerExperiment(t_sensor)
    
    print("\nStarting timer...")
    timer.start()
    
    # Simulate for some time
    world.run(1.5) # Simulates 1.5 seconds internally
    
    elapsed = timer.stop()
    print(f"Timer reading after 1.5s real simulation: {elapsed} seconds")
    
    # 3. Measure Position checking
    p_sensor = PositionSensor(obj)
    world.add_sensor(p_sensor) # To tick automatically
    p_exp = MeasurePositionExperiment(p_sensor)
    
    world.run(0.2) # Run a bit more so position sensor ticks
    
    pos = p_exp.measure()
    print(f"Measured Object Position: {pos}")

if __name__ == "__main__":
    run_discovery_test()
