"""Provides a chronometer interface for experiments to measure noisy elapsed elapsed time."""
class TimerExperiment:
    def __init__(self, time_sensor):
        self.sensor = time_sensor
        self.start_reading = 0.0
        self.is_running = False

    def start(self):
        self.start_reading = self.sensor.read()
        self.is_running = True

    def stop(self):
        if not self.is_running:
            return 0.0
            
        end_reading = self.sensor.read()
        self.is_running = False
        
        # Calculate elapsed time (could be slightly negative due to high noise or tiny durations)
        # So we clamp it to 0.0 logically
        elapsed = max(0.0, end_reading - self.start_reading)
        return elapsed

    def reset(self):
        self.is_running = False
        self.start_reading = 0.0
