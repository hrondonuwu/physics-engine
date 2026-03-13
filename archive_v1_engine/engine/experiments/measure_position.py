"""Provides a safe interface for experiments to query noisy coordinate states via position sensors."""
class MeasurePositionExperiment:
    def __init__(self, position_sensor):
        self.sensor = position_sensor

    def measure(self):
        return self.sensor.read()
