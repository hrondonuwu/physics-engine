"""Defines global simulation parameters, sensor noise profiles, and active capabilities."""
DT = 0.001

UNITS = {
    "length": "m",
    "time": "s",
    "mass": "kg"
}

GRAVITY = {
    "strength": -9.80665,
    "exponent": 2
}

SENSOR_CONFIG = {
    "position_sensor": {
        "noise_std": 0.05,
        "update_rate": 10
    },
    "weigh_sensor": {
        "precision_map": {
            "low": 0.5,
            "medium": 0.1,
            "high": 0.01
        }
    },
    "time_sensor": {
        "noise_std": 0.005
    }
}

ACTIVE_FORCES = ["gravity", "collisions", "friction"]

ACTIVE_INTEGRATOR = "euler"

INTEGRATOR_CONFIG = {
    "euler": {},
    "rk4": {},
    "verlet": {"substeps": 4}
}
