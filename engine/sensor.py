import numpy as np
from scipy.signal import savgol_filter
from typing import Tuple

from .config import STEPS, DT

def apply_sensor_noise(trajectory_pos: np.ndarray, trajectory_vel: np.ndarray, noise_level: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Inject Gaussian noise and estimate accelerations via Savitzky-Golay filtering.

    Parameters
    ----------
    trajectory_pos : numpy.ndarray
        Ground-truth positions, shape (STEPS, N, D).
    trajectory_vel : numpy.ndarray
        Ground-truth velocities, shape (STEPS, N, D).
    noise_level : float
        Standard deviation for Gaussian noise.

    Returns
    -------
    tuple
        A tuple containing:
        - noisy_positions : numpy.ndarray
            Shape (STEPS, N, D).
        - noisy_velocities : numpy.ndarray
            Shape (STEPS, N, D).
        - sg_accelerations : numpy.ndarray
            Shape (STEPS, N, D).
    
    Notes
    -----
    The Savitzky-Golay filter utilizes a window length of 11 and a polynomial order of 3 as dictated by experimental constraints.
    """
    noisy_positions = trajectory_pos + np.random.normal(0.0, noise_level, trajectory_pos.shape)
    noisy_velocities = trajectory_vel + np.random.normal(0.0, noise_level, trajectory_vel.shape)
    
    window_length = 11
    if STEPS < window_length:
        window_length = STEPS if STEPS % 2 != 0 else STEPS - 1
        
    if window_length > 3:
        sg_accelerations = savgol_filter(
            noisy_velocities,
            window_length=window_length,
            polyorder=3,
            deriv=1,
            delta=DT,
            axis=0,
            mode='interp'
        )
    else:
        sg_accelerations = np.zeros_like(noisy_velocities)
        if STEPS > 1:
            sg_accelerations[1:] = (noisy_velocities[1:] - noisy_velocities[:-1]) / DT
            
    return noisy_positions, noisy_velocities, sg_accelerations
