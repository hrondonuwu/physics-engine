import numpy as np
from typing import Tuple

from .config import DT, STEPS, MAX_POS, MIN_DIST, MIN_STD
from .integrator import step_velocity_verlet
from .forces import evaluate_forces

def run_simulation(masses: np.ndarray, init_pos: np.ndarray, init_vel: np.ndarray, force_graph: dict) -> Tuple[bool, np.ndarray, np.ndarray]:
    """
    Execute mathematical integration over 10,000 steps with strict validity constraints.

    Parameters
    ----------
    masses : numpy.ndarray
        Particle masses, shape (N,).
    init_pos : numpy.ndarray
        Initial particle positions, shape (N, D).
    init_vel : numpy.ndarray
        Initial particle velocities, shape (N, D).
    force_graph : dict
        Force interaction graph dictionary specifying active forces and parameters.

    Returns
    -------
    tuple
        A tuple containing:
        - valid : bool
            True if the simulation completed without degeneracy or singularity, False otherwise.
        - trajectory_pos : numpy.ndarray
            Positions matrix, shape (STEPS, N, D).
        - trajectory_vel : numpy.ndarray
            Velocities matrix, shape (STEPS, N, D).
    """
    n_particles, dims = init_pos.shape
    
    pos = init_pos.copy()
    vel = init_vel.copy()
    mass_col = masses.reshape(-1, 1)
    
    trajectory_pos = np.zeros((STEPS, n_particles, dims))
    trajectory_vel = np.zeros((STEPS, n_particles, dims))
    
    acc = evaluate_forces(pos, vel, force_graph) / mass_col
    
    valid = True

    for step in range(STEPS):
        trajectory_pos[step] = pos
        trajectory_vel[step] = vel
        
        if not np.all(np.isfinite(pos)) or np.max(np.abs(pos)) > MAX_POS * 10:
            valid = False
            break
            
        pos, vel, acc = step_velocity_verlet(
            pos, vel, acc, DT, mass_col,
            force_func=lambda p, v: evaluate_forces(p, v, force_graph)
        )
        
        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dist, np.inf)
        if np.min(dist) < MIN_DIST:
            valid = False
            break
            
    if valid:
        if not np.all(np.isfinite(trajectory_pos)) or np.max(np.abs(trajectory_pos)) > MAX_POS:
            valid = False
            
    if valid:
        pos_std = np.std(trajectory_pos, axis=0)
        pos_std_norm = np.linalg.norm(pos_std, axis=-1)
        if np.any(pos_std_norm < MIN_STD):
            valid = False

    return valid, trajectory_pos, trajectory_vel
