"""API entry point for the physics engine simulations."""

from engine.world import run_simulation
from engine.sensor import apply_sensor_noise

def run_experiment(masses, init_pos, init_vel, force_graph, noise_level):
    """
    Execute a full 10,000-step simulation and inject sensor noise into the resulting trajectories.

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
    noise_level : float
        Standard deviation of the injected Gaussian noise.

    Returns
    -------
    tuple
        A tuple containing:
        - valid : bool
            True if the simulation completed without degeneracy or singularity, False otherwise.
        - noisy_pos : numpy.ndarray
            Noisy position trajectories, shape (STEPS, N, D).
        - noisy_vel : numpy.ndarray
            Noisy velocity trajectories, shape (STEPS, N, D).
        - sg_acc : numpy.ndarray
            Accelerations estimated via Savitzky-Golay filtering, shape (STEPS, N, D).
    """
    valid, pr_pos, pr_vel = run_simulation(masses, init_pos, init_vel, force_graph)
    noisy_pos, noisy_vel, sg_acc = apply_sensor_noise(pr_pos, pr_vel, noise_level)
    return valid, noisy_pos, noisy_vel, sg_acc
