import numpy as np

def step_velocity_verlet(pos: np.ndarray, vel: np.ndarray, acc: np.ndarray, dt: float, mass: np.ndarray, force_func: callable) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform a single step of the Velocity Verlet symplectic numerical integration.

    Parameters
    ----------
    pos : numpy.ndarray
        Particle positions, shape (N, D).
    vel : numpy.ndarray
        Particle velocities, shape (N, D).
    acc : numpy.ndarray
        Particle accelerations, shape (N, D).
    dt : float
        Integration time step size.
    mass : numpy.ndarray
        Particle masses, shape (N, 1).
    force_func : callable
        Function taking (pos, vel) and returning the forces of shape (N, D).

    Returns
    -------
    tuple
        A tuple containing:
        - pos_new : numpy.ndarray
            New positions at t + dt, shape (N, D).
        - vel_new : numpy.ndarray
            New velocities at t + dt, shape (N, D).
        - acc_new : numpy.ndarray
            New accelerations at t + dt, shape (N, D).
            
    Notes
    -----
    Velocity Verlet is perfectly symplectic strictly for conservative (position-dependent) forces. 
    For dissipative (velocity-dependent) forces like drag, it requires an explicit half-step approximation 
    for velocity, meaning perfect symplecticity is lost and energy drift will occur.
    """
    v_half = vel + 0.5 * acc * dt
    pos_new = pos + v_half * dt
    
    forces_new = force_func(pos_new, v_half)
    acc_new = forces_new / mass
    
    vel_new = v_half + 0.5 * acc_new * dt
    
    return pos_new, vel_new, acc_new
