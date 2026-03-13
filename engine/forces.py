import numpy as np

def _safe_dir(diff: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """
    Safely compute direction vectors, avoiding division by zero.

    Parameters
    ----------
    diff : numpy.ndarray
        Displacement vectors, shape (N, N, D).
    dist : numpy.ndarray
        Distance magnitudes, shape (N, N).

    Returns
    -------
    numpy.ndarray
        Normalized direction vectors, shape (N, N, D).
    """
    safe_dist = np.maximum(dist, 1e-12)
    return diff / safe_dist[..., np.newaxis]

def linear_spring(diff: np.ndarray, dist: np.ndarray, k: np.ndarray, r0: np.ndarray) -> np.ndarray:
    """
    Compute linear spring force.

    Parameters
    ----------
    diff : numpy.ndarray
        Displacement vectors, shape (N, N, D).
    dist : numpy.ndarray
        Distance magnitudes, shape (N, N).
    k : numpy.ndarray
        Spring constants, shape (N, N).
    r0 : numpy.ndarray
        Rest lengths, shape (N, N).

    Returns
    -------
    numpy.ndarray
        Force vectors, shape (N, N, D).
    """
    direction = _safe_dir(diff, dist)
    mag = -k * (dist - r0)
    return mag[..., np.newaxis] * direction

def duffing_oscillator(diff: np.ndarray, dist: np.ndarray, k: np.ndarray, r0: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    Compute Duffing oscillator force.

    Parameters
    ----------
    diff : numpy.ndarray
        Displacement vectors, shape (N, N, D).
    dist : numpy.ndarray
        Distance magnitudes, shape (N, N).
    k : numpy.ndarray
        Linear spring constants, shape (N, N).
    r0 : numpy.ndarray
        Rest lengths, shape (N, N).
    alpha : numpy.ndarray
        Nonlinear spring constants, shape (N, N).

    Returns
    -------
    numpy.ndarray
        Force vectors, shape (N, N, D).
    """
    direction = _safe_dir(diff, dist)
    disp = dist - r0
    mag = -k * disp - alpha * (disp ** 3)
    return mag[..., np.newaxis] * direction

def velocity_damper(vel: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Compute global velocity damping force.

    Parameters
    ----------
    vel : numpy.ndarray
        Particle velocities, shape (N, D).
    c : numpy.ndarray
        Damping coefficients, shape (N, 1) or scalar.

    Returns
    -------
    numpy.ndarray
        Damping force vectors, shape (N, D).
    """
    return -c * vel

def newtonian_gravity(diff: np.ndarray, dist: np.ndarray, G: float, m_i: np.ndarray, m_j: np.ndarray) -> np.ndarray:
    """
    Compute Newtonian gravitational force.

    Parameters
    ----------
    diff : numpy.ndarray
        Displacement vectors, shape (N, N, D).
    dist : numpy.ndarray
        Distance magnitudes, shape (N, N).
    G : float
        Gravitational constant.
    m_i : numpy.ndarray
        Masses of particle i, shape (N, 1).
    m_j : numpy.ndarray
        Masses of particle j, shape (1, N).

    Returns
    -------
    numpy.ndarray
        Force vectors, shape (N, N, D).
    """
    direction = _safe_dir(diff, dist)
    safe_dist = np.maximum(dist, 1e-12)
    mag = -G * (m_i * m_j) / (safe_dist ** 2)
    return mag[..., np.newaxis] * direction

def coulomb(diff: np.ndarray, dist: np.ndarray, k_e: float, q_i: np.ndarray, q_j: np.ndarray) -> np.ndarray:
    """
    Compute Coulomb force.

    Parameters
    ----------
    diff : numpy.ndarray
        Displacement vectors, shape (N, N, D).
    dist : numpy.ndarray
        Distance magnitudes, shape (N, N).
    k_e : float
        Coulomb constant.
    q_i : numpy.ndarray
        Charges of particle i, shape (N, 1).
    q_j : numpy.ndarray
        Charges of particle j, shape (1, N).

    Returns
    -------
    numpy.ndarray
        Force vectors, shape (N, N, D).
    """
    direction = _safe_dir(diff, dist)
    safe_dist = np.maximum(dist, 1e-12)
    mag = k_e * (q_i * q_j) / (safe_dist ** 2)
    return mag[..., np.newaxis] * direction

def yukawa(diff: np.ndarray, dist: np.ndarray, k: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    """
    Compute Yukawa force.

    Parameters
    ----------
    diff : numpy.ndarray
        Displacement vectors, shape (N, N, D).
    dist : numpy.ndarray
        Distance magnitudes, shape (N, N).
    k : numpy.ndarray
        Coupling constants, shape (N, N).
    kappa : numpy.ndarray
        Inverse length scales, shape (N, N).

    Returns
    -------
    numpy.ndarray
        Force vectors, shape (N, N, D).
    """
    direction = _safe_dir(diff, dist)
    safe_dist = np.maximum(dist, 1e-12)
    mag = k * np.exp(-kappa * safe_dist) * (1.0 / (safe_dist ** 2) + kappa / safe_dist)
    return mag[..., np.newaxis] * direction

def morse(diff: np.ndarray, dist: np.ndarray, D_e: np.ndarray, a: np.ndarray, r_e: np.ndarray) -> np.ndarray:
    """
    Compute Morse potential force.

    Parameters
    ----------
    diff : numpy.ndarray
        Displacement vectors, shape (N, N, D).
    dist : numpy.ndarray
        Distance magnitudes, shape (N, N).
    D_e : numpy.ndarray
        Well depths, shape (N, N).
    a : numpy.ndarray
        Well widths, shape (N, N).
    r_e : numpy.ndarray
        Equilibrium distances, shape (N, N).

    Returns
    -------
    numpy.ndarray
        Force vectors, shape (N, N, D).
    """
    direction = _safe_dir(diff, dist)
    exp_term = np.exp(-a * (dist - r_e))
    mag = -2 * a * D_e * (1 - exp_term) * exp_term
    return mag[..., np.newaxis] * direction

def buckingham(diff: np.ndarray, dist: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Compute Buckingham potential force.

    Parameters
    ----------
    diff : numpy.ndarray
        Displacement vectors, shape (N, N, D).
    dist : numpy.ndarray
        Distance magnitudes, shape (N, N).
    A : numpy.ndarray
        Repulsive exponential constants, shape (N, N).
    B : numpy.ndarray
        Repulsive exponential scales, shape (N, N).
    C : numpy.ndarray
        Attractive dispersion constants, shape (N, N).

    Returns
    -------
    numpy.ndarray
        Force vectors, shape (N, N, D).
    """
    direction = _safe_dir(diff, dist)
    safe_dist = np.maximum(dist, 1e-12)
    mag = A * B * np.exp(-B * safe_dist) - 6 * C / (safe_dist ** 7)
    return mag[..., np.newaxis] * direction

def hertzian_contact(diff: np.ndarray, dist: np.ndarray, K: np.ndarray, R_sum: np.ndarray) -> np.ndarray:
    """
    Compute Hertzian contact force.

    Parameters
    ----------
    diff : numpy.ndarray
        Displacement vectors, shape (N, N, D).
    dist : numpy.ndarray
        Distance magnitudes, shape (N, N).
    K : numpy.ndarray
        Contact stiffness constants, shape (N, N).
    R_sum : numpy.ndarray
        Sum of radii for contact, shape (N, N).

    Returns
    -------
    numpy.ndarray
        Force vectors, shape (N, N, D).
    """
    direction = _safe_dir(diff, dist)
    mag = np.where(dist < R_sum, K * (R_sum - dist)**1.5, 0.0)
    return mag[..., np.newaxis] * direction

def pairwise_drag(vel: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Compute pairwise drag force between particles.

    Parameters
    ----------
    vel : numpy.ndarray
        Particle velocities, shape (N, D).
    c : numpy.ndarray
        Drag coefficients, shape (N, N).

    Returns
    -------
    numpy.ndarray
        Drag force vectors, shape (N, N, D).
    """
    diff_vel = vel[:, None, :] - vel[None, :, :]
    return -c[..., np.newaxis] * diff_vel

def lennard_jones(diff: np.ndarray, dist: np.ndarray, epsilon: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    Compute Lennard-Jones potential force.

    Parameters
    ----------
    diff : numpy.ndarray
        Displacement vectors, shape (N, N, D).
    dist : numpy.ndarray
        Distance magnitudes, shape (N, N).
    epsilon : numpy.ndarray
        Depth of potential well, shape (N, N).
    sigma : numpy.ndarray
        Finite distance where inter-particle potential is zero, shape (N, N).

    Returns
    -------
    numpy.ndarray
        Force vectors, shape (N, N, D).
    """
    direction = _safe_dir(diff, dist)
    safe_dist = np.maximum(dist, 1e-12)
    ratio_6 = (sigma / safe_dist)**6
    ratio_12 = ratio_6**2
    mag = 24 * epsilon * (2 * ratio_12 - ratio_6) / safe_dist
    return mag[..., np.newaxis] * direction

def evaluate_forces(pos: np.ndarray, vel: np.ndarray, force_graph: dict) -> np.ndarray:
    """
    Evaluate total forces on all particles.

    Parameters
    ----------
    pos : numpy.ndarray
        Particle positions, shape (N, D).
    vel : numpy.ndarray
        Particle velocities, shape (N, D).
    force_graph : dict
        Force interaction graph dictionary specifying active forces and parameters.

    Returns
    -------
    numpy.ndarray
        Total force vectors on each particle, shape (N, D).
    """
    n_particles = pos.shape[0]
    
    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    
    np.fill_diagonal(dist, np.inf)
    
    total_force = np.zeros_like(pos)
    
    with np.errstate(invalid='ignore', divide='ignore'):
        for force_name, params in force_graph.items():
            if force_name == "velocity_damper":
                total_force += velocity_damper(vel, **params)
            elif force_name == "pairwise_drag":
                f_ij = pairwise_drag(vel, **params)
                f_ij[np.eye(n_particles, dtype=bool)] = 0.0
                total_force += np.sum(f_ij, axis=1)
            else:
                if force_name == "linear_spring":
                    f_ij = linear_spring(diff, dist, **params)
                elif force_name == "duffing_oscillator":
                    f_ij = duffing_oscillator(diff, dist, **params)
                elif force_name == "newtonian_gravity":
                    f_ij = newtonian_gravity(diff, dist, **params)
                elif force_name == "coulomb":
                    f_ij = coulomb(diff, dist, **params)
                elif force_name == "yukawa":
                    f_ij = yukawa(diff, dist, **params)
                elif force_name == "morse_potential":
                    f_ij = morse(diff, dist, **params)
                elif force_name == "buckingham":
                    f_ij = buckingham(diff, dist, **params)
                elif force_name == "hertzian_contact":
                    f_ij = hertzian_contact(diff, dist, **params)
                elif force_name == "lennard_jones":
                    f_ij = lennard_jones(diff, dist, **params)
                else:
                    raise ValueError(f"Unknown force type in force_graph: {force_name}")
                    
                f_ij[np.eye(n_particles, dtype=bool)] = 0.0
                total_force += np.sum(f_ij, axis=1)
                
    return total_force
