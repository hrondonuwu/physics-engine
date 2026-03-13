import pytest
import numpy as np

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api import run_experiment
from engine.config import MIN_DIST, MAX_POS
from engine.world import STEPS
from engine.forces import evaluate_forces

def test_energy_conservation():
    """1. Symplectic Energy Conservation Test: Duffing Oscillator"""
    masses = np.array([1.0, 1.0])
    # Place them symmetrically with a small displacement from r0=1.0
    init_pos = np.array([[-0.6], [0.6]])
    # Initial velocity zero to prevent large kinetic energy crossing
    init_vel = np.array([[0.0], [0.0]])
    
    k_mat = np.array([[0.0, 1.0], [1.0, 0.0]])
    r0_mat = np.array([[0.0, 1.0], [1.0, 0.0]])
    alpha_mat = np.array([[0.0, 0.1], [0.1, 0.0]])
    
    force_graph = {
        "duffing_oscillator": {
            "k": k_mat,
            "r0": r0_mat,
            "alpha": alpha_mat
        }
    }
    
    valid, p, v, a = run_experiment(masses, init_pos, init_vel, force_graph, noise_level=0.0)
    
    assert valid, "Simulation should be valid for energy conservation test. Check coordinates/velocities."
    
    def calc_energy(step_idx):
        pos = p[step_idx]
        vel = v[step_idx]
        
        # Kinetic energy
        K = 0.5 * np.sum(masses[:, None] * vel**2)
        
        # Potential energy
        diff = pos[0] - pos[1]
        dist = np.linalg.norm(diff)
        
        k = k_mat[0, 1]
        r0 = r0_mat[0, 1]
        alpha = alpha_mat[0, 1]
        
        disp = dist - r0
        U = 0.5 * k * disp**2 + 0.25 * alpha * disp**4
        return K + U
        
    E_initial = calc_energy(0)
    E_final = calc_energy(-1)
    
    assert E_initial > 0, "Initial energy must be > 0 for valid relative error checking."
    
    error_pct = abs(E_initial - E_final) / E_initial
    assert error_pct < 1e-4, f"Energy conservation violated! Relative error {error_pct*100:.4f}% > 0.01%. E_initial: {E_initial:.6f}, E_final: {E_final:.6f}"

def test_singularity_trap():
    """2. The Singularity Trap: Newtonian Gravity"""
    masses = np.array([1.0, 1.0])
    
    # Place masses directly on top of each other (dist < MIN_DIST)
    init_pos = np.array([[0.0, 0.0], [MIN_DIST * 0.5, 0.0]])
    init_vel = np.array([[0.0, 0.0], [0.0, 0.0]])
    
    force_graph = {
        "newtonian_gravity": {
            "G": 1.0,
            "m_i": np.array([[1.0, 1.0], [1.0, 1.0]]),
            "m_j": np.array([[1.0, 1.0], [1.0, 1.0]])
        }
    }
    
    valid, _, _, _ = run_experiment(masses, init_pos, init_vel, force_graph, noise_level=0.0)
    
    assert not valid, f"Engine failed to catch division-by-zero! Distance < MIN_DIST={MIN_DIST} did not trigger INVALID."

def test_explosion_trap():
    """3. The Explosion Trap: Coulomb Repulsion"""
    masses = np.array([1.0, 1.0])
    
    # Place them moderately close
    init_pos = np.array([[-0.1, 0.0], [0.1, 0.0]])
    init_vel = np.array([[0.0, 0.0], [0.0, 0.0]])
    
    # Massive repulsive charges designed to blast particles past MAX_POS
    force_graph = {
        "coulomb": {
            "k_e": 1e9, 
            "q_i": np.array([[1.0, 1.0], [1.0, 1.0]]),
            "q_j": np.array([[1.0, 1.0], [1.0, 1.0]])
        }
    }
    
    valid, p, _, _ = run_experiment(masses, init_pos, init_vel, force_graph, noise_level=0.0)
    
    assert not valid, f"Engine failed to catch explosion! Huge repulsion did not trigger INVALID (exceed MAX_POS={MAX_POS})."

def test_sensor_accuracy_verification():
    """4. Sensor Accuracy Verification: Linear Spring"""
    masses = np.array([1.0, 1.0])
    init_pos = np.array([[-0.7], [0.7]])
    init_vel = np.array([[0.0], [0.0]])
    
    k_mat = np.array([[0.0, 1.0], [1.0, 0.0]])
    r0_mat = np.array([[0.0, 1.0], [1.0, 0.0]])
    
    force_graph = {
        "linear_spring": {
            "k": k_mat,
            "r0": r0_mat
        }
    }
    
    valid, noisy_pos, noisy_vel, sg_a = run_experiment(masses, init_pos, init_vel, force_graph, noise_level=0.001)
    
    assert valid, "Simulation mathematically exploded, which should not happen for simple linear spring."
    
    # Produce cleanly integrated truth 
    valid_clean, clean_pos, clean_vel, _ = run_experiment(masses, init_pos, init_vel, force_graph, noise_level=0.0)
    
    # Calculate analytical accelerations step-by-step
    true_accel = np.zeros_like(clean_pos)
    for i in range(STEPS):
        true_accel[i] = evaluate_forces(clean_pos[i], clean_vel[i], force_graph) / masses[:, None]
        
    # Exclude edges due to filter border artifacts
    margin = 50
    mae = np.mean(np.abs(sg_a[margin:-margin] - true_accel[margin:-margin]))
    
    assert mae < 0.2, f"Savitzky-Golay estimated accelerations wildly off from analytical truth. MAE: {mae}"
