"""
physics.py

Contains a function which calculates the unnormalised Maxwell-Boltzmann
speed distribution in 2D

Functions:
    - maxwell: Computes the unnormalised Maxwell–Boltzmann speed distribution in 2D.

This module is used to compare simulated speed distributions with theoretical
expectations, particularly in the context of equilibrium statistical mechanics.
"""
import numpy as np


def maxwell(speed, kbt, mass=1.):
    """
    Calculate the unnormalised Maxwell–Boltzmann speed distribution in 2D.

    This function returns the probability density function (PDF) for a particle
    of given mass and thermal energy (kBT), evaluated at a specified speed.
    The form used is appropriate for two-dimensional systems:
        f(v) ∝ v * exp(-mv² / 2kBT)

    Args:
        speed (float or array-like): Speed or array of speeds at which to evaluate the distribution.
        kbt (float): The product of Boltzmann’s constant and temperature (k_B * T).
        mass (float, optional): Particle mass. Defaults to 1.0.

    Returns:
        np.ndarray: Unnormalised PDF values at the given speed(s).
    """
    speed = np.array(speed)
    func = mass / (2 * kbt) * speed * \
        np.exp(- mass * speed * speed / (2 * kbt))
    return func
