import numpy as np
import matplotlib.pyplot as plt

# ================================================================
# 1D Linear Advection Equation with Limiters 
#
# This code solves the one-dimensional linear advection equation
#
#     ∂φ/∂t + a ∂φ/∂x = 0
#
# on a periodic domain using an explicit finite-difference 
# The spatial discretization is based on second-order accurate
# Total Variation Diminishing(TVD)schemes with flux limiter functions.
#
# Four different limiter functions are implemented:
#   1) First-order upwind scheme
#   2) Lax–Wendroff scheme
#   3) Beam–Warming scheme
#   4) Van Leer flux limiter
#
# Time integration is performed explicitly under a CFL stability
# condition. The initial condition is a Gaussian pulse, which is
# advected across the domain over one full period.
#
# The code is divided into two main parts:
#   - Part 1: time evolution and real-time visualization of the
#     numerical solution for each limiter type
#   - Part 2: grid convergence study using the maximum norm of the
#     error after one advection period
#
# Periodic boundary conditions are enforced using array indexing
# and circular shifts. A small regularization parameter is used
# to avoid division by zero in the limiter evaluation.

# The numerical scheme is written in differential form.
# The solution is updated directly using finite differences
# that approximate the spatial derivative.
#
# Typical update:
#
#   ϕᵢⁿ⁺¹ = ϕᵢⁿ − c (ϕᵢ − ϕᵢ₋₁) − (c (1 − c) / 2) ( ψᵢ₊₁⧸₂ Δ⁺ϕᵢ − ψᵢ₋₁⧸₂ Δ⁻ϕᵢ )
#
#      where Δ⁺ϕᵢ = ϕᵢ₊₁ − ϕᵢ  and  Δ⁻ϕᵢ = ϕᵢ − ϕᵢ₋₁.
#
# where Delta^+ and Delta^- are forward and backward differences.

# Flux limiter (psi) functions

def psi_fun(theta, limiter_type):
    """
    Flux limiter functions used in TVD schemes

    limiter_type:
        1 -> First-order upwind
        2 -> Lax-Wendroff
        3 -> Beam-Warming
        4 -> Van Leer limiter
    """
    if limiter_type == 1:
        return np.zeros_like(theta)
    elif limiter_type == 2:
        return np.ones_like(theta)
    elif limiter_type == 3:
        return theta
    elif limiter_type == 4:
        return (theta + np.abs(theta)) / (1 + np.abs(theta))



# PART 1: Time evolution and visualization


# Domain and discretization
L = 1.0              # Domain length
N = 60               # Number of grid points
a = 1.0              # Advection velocity
c = 0.3              # CFL number

x = np.linspace(0, L, N, endpoint=False)
dx = L / N

# Time step from CFL condition
dt = c * dx / abs(a)

# Final time (one full advection period)
T = L / abs(a)

# Number of time steps
Nt = int(T / dt)
dt = T / Nt
c = a * dt / dx

# Small number to avoid division by zero
eps = 1e-12

# Initial condition: Gaussian pulse
phi0 = np.exp(-100 * (x - L / 2)**2)



# Loop over limiter types

for limiter in range(1, 5):

    # Reset solution
    phi = phi0.copy()

    for it in range(Nt):

        # Periodic index shifts
        phi_p  = np.roll(phi, -1)   # i+1
        phi_m  = np.roll(phi,  1)   # i-1
        phi_mm = np.roll(phi,  2)   # i-2

        # Finite differences
        dphi_plus  = phi_p - phi
        dphi       = phi - phi_m
        dphi_minus = phi_m - phi_mm

        # Ratio of consecutive gradients
        theta   = dphi_minus / (dphi + eps)
        theta_p = dphi / (dphi_plus + eps)

        # Limiter functions
        psi     = psi_fun(theta, limiter)
        psi_p   = psi_fun(theta_p, limiter)

        # TVD update formula
        phi = (
            phi
            - c * dphi
            - 0.5 * c * (1 - c)
            * (psi_p * dphi_plus - psi * dphi)
        )

        # Visualization 
        plt.clf()
        plt.plot(x, phi0, 'r--', label='Initial condition')
        plt.plot(x, phi, 'k', label='Numerical solution')
        plt.title(f'Flux limiter type {limiter}')
        plt.axis([0, L, -0.2, 1.2])
        plt.legend()
        plt.pause(0.01)

plt.show()



# PART 2: Grid convergence study

Nmin = 100
Nmax = 300
Nstep = 50

error = []
Nvec = []

# Use Van Leer limiter (best TVD behavior)
limiter = 4

for N in range(Nmin, Nmax + 1, Nstep):

    x = np.linspace(0, L, N, endpoint=False)
    dx = L / N

    dt = c * dx / abs(a)
    T = L / abs(a)
    Nt = int(T / dt)
    dt = T / Nt
    c = a * dt / dx

    phi0 = np.exp(-100 * (x - L / 2)**2)
    phi = phi0.copy()

    for it in range(Nt):

        phi_p  = np.roll(phi, -1)
        phi_m  = np.roll(phi,  1)
        phi_mm = np.roll(phi,  2)

        dphi_plus  = phi_p - phi
        dphi       = phi - phi_m
        dphi_minus = phi_m - phi_mm

        theta   = dphi_minus / (dphi + eps)
        theta_p = dphi / (dphi_plus + eps)

        psi     = psi_fun(theta, limiter)
        psi_p   = psi_fun(theta_p, limiter)

        phi = (
            phi
            - c * dphi
            - 0.5 * c * (1 - c)
            * (psi_p * dphi_plus - psi * dphi)
        )

    # Maximum norm error
    error.append(np.max(np.abs(phi - phi0)))
    Nvec.append(N)

# Plot convergence results
plt.figure()
plt.plot(Nvec, error, 'o-k')
plt.xlabel('Number of grid points N')
plt.ylabel('Maximum error')
plt.title('Grid convergence study (Van Leer limiter)')
plt.grid(True)
plt.show()

