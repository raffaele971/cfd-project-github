import numpy as np
import matplotlib.pyplot as plt

# 1D Linear Advection Equation with Flux Limiters (TVD Schemes)
#
# This code solves the one-dimensional linear advection equation
#
#     ∂φ/∂t + a ∂φ/∂x = 0
#
# on a periodic domain using an explicit finite-volume formulation.
# The spatial discretization is based on
# second-order accurate Total Variation Diminishing (TVD)
# schemes with flux limiter functions.
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
# Numerical update:
#
#   ϕᵢⁿ⁺¹ = ϕᵢⁿ − (Δt / Δx) ( Fᵢ₊₁⧸2 − Fᵢ₋₁⧸2 )
#
# with numerical flux:
#
#   Fᵢ₊₁⧸₂ = a ϕᵢ + (a / 2) (1 − c) ψᵢ₊₁⧸₂ ( ϕᵢ₊₁ − ϕᵢ )
#
# This is a conservative discretization.
# The solution represents cell averages and is updated
# by balancing the fluxes entering and leaving each cell.
# Discrete conservation is guaranteed by construction.

# Flux limiter (psi) functions

def psi_fun(theta, limiter_type):
    """
    Flux limiter functions for TVD schemes
    """
    if limiter_type == 1:      # First-order upwind
        return np.zeros_like(theta)
    elif limiter_type == 2:    # Lax-Wendroff
        return np.ones_like(theta)
    elif limiter_type == 3:    # Beam-Warming
        return theta
    elif limiter_type == 4:    # Van Leer
        return (theta + np.abs(theta)) / (1 + np.abs(theta))



# Problem parameters

L = 1.0            # Domain length
N = 60             # Number of control volumes
a = 1.0            # Advection velocity
cfl = 0.3          # CFL number

dx = L / N
x = np.linspace(0, L, N, endpoint=False) + 0.5 * dx  # Cell centers

dt = cfl * dx / abs(a)
T = L / abs(a)
Nt = int(T / dt)
dt = T / Nt
c = a * dt / dx

eps = 1e-12        # Small number to avoid division by zero

# Initial condition (cell averages)
phi0 = np.exp(-100 * (x - L / 2)**2)


# PART 1: Time evolution for different limiters

for limiter in range(1, 5):

    phi = phi0.copy()

    for it in range(Nt):

        # Periodic indexing
        phi_ip1 = np.roll(phi, -1)
        phi_im1 = np.roll(phi,  1)
        phi_im2 = np.roll(phi,  2)

        # Slopes
        dphi_plus  = phi_ip1 - phi
        dphi       = phi - phi_im1
        dphi_minus = phi_im1 - phi_im2

        # Theta ratios
        theta   = dphi_minus / (dphi + eps)
        theta_p = dphi / (dphi_plus + eps)

        # Flux limiters
        psi     = psi_fun(theta, limiter)
        psi_p   = psi_fun(theta_p, limiter)

        
        # Numerical fluxes at interfaces (a > 0 upwinding)
       
        F_iphalf = (
            a * phi
            + 0.5 * a * (1 - c) * psi_p * dphi_plus
        )

        F_imhalf = (
            a * phi_im1
            + 0.5 * a * (1 - c) * psi * dphi
        )

        # Finite volume update
        phi = phi - (dt / dx) * (F_iphalf - F_imhalf)

        # Visualization
        plt.clf()
        plt.plot(x, phi0, 'r--', label='Initial condition')
        plt.plot(x, phi, 'k', label='Numerical solution')
        plt.title(f'Finite Volume – limiter type {limiter}')
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

limiter = 4   # Van Leer limiter

for N in range(Nmin, Nmax + 1, Nstep):

    dx = L / N
    x = np.linspace(0, L, N, endpoint=False) + 0.5 * dx

    dt = cfl * dx / abs(a)
    Nt = int(T / dt)
    dt = T / Nt
    c = a * dt / dx

    phi0 = np.exp(-100 * (x - L / 2)**2)
    phi = phi0.copy()

    for it in range(Nt):

        phi_ip1 = np.roll(phi, -1)
        phi_im1 = np.roll(phi,  1)
        phi_im2 = np.roll(phi,  2)

        dphi_plus  = phi_ip1 - phi
        dphi       = phi - phi_im1
        dphi_minus = phi_im1 - phi_im2

        theta   = dphi_minus / (dphi + eps)
        theta_p = dphi / (dphi_plus + eps)

        psi     = psi_fun(theta, limiter)
        psi_p   = psi_fun(theta_p, limiter)

        F_iphalf = a * phi + 0.5 * a * (1 - c) * psi_p * dphi_plus
        F_imhalf = a * phi_im1 + 0.5 * a * (1 - c) * psi * dphi

        phi = phi - (dt / dx) * (F_iphalf - F_imhalf)

    error.append(np.max(np.abs(phi - phi0)))
    Nvec.append(N)

# Plot convergence
plt.figure()
plt.plot(Nvec, error, 'o-k')
plt.xlabel('Number of cells N')
plt.ylabel('Maximum error')
plt.title('Grid convergence – Finite Volume (Van Leer)')
plt.grid(True)
plt.show()
