import numpy as np
import matplotlib.pyplot as plt

# =========================
# PARAMETERS
# =========================
L = 1.0          # Half-width of domain [-L, L]
a = 1.0          # Convection velocity

# -------------------------
# FUNCTION: Lax-Wendroff update with periodic BC
# -------------------------
def lax_wendroff(Phi, a, dt, h):
    """Perform one time step of the Lax–Wendroff scheme with periodic BC."""
    lam = dt / h
    Phi_roll = np.roll(Phi, -1)   # Phi_{i+1} with wrap-around
    Phi_m1 = np.roll(Phi, 1)      # Phi_{i-1} with wrap-around

    # Compute fluxes
    # The Lax–Wendroff scheme for 1D linear convection can be written as:
    # F_{i+1/2} = a*Phi_i + 0.5 * a*(1 - a*lambda)*(Phi_{i+1} - Phi_i)
    # Phi^{n+1}_i = Phi^n_i - lambda * (F_{i+1/2} - F_{i-1/2})
    # Here, lambda = Dt/h and F_{i+1/2} is the flux between nodes i and i+1
    
    F = a * Phi + 0.5 * a * (1 - a * lam) * (Phi_roll - Phi)
    Fm = a * Phi_m1 + 0.5 * a * (1 - a * lam) * (Phi - Phi_m1)

    # Update solution
    Phi_new = Phi - lam * (F - Fm)
    return Phi_new

# =========================
# POINT 1: Unsteady Convection
# =========================
N = 71
x = np.linspace(-L, L, N)[:-1]  # Periodic BC: remove last point
h = x[1] - x[0]

# Initial condition: triangular pulse
Phi0 = np.maximum(0, L/3 - np.abs(x))
Phi = Phi0.copy()

# Time parameters
C = 0.7
Dt = C * h / a
Nt = round((2*L / a) / Dt)       # Final time = one domain traversal
Dt = (2*L / a) / Nt             # Adjust Dt
C = a * Dt / h

# Plot initial condition
plt.ion()
fig, ax = plt.subplots()
for t in range(Nt):
    Phi = lax_wendroff(Phi, a, Dt, h)

    # Plot
    ax.clear()
    ax.plot(x, Phi, '.-k', label='Solution')
    ax.plot(x, Phi0, '.-r', label='Initial Condition')
    ax.set_title(f'Unsteady Convection (Courant={C:.2f})')
    ax.set_xlim([-L, L])
    ax.set_ylim([-(L/3), 2*L/3])
    ax.legend()
    plt.pause(0.01)
plt.ioff()
plt.show()

# =========================
# POINT 2: Transition Matrix Norm vs Courant Number
# =========================
N = 21
x = np.linspace(-L, L, N)[:-1]
h = x[1] - x[0]

NC = 51
CVec = np.linspace(0, 2, NC)
NormaT = np.zeros(NC)
I = np.eye(N-1)

for iC, C in enumerate(CVec):
    dt = C * h / a
    lam = dt / h
    T = np.zeros((N-1, N-1))
    for ii in range(N-1):
        Phi = I[:, ii]
        PhiN = lax_wendroff(Phi, a, dt, h)
        T[:, ii] = PhiN
    NormaT[iC] = np.linalg.norm(T)

# Plot norm vs Courant number
plt.figure()
plt.plot(CVec, NormaT, 'k.-')
plt.grid(True)
plt.title('Norm of the Transition Matrix')
plt.xlabel('Courant Number')
plt.ylabel('||T||')
plt.show()

