import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Parametri dominio rettangolare ---
Lx = 4.0
Ly = 1.0
NX = 81
NY = 21
dx = Lx / (NX - 1)
dy = Ly / (NY - 1)

# --- Parametri fisici ---
NU = 0.1
dt = 0.001
N_ITER = 500
U_TOP = 1.0

# --- Inizializzazione ---
x = np.linspace(0, Lx, NX)
y = np.linspace(0, Ly, NY)
X, Y = np.meshgrid(x, y)

psi = np.zeros((NY, NX))
zeta = np.zeros_like(psi)

# --- Funzioni ausiliarie ---
def laplacian(f):
    return (
        (f[1:-1, 0:-2] - 2*f[1:-1, 1:-1] + f[1:-1, 2:]) / dx**2 +
        (f[0:-2, 1:-1] - 2*f[1:-1, 1:-1] + f[2:, 1:-1]) / dy**2
    )

def compute_velocity(psi):
    u = np.zeros_like(psi)
    v = np.zeros_like(psi)
    u[1:-1, 1:-1] = (psi[2:, 1:-1] - psi[0:-2, 1:-1]) / (2*dy)
    v[1:-1, 1:-1] = -(psi[1:-1, 2:] - psi[1:-1, 0:-2]) / (2*dx)
    return u, v

# --- Preparazione figura ---
fig, ax = plt.subplots(figsize=(8,2.5))
contour = ax.contourf(X, Y, psi, 50, cmap='coolwarm')
quiver = ax.quiver(X[::2,::2], Y[::2,::2], np.zeros_like(X[::2,::2]), np.zeros_like(Y[::2,::2]), color='black')
ax.set_xlim(0, Lx)
ax.set_ylim(0, Ly)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Lid-driven cavity (ψ-ζ) - rettangolare")
ax.set_aspect('auto')
fig.colorbar(contour, ax=ax, label='Streamfunction ψ')

# --- Funzione di aggiornamento ---
def update(frame):
    global psi, zeta
    
    # --- Calcolo velocità ---
    u, v = compute_velocity(psi)
    
    # --- Aggiornamento vorticità (esplicito) ---
    zeta_new = np.copy(zeta)
    zeta_new[1:-1,1:-1] = (
        zeta[1:-1,1:-1] +
        dt * (
            - u[1:-1,1:-1]*(zeta[1:-1,2:] - zeta[1:-1,0:-2])/(2*dx)
            - v[1:-1,1:-1]*(zeta[2:,1:-1] - zeta[0:-2,1:-1])/(2*dy)
            + NU * laplacian(zeta)
        )
    )
    
    # Condizioni al contorno vorticità (no-slip)
    zeta_new[0, :]    = -2*psi[0, :] / dy**2
    zeta_new[-1, :]   = -2*(psi[-1, :] - U_TOP*dy) / dy**2
    zeta_new[:, 0]    = -2*psi[:, 0] / dx**2
    zeta_new[:, -1]   = -2*psi[:, -1] / dx**2
    
    zeta = zeta_new
    
    # --- Risolvo Poisson per psi con Gauss-Seidel ---
    psi_new = np.copy(psi)
    for _ in range(20):  # meno iterazioni per velocità
        psi_new[1:-1,1:-1] = 0.25 * (
            psi_new[1:-1,0:-2] + psi_new[1:-1,2:] +
            psi_new[0:-2,1:-1] + psi_new[2:,1:-1] +
            dx*dy * zeta[1:-1,1:-1]
        )
        # Condizioni al contorno psi=0
        psi_new[0,:] = 0.0
        psi_new[-1,:] = 0.0
        psi_new[:,0] = 0.0
        psi_new[:,-1] = 0.0
        
    psi = psi_new
    
    # --- Aggiorno plot ---
    for c in ax.collections:
        c.remove()
    contour = ax.contourf(X, Y, psi, 50, cmap='coolwarm')
    
    u, v = compute_velocity(psi)
    quiver.set_UVC(u[::2,::2], v[::2,::2])
    
    return contour, quiver

# --- Animazione ---
ani = FuncAnimation(fig, update, frames=N_ITER, interval=50, blit=False)
plt.show()



