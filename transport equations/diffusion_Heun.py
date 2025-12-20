import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# global paramteres
k = 1.0
a = 1.0

    #Functions

def esfun_b(phi):
    DphiDt = np.zeros(2)
    DphiDt[0] = phi[1]
    DphiDt[1] = (a / k) * phi[1]
    return DphiDt


def heun(x, phi0):
    N = len(x)
    h = x[1] - x[0]

    PHI = np.zeros((N, 2))
    PHI[0, :] = phi0

    phi = phi0.copy()

    for ix in range(1, N):
        F1 = esfun_b(phi)
        phi2 = phi + h * F1 / 3.0

        F2 = esfun_b(phi2)
        phi3 = phi + 2.0 * h * F2 / 3.0

        F3 = esfun_b(phi3)

        phi = phi + h * ((1.0 / 4.0) * F1 + (3.0 / 4.0) * F3)
        PHI[ix, :] = phi

    return PHI



# Code

L = 1.0
N = 50
x = np.linspace(0, L, N)

dPhi0 = 1.0
PhiL = 0.0

# Problem 1
Phi0A = np.array([1.0, dPhi0])
PHIA = heun(x, Phi0A)
PhiLA = PHIA[-1, 0]

# Problem 2
Phi0B = np.array([2.0, dPhi0])
PHIB = heun(x, Phi0B)
PhiLB = PHIB[-1, 0]

C2 = (PhiL - PhiLA) / (PhiLB - PhiLA)
C1 = 1.0 - C2

PHI = C1 * PHIA + C2 * PHIB

plt.plot(x, PHI[:, 0], 'k.-')
plt.xlabel('x')
plt.ylabel(r'$\phi(x)$')
plt.title('Solution')
plt.grid(True)
plt.show()
