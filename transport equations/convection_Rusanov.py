import numpy as np
import matplotlib.pyplot as plt


def circulant(v):
    v = np.asarray(v)
    n = len(v)
    C = np.zeros((n, n))
    for i in range(n):
        C[i, :] = np.roll(v, i)
    return C


# solutor for transport equation

def solve_transport(N, c=1.0, w=1.0, plot=False):
    L = 2 * np.pi
    a = 1.0

    # Grid
    x = np.linspace(0, L, N)
    h = x[1] - x[0]

    # initial condition
    phi0 = np.exp(-10 * (x - np.pi) ** 2)

    dt = c * h / a
    T = L / abs(a)
    Nt = round(T / dt)
    dt = T / Nt
    c = a * dt / h

  
    I = np.eye(N - 1)

    
    v = np.zeros(N - 1)
    v[1] = 1
    A = circulant(v)

    v = np.zeros(N - 1)
    v[0] = 0.5 + c / 3
    v[1] = 0.5 - c / 3
    B = circulant(v)

    Cmat = np.linalg.solve(A, B)

    
    # differential operators
    
    v = np.zeros(N - 1)
    v[-1] = -1
    v[1] = 1
    D1 = circulant(v)
    D2 = D1.copy()

    v = np.zeros(N - 1)
    v[-2] = -1
    v[2] = 1
    D4 = circulant(v)

    v = np.zeros(N - 1)
    v[-1] = 1
    v[1] = 1
    Mu2 = circulant(v)

    v = np.zeros(N - 1)
    v[-2] = 1
    v[2] = 1
    Mu4 = circulant(v)

    Delta4 = Mu4 - 4 * Mu2 + 6 * I

   
    phi = phi0[:-1].copy()

    # ------------------------------------------------------
    # Time marching
    # ------------------------------------------------------
    for _ in range(Nt):
        phi1 = Cmat @ phi
        phi2 = phi - (2 * c / 3) * (D1 @ phi1)

        phi = ((I - (c / 24) * (7 * D2 - 2 * D4)
                - (w / 24) * Delta4) @ phi
               - (3 * c / 8) * (D2 @ phi2))

        if plot:
            plt.clf()
            plt.plot(x, phi0, 'r--', label='iniziale')
            plt.plot(x, np.r_[phi, phi[0]], 'k', label='numerica')
            plt.axis([0, L, -0.1, 1.1 * np.max(phi0)])
            plt.legend()
            plt.pause(0.01)

    return phi, phi0, x


    # principal esecution

if __name__ == "__main__":

    N = 100
    c = 1.0
    w = (4 * c**2 + 1) * (4 - c**2) / 5  

    phi, phi0, x = solve_transport(N, c, w, plot=True)

    #  CONVERGENCE STUDY
    Nmin, Nmax, Nstep = 20, 300, 10
    Nvec = []
    err = []

    for N in range(Nmin, Nmax + 1, Nstep):
        phi, phi0, _ = solve_transport(N, c=1.0, w=1.0)
        err.append(np.max(np.abs(phi - phi0[:-1])))
        Nvec.append(N)

    plt.figure()
    plt.loglog(Nvec, err, 'o-')
    plt.xlabel('numbers of point')
    plt.ylabel('max error')
    plt.grid(True)
    plt.show()



