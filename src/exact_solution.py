# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import scipy.integrate as integrate
import os

def cole_hopf_exact_point(x, t, nu=0.01):
    """
    Computes exact analytical solution of 1D Viscous Burgers' Equation:
    u_t + u * u_x = nu * u_xx
    Initial Condition: u(x, 0) = sin(pi * x) for x in [0, 1]
    Boundary Condition: u(0, t) = u(1, t) = 0
    Using Cole-Hopf Transformation with numerical quadrature.
    """
    if t == 0.0:
        return np.sin(np.pi * x)
    
    # Boundary enforcement
    if np.isclose(x, 0.0) or np.isclose(x, 1.0):
        return 0.0

    # Integration bounds around x where kernel is significant
    sigma = np.sqrt(2 * nu * t)
    eta_min = max(-2.0, x - 10 * sigma)
    eta_max = min(3.0, x + 10 * sigma)

    def f_num(eta):
        psi = (1.0 - np.cos(np.pi * eta)) / (np.pi)
        exponent = -((x - eta) ** 2) / (4.0 * nu * t) - psi / (2.0 * nu)
        return ((x - eta) / t) * np.exp(exponent)

    def f_den(eta):
        psi = (1.0 - np.cos(np.pi * eta)) / (np.pi)
        exponent = -((x - eta) ** 2) / (4.0 * nu * t) - psi / (2.0 * nu)
        return np.exp(exponent)

    num, _ = integrate.quad(f_num, eta_min, eta_max, limit=100)
    den, _ = integrate.quad(f_den, eta_min, eta_max, limit=100)

    if den == 0.0:
        return 0.0

    return num / den

def compute_exact_grid(x_flat, t_flat, nu=0.01):
    """
    Computes exact solution over a 2D meshgrid defined by x_flat and t_flat.
    Returns array u_exact of shape (len(t_flat), len(x_flat)).
    """
    Nx = len(x_flat)
    Nt = len(t_flat)
    u_exact = np.zeros((Nt, Nx))

    for i, t in enumerate(t_flat):
        for j, x in enumerate(x_flat):
            u_exact[i, j] = cole_hopf_exact_point(x, t, nu=nu)

    return u_exact

if __name__ == '__main__':
    x_flat = np.linspace(0.0, 1.0, 256)
    t_flat = np.linspace(0.0, 1.0, 100)
    print(f"Calculating Cole-Hopf exact reference solution on grid {len(x_flat)}x{len(t_flat)}...")
    u_exact = compute_exact_grid(x_flat, t_flat, nu=0.01)
    
    os.makedirs('evaluation', exist_ok=True)
    np.save('evaluation/exact_solution.npy', u_exact)
    np.save('evaluation/x_grid.npy', x_flat)
    np.save('evaluation/t_grid.npy', t_flat)

    print("Successfully saved Cole-Hopf exact solution to evaluation/exact_solution.npy")
    print(f"Shape: {u_exact.shape}, Min: {u_exact.min():.5f}, Max: {u_exact.max():.5f}")
