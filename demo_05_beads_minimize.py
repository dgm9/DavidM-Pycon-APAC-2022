import numpy as np
import matplotlib.pyplot as plt
# from scipy.integrate import solve_ivp
from scipy.optimize import minimize

def get_energy(x, a, L_total, omega, alpha):
    E_periodic = alpha * np.sin(omega * x)
    # just the springs, no masses
    E_springs = (np.mod(np.roll(x, -1) - x, L_total) - a)**2
    return (E_periodic + E_springs).sum()


N = 30
a = 1.0  # atomic spacing
L_total = N * a


# periodic potential

omega_0 = 2 * np.pi / a
omega = (10 / N) * omega_0

alpha = 1.4 # strength of periodic potential


x_nominal = a * np.arange(N, dtype=float)

x0 = x_nominal.copy()   # actual starting positions

# x0[12] += 0.8    # displace one

E_now = get_energy(x0, a, L_total, omega, alpha)

print('E_now: ', E_now)

args = (a, L_total, omega, alpha)

tol = 0.01

wow = minimize(get_energy, x0=x0, args=args, tol=tol)

x_final = wow.x

E_final = get_energy(x_final, a, L_total, omega, alpha)

print('E_final: ', E_final)

if True:
    fig, ax = plt.subplots(1, 1, figsize=[5, 7.5])
    things = np.vstack((x0, x_final)).T
    for thing in things:  
        ax.plot(thing)
    plt.subplots_adjust(bottom=0.05, top=0.95)
    for x in x_nominal:
        ax.plot([0], [x], '.')
    for x in x_nominal:
        ax.plot([1], [x], '.')
    if True:
        y = np.linspace(0, 30, 601)
        x = -0.05 * alpha * np.sin(omega * y)
        zero = np.zeros_like(x)
        ax.plot(x - 0.05, y, '-k')
        ax.plot(x + 1.05, y, '-k')
        ax.plot(zero - 0.05, y, '--k', linewidth=0.5)
        ax.plot(zero + 1.05, y, '--k', linewidth=0.5)      
    plt.show()
