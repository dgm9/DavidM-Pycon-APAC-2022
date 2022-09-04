import numpy as np
import matplotlib.pyplot as plt
# from scipy.integrate import solve_ivp

def get_energy(x, a, L_total, omega, alpha):
    E_periodic = alpha * np.sin(omega * x)
    # just the springs, no masses
    E_springs = (np.mod(np.roll(x, -1) - x, L_total) - a)**2
    return E_periodic.sum(), E_springs.sum()


N = 30
a = 1.0
L_total = N * a

N_temperatures = 40
T_max = 0.1 
N_per_temperature = 1000
i_seed = 42

jiggle_start = 0.005

np.random.seed(i_seed)

Temperatures = T_max * np.linspace(1, 0, N_temperatures+1)[:-1]

jiggles = jiggle_start * np.linspace(1, 0, N_temperatures+1)[:-1]

# periodic potential

omega_0 = 2 * np.pi / a
omega = (10 / N) * omega_0

alpha = 1.4 # strength of periodic potential


x_nominal = a * np.arange(N, dtype=float)

x0 = x_nominal.copy()   # actual starting positions

# x0[16] += 0.8    # displace one


# Get initial energy
E_periodic_now, E_springs_now = get_energy(x0, a, L_total, omega, alpha)

E_now = E_periodic_now + E_springs_now

x = x0.copy()   
kept_numbers = []
x_all = [x.copy()]
x_steps = [x.copy()]
energies = []
jiggle = jiggles[0]
for T, jiggle in zip(Temperatures, jiggles):
    kept_low = 0
    kept_high = 0
    for i in range(N_per_temperature):
        x_try = x + jiggle * 2 * (np.random.random(N) - 0.5)
        E_periodic_try, E_springs_try = get_energy(x_try, a, L_total, omega, alpha)
        E_try = E_periodic_try + E_springs_try
        dE = E_try - E_now
        if dE <= 0: # always keep it!
            x = x_try
            E_now = E_try
            kept_low += 1
        else:  # sometimes keep it
            if np.exp(-dE/T) >= np.random.random():  ### IF! keep it sometimes
                x = x_try
                E_now = E_try
                kept_high += 1
            else:
                pass # other times throw it out, don't use it
        x_all.append(x.copy())
    kept_numbers.append([kept_low, kept_high])
    x_steps.append(x.copy())
    energies.append([E_periodic_try, E_springs_try])

kept_low, kept_high = np.array(kept_numbers).T

if True:
    plt.plot(kept_low, '-')
    plt.plot(kept_high, '--')
    plt.show()

if True:
     A, B = np.array(energies).T
     plt.plot(A, '-')
     plt.plot(B, '--')
     plt.show()

x_steps = np.array(x_steps).T

if False:
    fig, ax = plt.subplots(1, 1, figsize=[9, 6.5])
    for thing in x_steps:  
        ax.plot(thing)
    plt.show()

if True:
    fig, ax = plt.subplots(1, 1, figsize=[5, 7.5])
    # Temps_plot = np.hstack((Temperatures[:1], Temperatures))
    for thing in x_steps:  
        ax.plot(Temperatures, thing[1:])
    plt.subplots_adjust(bottom=0.05, top=0.95)
    ax.invert_xaxis()
    if True:
        for x in x_nominal:
            ax.plot([Temperatures.min()], [x], '.')
        for x in x_nominal:
            ax.plot([Temperatures.max()], [x], '.')
        if True:
            y = np.linspace(0, 30, 601)
            x = +0.002 * alpha * np.sin(omega * y) # + since axis reversed
            zero = np.zeros_like(x)
            ax.plot(x + Temperatures.min() - 0.01 , y, '-k')
            ax.plot(x + Temperatures.max() + 0.01, y, '-k')
            ax.plot(zero + Temperatures.min() - 0.01, y, '--k', linewidth=0.5)
            ax.plot(zero + Temperatures.max() + 0.01, y, '--k', linewidth=0.5)      
    plt.show()
