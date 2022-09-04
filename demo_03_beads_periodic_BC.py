import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def deriv(t, y, masses, damping, t_start):  ### NOT USED
    x, v = y.reshape(2, -1)
    # calculate acceleration on all atoms except the endpoints
    acc = np.zeros_like(x)
    # force from right: (x[2:] - x[1:-1] - a)
    # force from left: -(x[1:-1] - x[2:] - a)
    acc[1:-1] = (x[2:] + x[:-2] - 2 * x[1:-1]) / masses [1:-1]
    acc[1:-1] += -damping * v[1:-1]
    acc *= (t >= t_start)
    return np.hstack((v, acc))

def deriv_periodic(t, y, masses, damping, t_start, L_total):   ### USING THIS!
    x, v = y.reshape(2, -1)
    ### periodic BC
    a_p1 = np.mod(np.roll(x, -1) - x, L_total)
    a_m1 = np.mod(x - np.roll(x, 1), L_total)
    acc = (a_p1 - a_m1) / masses - damping * v
    acc *= (t >= t_start)
    return np.hstack((v, acc))

N = 30
a = 1.0
L_total = N * a ###

method = 'DOP853'
dense_output = False
m = 1.0
damping = 5.5  ### 0.05   # explore critical and over- damping

x_nominal = a * np.arange(N, dtype=float)

x0 = x_nominal.copy()
x0[10] += 0.8

v0 = np.zeros_like(x0)

y0 = np.hstack((x0, v0))

masses = m * np.ones_like(x0)
# masses[20] *= 5.

t_eval = np.linspace(0, 100, 501)
t_span = t_eval.min(), t_eval.max()
t_start = 20.

args = (masses, damping, t_start, L_total)

### fun=deriv_periodic
answer = solve_ivp(fun=deriv_periodic, t_span=t_span, y0=y0, method=method,
                   t_eval=t_eval, dense_output=dense_output, events=None,
                   vectorized=False, args=args)

displacements = x_nominal[:, None] + 2 * (answer.y[:N] - x_nominal[:, None])

if True:
    fig, ax = plt.subplots(1, 1, figsize=[5, 7.5])
    for thing in displacements:  # answer.y[:N]:
        ax.plot(t_eval, thing)
    plt.subplots_adjust(bottom=0.05, top=0.95)
    ax.plot(np.zeros(N), x_nominal, '.k')
    ax.plot(np.zeros(N) + t_eval.max(), x_nominal, '.k')
    plt.show()
