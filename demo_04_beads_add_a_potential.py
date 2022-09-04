import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def deriv_periodic_and_potential(t, y, masses, damping, t_start, L_total,
                                 omega, alpha):
    x, v = y.reshape(2, -1)
    F_periodic = alpha * np.cos(omega * x)
    # periodic BC
    a_p1 = np.mod(np.roll(x, -1) - x, L_total)
    a_m1 = np.mod(x - np.roll(x, 1), L_total)
    acc = (a_p1 - a_m1) / masses - damping * v + F_periodic
    acc *= (t >= t_start)
    return np.hstack((v, acc))

N = 30 
a = 1.0
L_total = N * a
method = 'DOP853'
dense_output = False
m = 1.0
damping = 0.5 ### 0.1
total_time = 100.
nsteps = 501
t_start = 0.05 * total_time

# periodic potential

omega_0 = 2 * np.pi / a
omega = (5 / N) * omega_0

alpha = 0.6 # strength of periodic force

x_nominal = a * np.arange(N, dtype=float)

x0 = x_nominal.copy()   # actual starting positions

x0[12] += 0.8    # displace one

v0 = np.zeros_like(x0)

y0 = np.hstack((x0, v0))

masses = m * np.ones_like(x0)

# masses[32] *= 5.  # add a defect


t_eval = np.linspace(0, total_time, nsteps)
t_span = t_eval.min(), t_eval.max()

args = (masses, damping, t_start, L_total, omega, alpha)

answer = solve_ivp(fun=deriv_periodic_and_potential, t_span=t_span, y0=y0,
                   method=method, t_eval=t_eval, dense_output=dense_output,
                   events=None, vectorized=False, args=args)

displacements = x_nominal[:, None] + 2 * (answer.y[:N] - x_nominal[:, None])

if True:
    fig, ax = plt.subplots(1, 1, figsize=[5, 7.5])
    for thing in displacements:  # answer.y[:N]:
        ax.plot(t_eval, thing)
    plt.subplots_adjust(bottom=0.05, top=0.95)
    ax.plot(np.zeros(N), x_nominal, '.k')
    ax.plot(np.zeros(N) + t_eval.max(), x_nominal, '.k')
    if True:
        y = np.linspace(0, 30, 601)
        x = -5 * alpha * np.cos(omega * y)
        zero = np.zeros_like(x)
        ax.plot(x + t_eval.min() - 5, y, '-k')
        ax.plot(x + t_eval.max() + 5, y, '-k')
        ax.plot(zero + t_eval.min() - 5, y, '--k', linewidth=0.5)
        ax.plot(zero + t_eval.max() + 5, y, '--k', linewidth=0.5)      
    plt.show()


"""
# preallocate empty array and assign slice by chrisaycock
def shift5(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result
"""
