import numpy as np
import matplotlib.pyplot as plt

# we need both Energy and Force (dEnergy/dx):

def Force(x, L_total, omega, alpha):
    F_periodic = -alpha * np.cos(omega * x) # F = -grad(E)
    F_p1 = np.mod(np.roll(x, -1) - x, L_total)
    F_m1 = np.mod(x - np.roll(x, 1), L_total)
    F = (F_p1 - F_m1) + F_periodic
    return F

def Energy(x, a, L_total, omega, alpha):
    E_periodic = alpha * np.sin(omega * x) / omega
    # just the springs, no masses
    E_springs = 0.5 * (np.mod(np.roll(x, -1) - x, L_total) - a)**2
    return (E_periodic + E_springs).sum()

N = 30
a = 1.0
N_max = 100

L_total = N * a

omega_0 = 2 * np.pi / a
omega = (10 / N) * omega_0

alpha = 3. # strength of periodic force


alpha_FIRE_start = 0.25
f_alpha_FIRE = 0.99
delta_t_start = 0.01 # HEY!
delta_t_max = 10. * delta_t_start
delta_t_min = 0.02 * delta_t_start
delta_t_fdec = 0.5
N_delay = 20

# initialize x(t) and F(x(t))
x_nominal = a * np.arange(N, dtype=float)
x0 = x_nominal.copy()   # actual starting positions

v0 = np.zeros_like(x0)

alpha_FIRE = alpha_FIRE_start
delta_t = delta_t_start
f_delta_t_grow = 1.1
Npgt0 = 0

x = x0.copy()
v = v0.copy()
results = [[x.copy(), v.copy()]]
energy = [Energy(x0, a, L_total, omega, alpha)]
delta_ts = [delta_t_start]
t = 0.
for i in range(N_max):
    F = Force(x, L_total, omega, alpha)
    P = np.dot(F, v) # force you feel *dot* where you are going
    if P > 0:
        Npgt0 += 1
        F = Force(x, L_total, omega, alpha)
        F_norm = F / np.sqrt((F**2).sum())  # or np.linalg.norm()
        v_norm = v / np.sqrt((v**2).sum())
        v = (1-alpha_FIRE) * v + alpha_FIRE * F * (v_norm / F_norm) # This is it!
        if Npgt0 > N_delay:
            delta_t = min(f_delta_t_grow * delta_t, delta_t_max)
            alpha *= f_alpha_FIRE
    else: # P <= 0
        Npgt0 = 0
        v[:] = 0.   # stop! literally!
        delta_t = delta_t_fdec * delta_t
        alpha_FIRE = alpha_FIRE_start
    # now use https://en.wikipedia.org/wiki/Verlet_integration#Velocity_Verlet
    # get new x and v
    x += v * delta_t + 0.5 * F * delta_t**2
    FF = Force(x, L_total, omega, alpha) # force at t + delta_t
    v += 0.5 * (F + FF) * delta_t
    # then
    results.append([x.copy(), v.copy()])
    energy.append(Energy(x, a, L_total, omega, alpha))
    delta_ts.append(delta_t)
    t += delta_t
    # check for convergence and break if you like
    # for example, has the energy stopped decreasing significantly?

positions, velocities = np.swapaxes(np.array(list(zip(*results))), 1, 2)
energy = np.array(energy)
delta_ts = np.array(delta_ts)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=[10, 4])

for thing in positions:
    ax1.plot(thing)
ax1.set_title('positions')
ax2.plot(delta_ts)
ax2.set_title('delta_t')
ax3.plot(energy)
ax3.set_title('energy')
plt.show()
        
        

if True:
    fig, ax = plt.subplots(1, 1, figsize=[5, 7.5])
    for thing in positions:  
        ax.plot(thing)
    plt.subplots_adjust(bottom=0.05, top=0.95)
    for x in x_nominal:
        ax.plot([0], [x], '.')
    for x in x_nominal:
        ax.plot([N_max], [x], '.')
    if True:
        y = np.linspace(0, 30, 601)
        x = -2 * alpha * np.sin(omega * y)
        zero = np.zeros_like(x)
        ax.plot(x - 0.05 * N_max, y, '-k')
        ax.plot(x + 1.05 * N_max, y, '-k')
        ax.plot(zero - 0.05 * N_max, y, '--k', linewidth=0.5)
        ax.plot(zero + 1.05 * N_max, y, '--k', linewidth=0.5)      
    plt.show()










    
