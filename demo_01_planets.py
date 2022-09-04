import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def deriv(t, y, M, G): # time derivative of state vector
    x, v = y.reshape(2, -1)
    acc = -G * M * x * ((x**2).sum())**-1.5  # - GM * x_hat / abs(x)^2
    return np.hstack((v, acc))

two_pi = 2 * np.pi

G = 1.   # Gravitational Constant
M = 1.   # Solar mass
a = 1.   # Astronomical Unit (AU)
e = 0.5   # eccentricity
r_peri = (1 - e) * a 
v_peri = (2/r_peri - 1/a)**0.5  

T_period = two_pi

t_eval = np.linspace(0, T_period, 201)
t_span = t_eval.min(), t_eval.max()

args = (M, G)

# initial state vector: [x, y, vx, vy]
y0 = np.array([r_peri, 0, 0, v_peri], dtype=float)

method = 'DOP853'
dense_output = False

# variable step sizes (internal)
answer = solve_ivp(deriv, t_span, y0, method=method, t_eval=t_eval,
                   dense_output=dense_output, events=None, args=args)

# interpolated final state vectors evaluated at times: t_eval
x, y, vx, vy = answer.y 


plt.plot(x, y)
plt.scatter(x[::10], y[::10], marker='o', c=t_eval[::10], cmap='jet')
plt.plot([0], [0], 'oy', ms=20)
plt.gca().set_aspect('equal')
plt.show()
