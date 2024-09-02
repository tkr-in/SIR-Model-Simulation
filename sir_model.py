import numpy as np
import matplotlib.pyplot as plt

# Parameters
beta = 0.3    # Infection rate
gamma = 0.1   # Recovery rate
N = 10000     # Total population

# Initial conditions
S0 = 9999    # Initial susceptible population
I0 = 1       # Initial infectious population
R0 = 0       # Initial recovered population
days = 160    # Simulation period in days

# Time array
t = np.linspace(0, days, days)

# SIR differential equations
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Initial conditions vector
y0 = S0, I0, R0

# Integrate the SIR equations over the time grid, t
ret = np.zeros((days, 3))
ret[0] = y0
for i in range(1, days):
    ret[i] = ret[i-1] + np.array(deriv(ret[i-1], t[i], N, beta, gamma)) * (t[i] - t[i-1])

S, I, R = ret.T

# Plot the data
fig = plt.figure(figsize=(10, 6))
plt.plot(t, S, 'b', label='Susceptible')
plt.plot(t, I, 'r', label='Infected')
plt.plot(t, R, 'g', label='Recovered')
plt.xlabel('Time /days')
plt.ylabel('Number')
plt.title('SIR Model')
plt.legend()
plt.grid()
plt.show()
