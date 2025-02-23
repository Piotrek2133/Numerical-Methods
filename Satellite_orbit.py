# -*- coding: utf-8 -*-
"""
Dumbel on orbit

Piotr Imiński 247088
Jan Gorzela 247086

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
M = 5.972e24     # Mass of Earth (kg)
# R = 6.371e6      # Radius of Earth (m)
R = 6.371e6  + 820000 # Distance of the satelite mass centre from the centre of the Earth

# Initial conditions
m1 = 200000.0         # Mass 1 (kg)
m2 = 200000.0         # Mass 2 (kg)
l = 1000.0            # Length of rod (m)
v = 8661.2            # Initial linear velocity of the center of mass (m/s)
a = np.pi / 4         # Initial angle of rod (radians)

r1 = abs(-l*m2/(m1+m2))
r2 = l - r1

# Function to calculate gravitational force
def gravitational_force(t, state):
    x, y, vx, vy, theta, omega = state
    r = np.sqrt(x**2 + y**2)
    F = -G * M * (m1 + m2) / r**3 * np.array([x, y])
    return F

# Function to return derivatives
def deriv(t, state):
    x, y, vx, vy, theta, omega = state
    dxdt = vx
    dydt = vy
    dvxdt, dvydt = gravitational_force(t, state) / (m1 + m2)
    dthetadt = omega
    
    # Coordinates of mass 1 and mass 2
    x1 = x-r1*np.cos(theta)
    y1 = y-r1*np.sin(theta)
    x2 = x+r2*np.cos(theta)
    y2 = y+r2*np.sin(theta)
    
    # Distances of mass 1 and mass 2 from center of Earth
    R1 = np.sqrt(x1**2 + y1**2)
    R2 = np.sqrt(x2**2 + y2**2)
    
    # Forces on mass 1 and mass 2
    F1 = -G * M * m1 / R1**3 * np.array([x1, y1])
    F2 = -G * M * m2 / R2**3 * np.array([x2, y2])
    
    r1v = np.array([-r1*np.cos(theta), -r1*np.sin(theta)])
    r2v = np.array([r2*np.cos(theta), r2*np.sin(theta)])
    
    I = m1 * r1**2 + m2 * r2**2
    
    # Torque
    MF1 = np.cross(r1v, F1)
    MF2 = np.cross(r2v, F2)
    
    domegadt = (MF1-MF2)/I
    
    return np.array([dxdt, dydt, dvxdt, dvydt, dthetadt, domegadt])

# Time parameters
t_end = 20000
t_span = (0, t_end)  # Time span
t_eval = np.linspace(0, t_end, 100000)  # Time vector

# Initial state
state0 = np.array([R, 0.0, 0.0, v, a, 0.0])  # Initial state [x, y, vx, vy, theta, omega]

# Solve ODE
sol = solve_ivp(deriv, t_span, state0, t_eval=t_eval)

x1 = sol.y[0]
y1 = sol.y[1]
vx = sol.y[2]
vy = sol.y[3]
theta = sol.y[4]
omega = sol.y[5]

x2 = x1-r1*np.cos(theta)
y2 = y1-r1*np.sin(theta)

x3 = x1+r2*np.cos(theta)
y3 = y1+r2*np.sin(theta)

# Plotting
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot()
ax.plot(sol.y[0], sol.y[1], '-r', label='centre of mass')
ax.plot(x2,y2,'-g', label='mass 1')
ax.plot(x3,y3,'-y', label='mass 2')
earth = plt.Circle((0.0, 0.0), 6.371e6, color='blue')
ax.add_patch(earth)
ax.legend()
plt.title('Orbit of the Artificial Satellite')
plt.xlabel('X position (m)')
plt.ylabel('Y position (m)')
plt.axis('equal')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[4] * 180 / np.pi)
plt.title('Orientation of the Satellite Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Angle (°)')
plt.grid(True)
plt.show()



#Checking


Ek = (m1+m2)*(vx**2+vy**2)/2                # Kinetic Energy

Eko = (m1 * r1**2 + m2 * r2**2) *omega**2/2  # Kinetic energy of the rotational movement


U = -G*M * (m1+m2)/np.sqrt(x1**2+y1**2)     # Potential Energy


fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot()
ax.plot(sol.t,Ek,'-g', label='Kinetic energy')
ax.plot(sol.t,Eko,'-b', label='Kinetic energy of rotational movement')
ax.plot(sol.t,U, '-y', label='Potential energy')
ax.plot(sol.t,U+Ek, '-r', label='Sum of energies')
ax.legend()
plt.title('Energies')
plt.xlabel('time [s]')
plt.ylabel('Energy [J]')
plt.grid(True)
plt.show()
