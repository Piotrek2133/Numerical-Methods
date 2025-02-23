# -*- coding: utf-8 -*-
"""
Piotr Imiński 247088
Jan Gorzela 247086

Problem 14

Program to calculate the equalibrium position of the block on two springs.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve, norm
import numdifftools as ndt

# Constans: 
# l1 -- length of spring 1
# l2 -- length of spring 2
# k1 -- stiffeness coefficiant of spring 1
# k2 -- stiffeness coefficiant of spring 2
# m  -- mass of the block
# g  -- gravitational acceleration
# D  -- displcement of spring 2
# a  -- side of the box
# b  -- other side of the box

l1 = 0.2
l2 = 0.1
k1 = 475
k2 = 675
m = 5
g = 9.81
D = 0.3
a = 0.05
b = 0.1

# xyfi = [x, y, fi]                               x, y -- coordinates of point P1 in m, fi -- angle of inclination of the top side in rad


def f1(x1, y1, fi):                             # Returns sum of forces in y axis (Equation 2)
    
    l1c = np.sqrt(x1 ** 2 + y1 ** 2)            # Total length of spring 1
    d1 = l1c - l1                               # Elongation of spring 1
    x2 = x1 + np.sign(D) * b * np.cos(fi)       # x coordinate of point P2
    y2 = y1 + b * np.sin(fi)                    # y coordinate of point P2
    l2c = np.sqrt((D - x2) ** 2 + y2 ** 2)      # Total length of spring 2
    d2 = l2c - l2                               # Elongation of spring 2
    
    return k1 * d1 * abs(y1) / l1c + k2 * d2 * abs(y2) / l2c - m * g


def f2(x1, y1, fi):                             # Returns sum of forces in x axis (Equation 1)

    l1c = np.sqrt(x1 ** 2 + y1 ** 2)            # Total length of spring 1
    d1 = l1c - l1                               # Elongation of spring 1
    x2 = x1 + np.sign(D) * b * np.cos(fi)       # x coordinate of point P2
    y2 = y1 + b * np.sin(fi)                    # y coordinate of point P2
    l2c = np.sqrt((D - x2) ** 2 + y2 ** 2)      # Total length of spring 2
    d2 = l2c - l2                               # Elongation of spring 2
    
    return -k1 * d1 * x1 / l1c + k2 * d2 * (D - x2) / l2c


def f3(x1, y1, fi):                             # Returns sum of moments of forces (Equation 13)
    
    x2 = x1 + np.sign(D) * b * np.cos(fi)       # x coordinate of point P2
    y2 = y1 + b * np.sin(fi)                    # y coordinate of point P2
    l2c = np.sqrt((D - x2) ** 2 + y2 ** 2)      # Total length of spring 2
    d2 = l2c - l2                               # Elongation of spring 2
    
    return -b * k2 * d2 * np.sin(fi + np.pi / 2 + np.arccos(abs(y2) / l2c)) + 1 / 2 * np.sqrt(a ** 2 + b ** 2) * m * g * np.sin(np.pi / 2 + np.arctan(a / b) - fi)


def f(xyfi):                                    # Creates array of values of functions
    return np.array([f1(xyfi[0], xyfi[1], xyfi[2]), f2(xyfi[0], xyfi[1], xyfi[2]), f3(xyfi[0], xyfi[1], xyfi[2])])


J = ndt.Jacobian(f)

xyfi0 = np.array([-0.2,-0.3,-np.pi/9])          # Array of initial values
xyfi = [xyfi0]

more = True
cs = 1e-12                                      # Maximum error
maxcount = 1000                                 # Maximum number of loops
loopsc = 0                                      # Number of loops


while more:
    loopsc +=1
    corection = solve(J(xyfi[-1]),f(xyfi[-1]))  # Calculates the correction
    xyfi.append(xyfi[-1]-corection)             # Adds to the list of solutions corrected values
    res = norm(f(xyfi[-1]))
    if res < cs:                                # Checks whether the error is satisfactionary small
        print("Coordinates of point P1:\n", xyfi[-1])
        more = False
    if loopsc > maxcount:
        more = False


# Coordinates of the vertices of the rectangle:
P1 = [xyfi[-1][0], xyfi[-1][1]]
P2 = [xyfi[-1][0] + np.sign(D) * b * np.cos(xyfi[-1][2]), xyfi[-1][1] + b * np.sin(xyfi[-1][2])]
P3 = [xyfi[-1][0] + np.sign(D) * a * np.sin(xyfi[-1][2]), xyfi[-1][1] - a * np.cos(xyfi[-1][2])]
P4 = [xyfi[-1][0] + np.sign(D) * b * np.cos(xyfi[-1][2]) + np.sign(D) * a * np.sin(xyfi[-1][2]), xyfi[-1][1] + b*np.sin(xyfi[-1][2]) - a * np.cos(xyfi[-1][2])]

Pc = [(P1[0]+P4[0])/2,(P2[1]+P3[1])/2]          # Coordinates of center of mass
fid = xyfi[-1][2] * 180 / np.pi                  # Angle in degrees


#
print("\nCoordinates of mass center:\n", Pc, "m", fid, "°")

# Plots:
fig = plt.figure()
ax = fig.add_subplot(111)
ax.axis('equal')
tyt = "Equalibrium point of the block on two springs.\n Stiffness coefficiant of springs: k1 = {:.2f} [N/m], k2 = {:.2f} [N/m],\n length of the springs: L1 = {:.2f} [m], L2 = {:.2f} [m],\n displacement of spring 2: D = {:.2f} [m], mass of the block = {:.2f} [kg],\n Coordinates of mass center [{:.3f}, {:.3f}] [m],\n angle of the inclination of top side = {:.2f}°".format(k1,k2,l1,l2,D,m,Pc[0],Pc[1],fid)
ax.set(title=tyt,xlabel='x [m]',ylabel = 'y [m]')
ax.plot([0,xyfi[-1][0]],[0,xyfi[-1][1]],'-g')
ax.plot([D,P2[0]],[0,P2[1]],'-g')
ax.plot([P1[0],P2[0],P4[0],P3[0],P1[0]],[P1[1],P2[1],P4[1],P3[1],P1[1]], '-r')
ax.plot(Pc[0],Pc[1],'p')
plt.axhline(y=0, color='k')



# Check

fi = xyfi[-1][2]                # angle in rad

print("\nCoordinates of top corners:\nLeft: {}\nRight: {}".format(P1, P2))

l1c = np.sqrt(P1[0] ** 2 + P1[1] ** 2)
l2c = np.sqrt((D - P2[0]) ** 2 + P2[1] ** 2)
d1 = l1c - l1
d2 = l2c - l2  

F1 = np.array([-k1 * d1 * P1[0] / l1c , k1 * d1 * abs(P1[1]) / l1c])              # Forces on P1

F2 = np.array([k2 * d2 * (D - P2[0]) / l2c,  k2 * d2 * abs(P2[1]) / l2c])         # Forces on P2

Fg = np.array([0, -m * g])                                                        # Force of gravity

Fs = F1 + F2 + Fg                                                       # Sum of forces

# Moments of forces

Mc = 1 / 2 * np.sqrt(a ** 2 + b ** 2) * m * g * np.sin(np.pi / 2 + np.arctan(a / b) - fi)               # Moment of mass center

MP2 = -b * k2 * d2 * np.sin(fi + np.pi / 2 + np.arccos(abs(P2[1]) / l2c))       # Moment of P2

Ms = Mc + MP2                       # Sum of moments


print("\n\nTests:")
print("\nMoments of forces:\n\tmomC: {}\n\tmomP2: {}\n\tsum: {}".format(Mc,MP2,Ms))
print("\nForces:\n\tLeft: {}\n\tRight: {}\n\tGravity: {}\n\tsum: {}".format(F1,F2,Fg,Fs))





Fxs = f2(xyfi[-1][0],xyfi[-1][1],xyfi[-1][2])

Fys = f1(xyfi[-1][0],xyfi[-1][1],xyfi[-1][2])

Ms = f3(xyfi[-1][0],xyfi[-1][1],xyfi[-1][2])

print('\nSum of forces in x direction:\n', Fxs, " N")
print('\nSum of forces in y direction:\n', Fys, " N")
print('\nSum of moments of forces:\n', Ms, " N m")
