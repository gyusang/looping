import numpy as np
import pylab as pl

L = 0.5 # in m
M_m = 3
mu = 0.1
mu_s = 0.8
r = 0.01# in m

g = 9.80665

theta_0 = [np.pi/2, 0]
s_0 = [L*0.99-r*theta_0[0], 0]
steps_per_sec = 100000

def loop_pen(state, moving):
    theta = state[:2]
    s = state[2:4]
    if moving == 'CCW':
        theta_dd = (g*np.sin(theta[0]) - r*theta[1]*theta[1] - 2*s[1]*theta[1])/s[0]
        s_dd = (-g*np.cos(theta[0])+s[0]*theta[1]*theta[1]-r*theta_dd-M_m*np.exp(-mu*theta[0])*(r*theta_dd+g))/(1+M_m)
        T1_m = -s_dd + s[0]*theta[1]*theta[1] - g*np.cos(theta[0])
        if s[1] + r*theta[1] >= 0:
            moving = 'STOP'
        return moving, np.array([theta[1], theta_dd, s[1], s_dd])
    elif moving == 'STOP':
        theta_dd = (g*np.sin(theta[0])+r*theta[1]*theta[1])/s[0]
        s_d = -r*theta[1]
        s_dd = -r*theta_dd
        T1_m_min = M_m * g * np.exp(-mu_s * theta[0])
        T1_m_max = M_m * g * np.exp(mu_s * theta[0])
        T1_m = s[0]*theta[1]*theta[1]-g*np.cos(theta[0])
        if T1_m < -0.1:
            # moving = 'ERROR'
            print("Error! T1_m = %.2f"%T1_m)
        if T1_m <= T1_m_min:
            moving = 'CCW'
        elif T1_m >= T1_m_max:
            print('Amazing!!!')
            moving = 'CW'
            # TODO special analysis
        return moving, np.array([theta[1], theta_dd, s_d, s_dd])


def RK2(f,y0):
    # t = np.linspace(a,b,step)
    h=1/steps_per_sec
    Y=[y0]
    y=y0
    t = 0
    moving = 'STOP'
    while Y[-1][2] > 0.001 and Y[-1][0] >= 0:
        if t > 5 or moving == 'ERROR' or moving == 'CW':
            print('quit!')
            break
        moving, y_d_rk2 = f(y, moving)
        y_rk2 = y + h * 0.5 * y_d_rk2
        moving, y_d = f(y_rk2, moving)
        y = y + h * y_d
        Y.append(y)
        t += h
    return np.array(Y), np.linspace(0, t, len(Y))


y_0 = np.array(theta_0 + s_0)
a, t = RK2(loop_pen, np.array(theta_0 + s_0))
a = a[::100]
theta = a[:,0]
s = a[:,2]
X1 = r*np.cos(theta)-s*np.sin(theta)
Y1 = r*np.sin(theta)+s*np.cos(theta)
X2 = np.array([r]*len(a))
Y2 = (s+r*theta)-L

import matplotlib.pyplot as plt
import matplotlib.animation as animation
fig = plt.figure('Looping Pendulum')
ax = fig.gca()
line1, = ax.plot(X1[:1],Y1[:1],'g-')
line2, = ax.plot(X2[:1], Y2[:1], 'b-')
dot1, = ax.plot(X1[:1],Y1[:1],'g.')
dot2, = ax.plot(X2[:1], Y2[:1], 'b.')


def init():
    c = np.linspace(0, 2 * np.pi, 100)
    ax.plot(r * np.cos(c), r * np.sin(c), 'r-')
    line1.set_xdata([X1[0],r*np.cos(a[0,0])])
    line1.set_ydata([Y1[0],r*np.sin(a[0,0])])
    line2.set_xdata([X2[0],r])
    line2.set_ydata([Y2[0],0])
    dot1.set_xdata([X1[0]])
    dot1.set_ydata([Y1[0]])
    dot2.set_xdata([X2[0]])
    dot2.set_ydata([Y2[0]])
    return line1, line2, dot1, dot2


def animate(i):
    line1.set_xdata([X1[i], r * np.cos(theta[i])])
    line1.set_ydata([Y1[i], r * np.sin(theta[i])])
    line2.set_xdata([X2[i], r])
    line2.set_ydata([Y2[i], 0])
    dot1.set_xdata([X1[i]])
    dot1.set_ydata([Y1[i]])
    dot2.set_xdata([X2[i]])
    dot2.set_ydata([Y2[i]])
    return line1, line2, dot1, dot2


plt.axis('equal')
init()
xl = plt.xlim()
yl = plt.ylim()
plt.xlim((xl[0],-xl[0]))
plt.ylim((yl[0],-yl[0]))
ani = animation.FuncAnimation(fig, animate, range(len(a)), init_func=init, interval=1, blit=True)
plt.show()