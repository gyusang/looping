import numpy as np
import pylab as pl

L = 0.5 # in m
M_m = 4
mu = np.log(81/52)/np.pi
mu_s = np.log(99/52)/np.pi
print("mu: %.2f, mu_s: %.2f"%(mu, mu_s))
r = 0.023/2# in m

g = 9.80665

theta_0 = [np.pi/2, 0]
s_0 = [L*0.99-r*theta_0[0], 0]
steps_per_sec = 100000
fps = 100

def loop_pen(state, moving):
    theta = state[:2]
    s = state[2:4]
    if moving == 'CCW':
        theta_dd = (g*np.sin(theta[0]) - r*theta[1]*theta[1] - 2*s[1]*theta[1])/s[0]
        s_dd = (-g*np.cos(theta[0])+s[0]*theta[1]*theta[1]-r*theta_dd-M_m*np.exp(-mu*theta[0])*(r*theta_dd+g)) \
            /(1+M_m * np.exp(-mu*theta[0]))
        T1_m = -s_dd + s[0]*theta[1]*theta[1] - g*np.cos(theta[0])
        if s[1] + r*theta[1] >= 0:
            moving = 'STOP'
        return moving, np.array([theta[1], theta_dd, s[1], s_dd])
    elif moving == 'CW':
        theta_dd = (g * np.sin(theta[0]) - r * theta[1] * theta[1] - 2 * s[1] * theta[1]) / s[0]
        s_dd = (-g * np.cos(theta[0]) + s[0] * theta[1] * theta[1] - r * theta_dd - M_m * np.exp(mu * theta[0]) * (
                    r * theta_dd + g)) / (1 + M_m * np.exp(mu * theta[0]))
        T1_m = -s_dd + s[0] * theta[1] * theta[1] - g * np.cos(theta[0])
        if s[1] + r * theta[1] <= 0:
            moving = 'STOP'
        return moving, np.array([theta[1], theta_dd, s[1], s_dd])
    elif moving == 'STOP':
        theta_dd = (g*np.sin(theta[0])+r*theta[1]*theta[1])/s[0]
        s_d = -r*theta[1]
        s_dd = -r*theta_dd
        T1_m_min = M_m * g * np.exp(-mu_s * theta[0])
        T1_m_max = M_m * g * np.exp(mu_s * theta[0])
        T1_m = s[0]*theta[1]*theta[1]-g*np.cos(theta[0])
        if T1_m < -0.01:
            # moving = 'ERROR'
            print("Error! T1_m = %.2f"%T1_m)
        if T1_m <= T1_m_min:
            moving = 'CCW'
        elif T1_m >= T1_m_max:
            moving = 'CW'
            # TODO special analysis
        return moving, np.array([theta[1], theta_dd, s_d, s_dd])


def RK2(f,y0):
    # t = np.linspace(a,b,step)
    h=1/steps_per_sec
    Y=[y0]
    states = [0]
    y=y0
    t = 0
    moving = 'STOP'
    while Y[-1][2] > 0.001 and Y[-1][0] >= 0:
        if t > 5 or moving == 'ERROR':
            print('quit!')
            break
        moving, y_d_rk2 = f(y, moving)
        y_rk2 = y + h * 0.5 * y_d_rk2
        if moving == 'ERROR':
            print('quit!')
            break
        moving, y_d = f(y_rk2, moving)
        y = y + h * y_d
        Y.append(y)
        if moving == 'STOP':
            states.append(0)
        elif moving == 'CCW':
            states.append(-1)
        elif moving == 'CW':
            states.append(1)
        t += h
    return np.array(Y), np.linspace(0, t, len(Y)), np.array(states)


# y_0 = np.array(theta_0 + s_0)
a, t, states = RK2(loop_pen, np.array(theta_0 + s_0))
a = a[::steps_per_sec//fps]
t = t[::steps_per_sec//fps]
states = states[::steps_per_sec//fps]
theta = a[:,0]
s = a[:,2]
X1 = r*np.cos(theta)-s*np.sin(theta)
Y1 = r*np.sin(theta)+s*np.cos(theta)
X2 = np.array([r]*len(a))
Y2 = (s+r*theta)-L

import matplotlib.pyplot as plt
import matplotlib.animation as animation
fig = plt.figure('Looping Pendulum')
fig.patch.set_alpha(0.)
ax = fig.gca()
line1, = ax.plot(X1[:1],Y1[:1],'g-')
line2, = ax.plot(X2[:1], Y2[:1], 'b-')
dot1, = ax.plot(X1[:1],Y1[:1],'g.')
dot2, = ax.plot(X2[:1], Y2[:1], 'b.')
c = np.linspace(0, 2 * np.pi, 100)
circle, = ax.plot(r * np.cos(c), r * np.sin(c), 'r-')


def init():
    line1.set_xdata([X1[0],r*np.cos(a[0,0])])
    line1.set_ydata([Y1[0],r*np.sin(a[0,0])])
    line2.set_xdata([X2[0],r])
    line2.set_ydata([Y2[0],0])
    dot1.set_xdata([X1[0]])
    dot1.set_ydata([Y1[0]])
    dot2.set_xdata([X2[0]])
    dot2.set_ydata([Y2[0]])
    return line1, line2, dot1, dot2, circle


def animate(i):
    line1.set_xdata([-X1[i], -r * np.cos(theta[i])])
    line1.set_ydata([Y1[i], r * np.sin(theta[i])])
    line2.set_xdata([-X2[i], -r])
    line2.set_ydata([Y2[i], 0])
    dot1.set_xdata([-X1[i]])
    dot1.set_ydata([Y1[i]])
    dot2.set_xdata([-X2[i]])
    dot2.set_ydata([Y2[i]])
    return line1, line2, dot1, dot2, circle


init()
plt.axis('equal')
ax.relim()
xl = ax.get_xlim()
yl = ax.get_ylim()
ax.set_xlim((xl[0],-xl[0]))
ax.set_ylim((yl[0],-yl[0]))

fig2 = plt.figure('Graphs')
ax_s_t = fig2.add_subplot(321)
ax_t_t = fig2.add_subplot(322)
ax_h_t = fig2.add_subplot(323)
ax_w_t = fig2.add_subplot(324)
ax_hd_t = fig2.add_subplot(325)
ax_states = fig2.add_subplot(326)
ax_s_t.set_title("S-t")
ax_h_t.set_title("Y2-t")
ax_t_t.set_title("Theta-t")
ax_w_t.set_title("Omega-t")
ax_hd_t.set_title("v-t")
ax_states.set_title("states")
ax_s_t.plot(t, s, 'C1-')
ax_t_t.plot(t, theta, 'C1-')
ax_h_t.plot(t, Y2, 'C1-')
ax_w_t.plot(t, a[:,1], 'C1-')
ax_hd_t.plot(t, a[:,3]+r*a[:,1], 'C1-')
ax_hd_t.grid()
ax_states.plot(t, states, 'C1.')

fig2.tight_layout()

ani = animation.FuncAnimation(fig, animate, range(len(a)), init_func=init, interval=1000//fps, blit=True)
Writer = animation.writers['ffmpeg']
writer = Writer(fps=60, metadata=dict(artist='Sanggyu Lee'), bitrate=1800)
ani.save('looping.mp4', dpi=300, savefig_kwargs={'transparent':True, 'facecolor':'none'}, writer=writer)
# ani.save('looping.mp4', codec="png",
#          dpi=100, bitrate=-1,
#          savefig_kwargs={'transparent': True, 'facecolor': 'none'}, writer=writer)
plt.show()