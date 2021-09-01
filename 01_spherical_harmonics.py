#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.special as sc
import matplotlib.pyplot as plt
from numpy import sin, cos, sqrt


## Associated Legendre polynomial

def associated_legendre(l,m,x):
    return [sc.lpmn(m, l, xi)[0][m][l] for xi in x]


z = np.linspace(-1,1,200)

l = 2
m = 1

lp = associated_legendre(l, m, z)


## plotting example with legend

x=np.linspace(-np.pi,np.pi,200)
lts = ['-',':','--','-.','-']
for i in range(5):
    # In matplotlib, by default you over-plot.
    # No hold on
    plt.plot(x,sin(i*x)+i,lts[i],label="{0}*x".format(i))
plt.ylabel("sin")
plt.xlabel("x")

plt.show()


## Legendre plot
for l in [1, 2]:
    for m in range(0, l+1):
        legendre = associated_legendre(l, m, z)
        plt.plot(x, legendre, label=r"$P_{0}^{1}(cos\theta)$".format(l, m))
plt.legend()
plt.show()


## 3D plot

import matplotlib.pyplot as plt
from matplotlib import cm


phi = np.linspace(0, 2*np.pi, 180)
theta = np.linspace(0, np.pi, 90)
phi, theta = np.meshgrid(phi, theta)

# The Cartesian coordinates of the unit sphere
x = sin(theta) * cos(phi)
y = sin(theta) * sin(phi)
z = cos(theta)

Y21 = sqrt(3)*sin(theta)*cos(theta)*cos(phi)
#from scipy.special import sph_harm
# note (-1)^m Condon phase, and phi<->theta
#Y21 = sph_harm(1, 2, phi, theta).real 

# Calculate the spherical harmonic Y(l,m) and normalize to [0,1]
fmax, fmin = Y21.max(), Y21.min()
fcolors = (Y21 - fmin)/(fmax - fmin)

# Set the aspect ratio to 1 so our sphere looks spherical
fig = plt.figure(figsize=plt.figaspect(1.))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z,  rstride=1, cstride=1, facecolors=cm.seismic(fcolors))
# Turn off the axis planes
ax.set_axis_off()
plt.show()

from scipy.special import sph_harm

def harmonicPlot(m, l, ax):
    phi = np.linspace(0, 2*np.pi, 180)
    theta = np.linspace(0, np.pi, 90)
    phi, theta = np.meshgrid(phi, theta)
    
    # The Cartesian coordinates of the unit sphere
    x = sin(theta) * cos(phi)
    y = sin(theta) * sin(phi)
    z = cos(theta)
    
    
    # note (-1)^m Condon phase, and phi<->theta
    Y = sph_harm(m, l, phi, theta).real 
    
    # Calculate the spherical harmonic Y(l,m) and normalize to [0,1]
    fmax, fmin = Y.max(), Y.min()
    fcolors = (Y - fmin)/(fmax - fmin)
    
    # Set the aspect ratio to 1 so our sphere looks spherical
    
    ax.plot_surface(x, y, z,  rstride=1, cstride=1,
                    facecolors=cm.seismic(fcolors))
    # Turn off the axis planes
    ax.set_axis_off()
    ax.set_title(r"$Y^{0}_{1}$".format(m, l), fontsize=30)

    return ax
    

lm = [(1,0), (1,1), (2,0), (2,1), (2,2)]
fig, axes = plt.subplots(ncols=len(lm), subplot_kw={'projection':'3d'})
for i in range(0, len(lm)):
    l, m = lm[i]
    axes[i] = harmonicPlot(m, l, axes[i])

plt.show()