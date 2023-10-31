### Résolution de l'équation de Schrödinger dépendante du temps

## Imports ======

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

from matplotlib.ticker import LinearLocator

from mpl_toolkits.mplot3d import Axes3D


## ====== Variables ======

sauv = False  # enregistrer les figures ?

# type d'intégration
type_integ = 'EE'
N_iter = 10


Lx = 7  # longueur de l'intervale
x0 = 0
Nx = 2*10**1
dx = Lx/(Nx+1)  # pas de discrétisation de la position
Ix = [l*dx + x0 for l in range(0, Nx+2)]

Ly = 1*Lx  # longueur de l'intervale
y0 = 0
Ny = 1*Nx
dy = Ly/(Ny+1)  # pas de discrétisation de la position
Iy = [l*dy + y0 for l in range(0, Ny+2)]


T = 1*10**3  # temps total
Nt = 3*10**3
dt = T/Nt  # pas de discrétisation du temps
It = [i*dt for i in range(Nt+1)]

hbar = 5*10**-34
m = 9*10**-31

psi_t = np.zeros((Nt+1, Nx+2, Ny+2), dtype=complex)  # initialise psi_t

## ====== Fonctions ======


def get_x(l):
    return l*dx + x0


def get_y(l):
    return l*dy + y0



# ------ Potentiels ------

def V(x,y):
    return 0


# ------ état initial ------

def psi_stationnaire_puitsInf(x, y, n=1, m=1):
    return 2*np.sqrt(1/(Lx*Ly))*np.sin((n*np.pi*(x-x0))/Lx)*np.sin((m*np.pi*(y-y0))/Ly) + 0j


def psi_gaussien(x, y, sigma_x=1, sigma_y=1, x1=0, y1=0, kx=0, ky=0):
    prefacteur_x = 1/(2*np.pi*sigma_x**2)**(1/4)
    prefacteur_y = 1/(2*np.pi*sigma_y**2)**(1/4)
    prefacteur = prefacteur_x*prefacteur_y
    arg_x = -((x-Lx/2-x0-x1)**2)/(4*sigma_x**2)
    arg_y = -((y-Ly/2-y0-y1)**2)/(4*sigma_y**2)
    arg = arg_x + arg_y
    propag_x = np.exp(1j*kx*(x-Lx/2-x0))
    propag_y = np.exp(1j*ky*(y-Ly/2-y0))
    propag = propag_x * propag_y
    return prefacteur*np.exp(arg)*propag

def dirac(lx,ly):
    A = 1/(dx*dy)
    if lx == int((Nx+1)/3) and ly == int((Ny+1)/2) :
        return A
    else:
        return 0



psi0 = np.zeros((Nx, Ny), dtype=complex)

for lx in range(1, Nx+1):
    for ly in range(1, Ny+1):
        # psi0[ly-1, lx-1] = psi_stationnaire_puitsInf(lx*dx+x0, ly*dy+y0, n=2, m=3)
        psi0[lx-1, ly-1] = psi_gaussien(get_x(lx), get_y(ly), kx=10.5, ky=0, sigma_y=0.6, sigma_x=0.6, x1=-2)
        # psi0[lx-1, ly-1] = dirac(lx, ly)


psi_t[0, 1:-1, 1:-1] = psi0  # initialise psi en t=0 en gardant les bords nuls








## Fonctions ======

def conditions_fentes(lx, ly):
    s = 0.6 # largeur fentes
    a = 0.1 # demi-distance entre les fentes
    e = 0.1 # épaisseur plaque
    mid_x = Lx/2
    x = get_x(lx)
    position_ecran_fentes = (x <= mid_x + e) and (x >= mid_x - e)
    mid_y = Ly/2
    y = get_y(ly)
    fente1 = (y >= mid_y+a) and (y <= mid_y+a+s)
    fente2 = (y <= mid_y-a) and (y >= mid_y-a-s)
    return position_ecran_fentes and not (fente1 or fente2)


def dpsi_dt_ij(psi, lx, ly):
    if conditions_fentes(lx, ly):
        return 0+0j
    ECx = -1/(2*dx**2)*(psi[lx-1, ly] - 2*psi[lx, ly] + psi[lx+1, ly])
    ECy = -1/(2*dy**2)*(psi[lx, ly-1] - 2*psi[lx, ly] + psi[lx, ly+1])
    EP = V(get_x(lx), get_y(ly))
    return -1j*( hbar/m*(ECx + ECy) + EP/hbar)


def dpsi_dt(psi):
    res = np.zeros((Nx+2, Ny+2), dtype=complex)
    for lx in range(1, Nx+1):
        for ly in range(1, Ny+1):
            res[lx, ly] = dpsi_dt_ij(psi, lx, ly)
    return res



## Résolution ======

# calculs ------



# Méthode explicite ---

def euler_step(f, psi):
    return psi + dt * f(psi)


def euler_explicite(f):
    for i in tqdm(range(Nt)):
        psi_t[i+1, :, :] = euler_step(dpsi_dt, psi_t[i, :, :])



# ---



## Calcul ======


if type_integ == 'EE':
    euler_explicite(dpsi_dt)





## Traitement ======




P = np.abs(psi_t)**2  #  densité de probabilité

S = hbar*np.angle(psi_t)





## Animation ======

cm = plt.cm.get_cmap('viridis')


X, Y = np.meshgrid(Ix[1:-1], Iy[1:-1])


frames_max = Nt+1



# Animation avec matplotlib.animation ------

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(2*5, 1*5))

fig.tight_layout(pad=4)  # règle l'espacement

# ax.set(xlim=(-3, 3), ylim=(-3, 3))

cax = []
for (i, ax) in enumerate(axs.flat):
    if i == 0:
        vmax = 0.5
        vmin = 0
    elif i == 1:
        vmax = 3*hbar
        vmin = -3*hbar
    cax.append(ax.pcolormesh(Ix[:-1], Iy[:-1], P[0, 1:-1, 1:-1],
                             vmin=vmin, vmax=vmax, cmap='viridis'))  # vmax=0.25,
    # fig.colorbar(cax[i])



def animate_prob(i):
    '''anime psi'''
    Z = P[i, 1:-1, 1:-1]
    return cax[0].set_array(Z.flatten())


anim_prob = animation.FuncAnimation(
    fig, animate_prob, frames=frames_max, interval=1, blit=False, repeat=True)





def animate_S(i):
    '''anime psi'''
    Z = S[i, 1:-1, 1:-1]
    return cax[1].set_array(Z.flatten())


anim_S = animation.FuncAnimation(
    fig, animate_S, frames=frames_max, interval=1, blit=False, repeat=True)




plt.show()


x, y = np.meshgrid(Ix, Iy)


def animate_gradS(i):
    gradxS = np.gradient(S[i])[0]
    gradyS = np.gradient(S[i])[1]

    u = gradxS
    v = gradyS
    return plt.quiver(x, y, u, v)


# fig_g, ax_g = plt.subplots()

# anim_gradS = animation.FuncAnimation(
#     fig_g, animate_gradS, frames=frames_max, interval=1, blit=False, repeat=True)


# plt.show()

## Ecran


fig_e, ax_e = plt.subplots()
ax_e.set(ylim=(0, 0.25))




def animate_e(i):
    '''anime psi'''
    x_e = int((Nx+1)/3)
    P_e = P[i, -x_e, :]
    return plt.plot(Iy, P_e, color='blue')


anim_e = animation.FuncAnimation(
    fig_e, animate_e, frames=frames_max, interval=1, blit=True, repeat=True)

plt.show()
