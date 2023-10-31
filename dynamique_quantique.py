### Résolution de l'équation de Schrödinger dépendante du temps


## ====== Imports ======

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
import tqdm

matplotlib.rcParams['animation.ffmpeg_path'] = r"C:\\ffmpeg\\bin\\ffmpeg.exe"

# from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3


## ====== Variables ======

sauv = False # enregistrer les figures ?

# type d'intégration
type_integ = 'Crank et Nicolson'

# type de potentiel (à choisir)
type_V = 'harmonique'
w = 1

L = 6 # longueur de l'intervale
x0 = -L/2
Nx = 3*10**2
dx = L/(Nx+1) # pas de discrétisation de la position
x = [l*dx + x0 for l in range(0,Nx+2)]


T = 4*np.pi/w # temps total
Nt = 2*3*10**2
dt = T/Nt # pas de discrétisation du temps
t = [i*dt for i in range(Nt+1)]

psi_t = np.zeros((Nt+1, Nx+2), dtype=complex) # initialise psi_t



# ------ Potentiels ------

def V(x, type):
    '''Potentiel'''
    if type == 'puits infini':
        return V_cste(x)
    if type == 'harmonique':
        return V_harmonique(x)
    if type == 'double puits':
        return V_double_puits(x)


def V_cste(x, c=0):
    return c


def V_harmonique(x, w=w):
    return 1/2*w**2*(x)**2


def V_double_puits(x, DV=5, x1=1):
    return DV*((x/x1)**2-1)**2


V_x = np.array([V(val_x, type_V) for val_x in x])/10


# ------ état initial ------

def psi_stationnaire_puitsInf(x, m=1):
    return np.sqrt(2/L)*np.sin((m*np.pi*(x-x0))/L) + 0j


def psi_gaussien(x, sigma=0.2, shift=0.5, k=0):
    # pour bien faire : sigma = sqrt(w/2)
    prefacteur = 1/(sigma**2 * 2*np.pi)**(1/4)
    arg = -((x-L/2-x0-shift)**2)/(4*sigma**2)
    propag = np.exp(k*1j*(x-L/2-x0-shift))
    return prefacteur*np.exp(arg)*propag


if type_V == 'puits infini':
    # définit psi avec une fonction propre du puits infini de potentiel
    psi0 = np.array([psi_stationnaire_puitsInf(l*dx+x0, m=4) for l in range(1,Nx+1)])
elif type_V == 'harmonique':
    # définit psi avec un paquet d'onde gaussien
    psi0 = np.array([psi_gaussien(l*dx+x0, shift=3.5, k=0, sigma=np.sqrt(w/2)) for l in range(1, Nx+1)])
elif type_V == 'double puits':
    # définit psi avec un paquet d'onde gaussien
    psi0 = np.array([psi_gaussien(l*dx+x0, shift=1, k=0, sigma=0.2) for l in range(1, Nx+1)])



psi_t[0][1:-1] = psi0  # affecte les valeurs à psi en t=0 en gardant les bords nuls


## ====== Fonctions ======


def get_x(l):
    return l*dx + x0



def integ(f_x):
    return integ_carres(f_x)

def integ_carres(f_x):
    return np.sum(dx*np.array(f_x))

def integ_trapezes(f_x):
    return dx*((f_x[0]+f_x[-1])/2 + np.sum(np.array(f_x)[1:-1]))



def val_moy(A, psi):
    integrande = np.conjugate(psi) * A @ (psi) # calcule psi*(x) A psi(x)
    return np.real(integ(integrande)) # intègre le résultat sur tout l'intervalle


# ------ Opérateurs ------


# --- hamiltonien ---

H = np.zeros((Nx, Nx))
for l in range(1, Nx-1):
    H[l, l-1] = -1/(2*dx**2)
    H[l, l] = 1/dx**2 + V(get_x(l+1), type_V)
    H[l, l+1] = -1/(2*dx**2)

H[0, 0] = 1/dx**2 + V(get_x(1), type_V)
H[0, 1] = -1/(2*dx**2)
H[-1, -1] = 1/dx**2 + V(get_x(Nx), type_V)
H[-1, -2] = -1/(2*dx**2)


# --- position ---

X = np.zeros((Nx+2, Nx+2))
for l in range(0, Nx+2):
    X[l, l] = get_x(l)





## ====== Intégration ======


# ------ Méthode semi-implicite ------

def Crank_et_Nicolson():
    # on va résoudre le système AX = B (cf rapport)
    
    A = np.identity(Nx) + 1j*dt/2*H
    
    for j in tqdm.tqdm(range(0, Nt)):
        B = (np.identity(Nx) - 1j*dt/2*H) @ psi_t[j, 1:-1]
        psi_t[j+1, 1:-1] = np.linalg.solve(A, B)


# ------ Méthode explicite ------


def membre_droite(psi):
    return H @ (-1j * psi)

def euler_explicite_step(f, psi):
    return psi + dt * f(psi)

def euler_explicite(f):
    for i in range(Nt):
        psi_t[i+1, 1:-1] = euler_explicite_step(f, psi_t[i, 1:-1])



## === Calcul ===

if type_integ == 'Crank et Nicolson':
    Crank_et_Nicolson()
else : 
    euler_explicite(membre_droite)


# ====== Traitement du résultat ======

P_t = np.array([np.abs(p)**2 for p in psi_t]) # densité de probabilité
R_psi_t = np.array([np.real(p) for p in psi_t])  # partie réelle de psi
I_psi_t = np.array([np.imag(p) for p in psi_t])  # partie imaginaire de psi



# probabilité totale en fonction du temps
Ptot_t = [integ(p) for p in P_t]



# valeur moyenne de l'énergie en fonction du temps
E_moy_t = [val_moy(H, psi[1:-1]) for psi in psi_t]

# valeur moyenne de la position en fonction du temps
X_moy_t = [val_moy(X, psi) for psi in psi_t]



## ====== Affichage ======

frames_max = Nt+1
fact = 1
plt.rcParams['font.size'] = 12
plt.rcParams["savefig.dpi"] = 1000

# ------ Plot l'évolution ------

def plot_evol(n=2, m=3, save=False, psi=True, prop_temps=1):
    '''plot l'évolution de la fonction d'onde
    n : lignes, m : colonnes '''

    fig_evol, axs_evol = plt.subplots(nrows=n, ncols=m, figsize=(m*5, n*5))

    ind_t_max = Nt/prop_temps # l'indice en temps auquel on s'arrête
    step = ind_t_max/(n*m) # pas de temps


    fig_evol.tight_layout(pad=4) # règle l'espacement

    for (i, ax) in enumerate(axs_evol.flat):
        ax.set(xlabel=r'$x$', ylabel=r'$| \psi |^{2}$') # label des axes

        # bornes
        ax.set_xlim([x0, L+x0])
        ax.set_ylim([-1, 4])

        # donne l'indice en temps
        t_ind = int(i*step)

        ax.plot(x, V_x, color = 'black', label=r'$V$') # plot du potentiel
        ax.plot(x, P_t[t_ind], label=r'$| \psi |^{2}$') # plot de la densité de probabilité
        if psi:
            ax.plot(x, R_psi_t[t_ind], label=r'$\Re(\psi)$', color='orange') # plot de la partie réelle
            ax.plot(x, I_psi_t[t_ind], label=r'$\Im(\psi)$', color='purple') # plot de la partie imaginaire

        # titre du subplot
        ax.set_title(r'$t_{' + str(i) + '}$')

    # légende
    handles, labels = ax.get_legend_handles_labels()
    fig_evol.legend(handles, labels, loc='lower center', ncol=4,
                    bbox_transform=fig_evol.transFigure)

    if save:
        fig_evol.savefig('Rapport/Figures/Dynamique potentiel ' + type_V + '.png')

    plt.show()


if type_V == 'double puits':
    plot_evol(save=sauv, psi=False)

if type_V == 'harmonique':
    plot_evol(save=sauv, prop_temps=6.6)


# ------ Plot quantitées conservées en fonction du temps ------

def plot_conserv(save=False):
    # plt.title('')
    
    # axes
    plt.xlim(0, T)
    plt.ylabel('Erreur relative')
    plt.xlabel(r'$t$')
    
    var_Ptot = (np.array(Ptot_t)-np.ones(Nt+1))/1
    var_E_moy_t = (np.array(E_moy_t)-np.ones(Nt+1)*E_moy_t[0])/E_moy_t[0]
    

    plt.plot(t, var_E_moy_t, color='orange', linestyle='dotted', label=r'Variation relative de $\langle \hat{E} \rangle$')
    plt.plot(t, var_Ptot, color='blue', linestyle='dotted', label=r'Variation relative de $P_{total}$')
    plt.legend(loc='upper left')
    
    if save :
        plt.savefig('Rapport/Figures/Conservations potentiel ' + type_V + '.png')
    
    plt.show()

if type_V == 'puits infini':
    plot_conserv(save=sauv)


# ------ Plot X_moy en fonction du temps ------

def plot_X_moy(save=False):
    # plt.title('')
    
    # axes
    plt.xlim(0, T)
    plt.ylim(x0+L/2-2, L+x0-L/2+2)
    plt.ylabel(r'$\langle \hat{X} \rangle$')
    plt.xlabel(r'$t$')

    plt.plot(t, [0.5*np.cos(w*t_) for t_ in t], color='red', linestyle='solid', label='X classique') # plot du graphe classique
    plt.plot(t, X_moy_t, color='blue', linestyle='dashed', label=r'$\langle \hat{X} \rangle$ quantique') # plot du graphe calculé
    plt.legend()
    
    if save :
        plt.savefig('Rapport/Figures/X_moy potentiel ' + type_V + '.png')
    
    plt.show()

if type_V == 'harmonique':
    plot_X_moy(save=sauv)


# ------ Animation de l'évolution ------


def animate_psi(i, lines):
    '''anime psi'''
    # densité de probabilité
    lines[0].set_data(x, P_t[fact*i])

    # partie réelle
    lines[1].set_data(x, R_psi_t[fact*i])

    # partie imaginaire
    lines[2].set_data(x, I_psi_t[fact*i])

    return lines

def anim_evol():
    fig_evol = plt.figure()  # initialise la figure
    plt.plot(x, V_x, color = 'black', label=r'$V$') # affichage du potentiel

    # initialisation
    lines = [
    plt.plot([], [], label=r'$| \psi |^{2}$', color='blue')[0],
    plt.plot([], [], label=r'$\Re(\psi)$', color='orange')[0],
    plt.plot([], [], label=r'$\Im(\psi)$', color='purple')[0]
    ]

    # axes
    plt.xlim(x0, L+x0)
    plt.xlabel(r'$x$')
    plt.ylim(-1, 4)
    plt.ylabel(r'$| \psi |^{2}$')

    anim_psi = animation.FuncAnimation(fig_evol, animate_psi, frames=int(frames_max/fact), interval=17, blit=True, repeat=True, fargs=[lines])
    plt.legend()

    if sauv:
        writervideo = animation.PillowWriter(fps=24)
        anim_psi.save('Animation dynamique potentiel ' + type_V + '.gif', writer=writervideo)

    plt.show()


if type_V != 'puits infini':
    anim_evol()


# ------ Animation de l'évolution en 3D ------

def anim_evol_3D():
    fig_evol_3D = plt.figure()
    
    # axes
    plot3D = p3.Axes3D(fig_evol_3D)
    plot3D.set_xlim(x0, L+x0)
    plt.xlabel(r'$x$')
    plot3D.set_ylim(-5, 5)
    plt.ylabel(r'$\Re(\psi)$')
    plot3D.set_zlim(-5, 5)
    plot3D.set_zlabel(r'$\Im(\psi)$')
    
    # initialisation
    line, = plot3D.plot(x, R_psi_t[0], I_psi_t[0], label=r'\psi', color='purple')
    
    anim_psi_3D = animation.FuncAnimation(
    fig_evol_3D, animate3D, frames=int(frames_max/fact), interval=17, blit=True, repeat=True, fargs=[line])
    
    if sauv:
        writervideo = animation.PillowWriter(fps=24)
        anim_psi_3D.save('Animation 3D dynamique potentiel ' + type_V + '.gif', writer=writervideo)

    plt.show()



def animate3D(i, line):
    '''anime psi'''
    line.set_data(x, R_psi_t[fact*i])
    line.set_3d_properties(I_psi_t[fact*i])
    # plot_P = plot3D.plot(x, [0]*(Nx+2), P_t[i], color='blue')
    return line,


anim_evol_3D()


