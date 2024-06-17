import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

# la commande suivante agrandit les figures
plt.rcParams['figure.figsize'] = [9., 6.]


def verifier_gradient(f, g, x0):
    N = len(x0)
    gg = np.zeros(N)
    for i in range(N):
        eps = 1e-4
        e = np.zeros(N)
        e[i] = eps
        gg[i] = (f(x0 + e) - f(x0 - e)) / (2 * eps)
    print('erreur numerique dans le calcul du gradient: %g (doit etre petit)' % np.linalg.norm(g(x0) - gg))


y = np.array([0.1, 1.5, 2.1])


def g(k):
    return np.sum(np.maximum(y - k, 0)) - 1.0


k = scipy.optimize.root(g, x0=0, method='anderson').x
print(k)


def proj_simplexe(y):
    g = lambda k: np.sum(np.maximum(y - k, 0)) - 1.0
    sol = scipy.optimize.root(g, x0=0, method='anderson')  # scipy.optimize.fsolve(g,0)
    k = sol.x
    return np.maximum(y - k, 0)


# validation: calcul de produits scalaires
n = 10
y = np.random.randn(n)
p = proj_simplexe(y)
for i in range(n):
    ei = np.zeros(n)
    ei[i] = 1
    print(np.dot(y - p, p - ei))

Q = np.array([[1199.6242199, -225.74269344, 270.42617708, -112.31853678],
              [-225.74269344, 224.42514399, -157.75776414, 46.31290714],
              [270.42617708, -157.75776414, 600.37115079, -28.1665365],
              [-112.31853678, 46.31290714, -28.1665365, 21.77422792]])
r = np.array([0.20888442, 0.55953316, 0.00602209, 0.2042723])
r0 = 0.7 * np.max(r)
eta = 1e-4


def f(x):
    return .5 * np.dot(x, Q @ x) + .5 / eta * (np.dot(x, r) - r0) ** 2


def gradf(x):
    return Q @ x + 1 / eta * (np.dot(x, r) - r0) * r


# vérification du calcul du gradient
verifier_gradient(f, gradf, np.random.rand(len(r)))
