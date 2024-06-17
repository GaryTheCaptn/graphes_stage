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

print("-------------------------------------------------------------------------------------------------" + '\n')

normalized_m = [2, 3, 1, 2, 0]
normalized_v = [0, 1, 2, 2, 3]
inv_esp = 1 / 1000


def index_ligne_colonne(index, matrice_numeros):
    """
    :param index: l'index d'un x_k dans le vecteur x
    :param matrice_numeros: matrice avec les index de x, -1 sinon.
    :return: les indices i(ligne) et j (colonne) ou se trouve l'element x_i dans la matrice numeros
    """
    n = len(matrice_numeros)
    for i in range(0, n):
        for j in range(0, n):
            if matrice_numeros[i][j] == index:
                return i, j
    return -1, -1


def liste_numeros_meme_ligne(i, matrice_numeros):
    """
    :param i: un numero de ligne
    :param matrice_numeros: matrice avec les index de x, -1 sinon.
    :return: la liste des indices qui sont sur la ligne i.
    """
    res = []
    for j in range(i + 1, len(matrice_numeros)):
        res.append(matrice_numeros[i][j])
    return res


def liste_numeros_meme_colonne(j, matrice_numeros):
    """
    :param j: un numero de colonne
    :param matrice_numeros: matrice avec les index de x, -1 sinon.
    :return: la liste des indices qui sont sur la colonne j.
    """
    res = []
    for i in range(0, j):
        res.append(matrice_numeros[i][j])
    return res


def initialise_matrice_from_vect(x, n):
    matrice = [[0] * n for _ in range(n)]
    index = 0
    for i in range(n):
        for j in range(i + 1, n):
            matrice[i][j] = x[index]
            index += 1
    return matrice


def generation_matrice_numeros(n):
    """
    :param n: entier, taille de la matrice
    :return: une matrice avec les index de chaque case pour le vecteur colonne associe et des -1 pour les autres
    """
    M = [[-1] * n for _ in range(n)]
    index = 0
    for i in range(0, n):
        for j in range(i + 1, n):
            M[i][j] = index
            index += 1
    return M


def entropie_et_contraintes(x):
    res = 0
    # Entropie
    for x_i in x:
        if x_i > 0:
            res += x_i * np.log(x_i)

    # Reconstituer la matrice a partir du vect x pour faire les calculs sur les lignes et colonnes
    matrice = initialise_matrice_from_vect(x, n)

    # Contraintes sur les montees
    for i in range(n - 1):
        temp_sum = 0
        for j in range(i + 1, n):
            temp_sum += matrice[i][j]
        temp_sum += -normalized_m[i]
        res += inv_esp * temp_sum * temp_sum

    # Contraintes sur les descentes
    for j in range(1, n):
        temp_sum = 0
        for i in range(0, j):
            temp_sum += matrice[i][j]
        temp_sum += -normalized_v[j]
        res += inv_esp * temp_sum * temp_sum

    # Contraintes sur le caractère positif
    for x_i in x:
        res += max(-inv_esp * x_i, 0) ** 2

    return res


# Definition de la jacobienne
def jacobian_entropie_et_contraintes(x):
    res = []
    matrice_numeros = generation_matrice_numeros(n)
    index = 0
    for x_i in x:
        val = 0
        if x_i > 0:
            val += np.log(x_i) + 1
        i, j = index_ligne_colonne(index, matrice_numeros)
        liste_numeros_ligne = liste_numeros_meme_ligne(i, matrice_numeros)
        somme_ligne = 0
        for k in liste_numeros_ligne:
            somme_ligne += x[k]
        somme_ligne = 2 * inv_esp * (somme_ligne - normalized_m[i])
        val += somme_ligne

        liste_numeros_colonne = liste_numeros_meme_colonne(j, matrice_numeros)
        somme_colonne = 0
        for k in liste_numeros_colonne:
            somme_colonne += x[k]
        somme_colonne = 2 * inv_esp * (somme_colonne - normalized_v[j])
        val += somme_colonne
        res.append(val)
        index += 1
    return res


n = len(normalized_m)
verifier_gradient(entropie_et_contraintes, jacobian_entropie_et_contraintes, np.random.rand(int(n * (n - 1) / 2)))
