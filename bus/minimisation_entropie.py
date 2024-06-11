import math

import numpy as np
from math import *
import matplotlib.pyplot as plt
import scipy.optimize


def penalisation(m, d, eps):
    """
    :param m: liste des montees (contrainte)
    :param d: liste des descentes (contrainite)
    :param eps: valeur de la penalisation
    :return: la matrice OD minimisant l'entropie par la methode de penalisation
    """
    n = len(m)
    inv_esp = 1 / eps

    # Definition de la fonction a minimiser avec l'ajout des contraintes
    def entropie_et_contraintes(x):
        res = 0
        # Entropie
        for x_i in x:
            if x_i > 0:
                res += x_i * np.log(x_i)

        # Reconstituer la matrice pour faire les calculs sur les lignes et colonnes
        matrice = [[0] * n for _ in range(n)]
        index = 0
        for i in range(n):
            for j in range(i + 1, n):
                matrice[i][j] = x[index]
                index += 1

        # Contraintes sur les montees
        for i in range(n - 1):
            temp_sum = 0
            for j in range(i + 1, n):
                temp_sum += matrice[i][j]
            temp_sum += -m[i]
            res += inv_esp * temp_sum * temp_sum

        # Contraintes sur les descentes
        for j in range(1, n):
            temp_sum = 0
            for i in range(0, j):
                temp_sum += matrice[i][j]
            temp_sum += -d[j]
            res += inv_esp * temp_sum * temp_sum

        return res

    # Fonction scipy
    x0 = [0.1 for _ in range(int((n - 1) * n / 2))]
    resultat = scipy.optimize.minimize(entropie_et_contraintes, x0)

    return resultat


def vect_to_matrice(x, n):
    matrice = [[0] * n for _ in range(n)]
    index = 0
    for i in range(n):
        for j in range(i + 1, n):
            matrice[i][j] = x[index]
            index += 1
    return matrice


def affiche_matrice(M):
    """
    Affiche la matrice M dans la console Python
    :param M:
    """
    for ligne in M:
        print(str(ligne) + '\n')


m5 = [2, 3, 1, 2, 0]
v5 = [0, 1, 2, 2, 3]
res = penalisation(m5, v5, 0.00001)
affiche_matrice(vect_to_matrice(res.x, 5))

m6 = [5, 4, 6, 3, 1, 0]
v6 = [0, 2, 4, 3, 5, 5]
res = penalisation(m6, v6, 0.00001)
affiche_matrice(vect_to_matrice(res.x, 6))
