import scipy.optimize
import numpy as np
from bus import somme_colonne, somme_ligne


def generation_matrice_contraintes(n):
    d = int((n - 1) * n / 2)

    A = [[0] * d for _ in range(2 * n)]

    # Contraintes de montee
    ligne = 0
    colonne = 0
    compteur = n - 1
    while compteur > 0:
        for _ in range(compteur):
            A[ligne][colonne] = 1
            colonne += 1
        ligne += 1
        compteur += -1

    # Contraintes de descente
    ligne += 2  # Pour m[N-1] et v[0]

    # TODO
    affiche_matrice_propre(A)


def optimisation_scipy(m, v):
    n = len(m)
    d = int((n - 1) * n / 2)

    def entropie(x):
        res = 0
        # Entropie
        for x_i in x:
            if x_i > 0:
                res += x_i * np.log(x_i)
        return res

    # Contraintes sur les montees et descentes
    montes_descentes = m + v
    A = [[0] * d for _ in range(2 * n)]

    # Contraintes de montée
    compteur_colonne = n - 1
    compteur_ligne = 0
    while compteur_colonne < 0:
        for j in range(compteur_colonne):
            A[compteur_ligne][compteur_ligne + 1 + j] = 1
            compteur_ligne += 1
            compteur_colonne += -1

    # Contraintes de descentes
    compteur_ligne = 0
    while compteur_ligne < n - 1:
        for j in range(compteur_ligne):
            A[j][1 + compteur_ligne] = 1
            compteur_ligne += 1
    print(A)
    contraintes_montes_descentes = scipy.optimize.LinearConstraint(A, montes_descentes, montes_descentes)

    # Contraintes sur les valeurs positives
    zeros = [0 for _ in range(4 * n)]
    bnds = [(0, None) for _ in range(d)]

    # Vecteur initial
    x0 = [0.1 for _ in range(d)]

    # Minimisation par la méhtode de scipy
    resultat = scipy.optimize.minimize(entropie, x0, bounds=bnds, constraints=contraintes_montes_descentes)
    print(resultat)


def initialise_matrice_from_vect(x, n):
    matrice = [[0] * n for _ in range(n)]
    index = 0
    for i in range(n):
        for j in range(i + 1, n):
            matrice[i][j] = x[index]
            index += 1
    return matrice


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
        matrice = initialise_matrice_from_vect(x, n)

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


def qualite_resultat(vect_resultat, m, d):
    N = len(m)
    matrice_resultat = initialise_matrice_from_vect(vect_resultat, N)
    dist = 0
    # Distance pour le respect des sommes sur les lignes et les colonnes
    for i in range(N):
        dist += (somme_ligne(matrice_resultat, i) - m[i]) ** 2 + (somme_colonne(matrice_resultat, i) - d[i]) ** 2

    # Distance pour le respect des valeurs >= 0
    for x_i in vect_resultat:
        if x_i < 0:
            dist += (x_i) ** 2
    return dist


def variation_epsilon(m, d):
    res_trouve = True
    best_vector = []
    best_qualite = 100000
    eps = 0.1
    while res_trouve:
        eps = eps / 10
        resultat = penalisation(m, d, eps)
        res_trouve = resultat.success
        res_vector = resultat.x
        if res_trouve:
            qualite = qualite_resultat(res_vector, m, d)
            if qualite < best_qualite:
                best_qualite = qualite
                best_vector = res_vector
    return best_vector, best_qualite


def affiche_matrice_propre(M):
    """
    Affiche la matrice M dans la console Python
    :param M:
    """
    # Déterminer la largeur maximale d'un élément du tableau pour l'alignement
    largeur_max = 5
    for ligne in M:
        # Joindre les éléments de la ligne avec un espace et les aligner à droite selon la largeur maximale
        ligne_formatee = " ".join(f"{str(round(item, 3)):>{largeur_max}}" for item in ligne)
        print(ligne_formatee)
    print('\n')
    return None


generation_matrice_contraintes(4)
testing = False
if testing:
    m5 = [2, 3, 1, 2, 0]
    v5 = [0, 1, 2, 2, 3]
    vect_res5, qualite_res5 = variation_epsilon(m5, v5)
    affiche_matrice_propre(initialise_matrice_from_vect(vect_res5, 5))
    print("La qualité du resultat est de : ")
    print(str(qualite_res5))
    print(" ")

    m6 = [5, 4, 6, 3, 1, 0]
    v6 = [0, 2, 4, 3, 5, 5]
    vect_res6, qualite_res6 = variation_epsilon(m6, v6)
    affiche_matrice_propre(initialise_matrice_from_vect(vect_res6, 6))
    print("La qualité du resultat est de : ")
    print(str(qualite_res6))
    print(" ")
