import scipy.optimize
import time
import numpy as np
from bus import somme_colonne, somme_ligne


def affiche_matrice_propre(M):
    """
    Affiche la matrice M dans la console Python
    :param M:
    """
    # Déterminer la largeur maximale d'un élément du tableau pour l'alignement
    largeur_max = 9
    for ligne in M:
        # Joindre les éléments de la ligne avec un espace et les aligner à droite selon la largeur maximale
        ligne_formatee = " ".join(f"{str(round(item, 5)):>{largeur_max}}" for item in ligne)
        print(ligne_formatee)
    print('\n')
    return None


def normalisaiton_vecteurs(m, v):
    K = sum(m)
    normalized_m = [m_i / K for m_i in m]
    normalized_v = [v_i / K for v_i in v]
    return normalized_m, normalized_v


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


def vecteur_initial(m, v):
    n = len(m)
    d = int((n - 1) * n / 2)
    x0 = [0 for _ in range(d)]
    matrice_numeros = generation_matrice_numeros(n)
    for i in range(0, n - 1):
        for j in range(i + 1, n):
            index = matrice_numeros[i][j]
            x0[index] = m[i] * v[j]
    return x0


def initialise_matrice_from_vect(x, n):
    matrice = [[0] * n for _ in range(n)]
    index = 0
    for i in range(n):
        for j in range(i + 1, n):
            matrice[i][j] = x[index]
            index += 1
    return matrice


def qualite_resultat(vect_resultat, m, v):
    """
    :param vect_resultat: un vecteur avec les resultats d'une optimisation
    :param m: les montees (non-normalises)
    :param v: les descentes (non-normalises)
    :return: la qualite du resultat vis-a-vis du respect des contraintes
    """
    N = len(m)
    matrice_resultat = initialise_matrice_from_vect(vect_resultat, N)
    normalized_m, normalized_v = normalisaiton_vecteurs(m, v)
    dist = 0
    # Distance pour le respect des sommes sur les lignes et les colonnes
    for i in range(N):
        dist += (somme_ligne(matrice_resultat, i) - normalized_m[i]) ** 2 + (
                somme_colonne(matrice_resultat, i) - normalized_v[i]) ** 2

    # Distance pour le respect des valeurs >= 0
    for x_i in vect_resultat:
        if x_i < 0:
            dist += (x_i) ** 2
    return dist


# Methode Trust-Region Constrained Algorithm de Scipy

def generation_matrice_contraintes(n):
    d = int((n - 1) * n / 2)

    # Matrice numéros
    matrice_numeros = generation_matrice_numeros(n)

    # Matrice vierge pour les contraintes
    A = [[0] * d for _ in range(2 * n)]

    # Contraintes de montees
    for i in range(0, n):

        for j in range(1, n):
            temp_index = matrice_numeros[i][j]
            if temp_index != -1:
                A[i][temp_index] = 1

    # Contraintes de descentes
    ligne_descente = n
    for j in range(0, n):
        for i in range(0, n):
            temp_index = matrice_numeros[i][j]
            if temp_index != -1:
                A[ligne_descente + j][temp_index] = 1

    return A


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

    def jacobian_entropie(x):
        res = []
        for x_i in x:
            res += [np.log(x_i) + 1]
        return res

    def hessian_entropie(x):
        res = []
        for i in range(len(x)):
            ligne_res = []
            for j in range(len(x)):
                if i != j:
                    ligne_res.append(0)
                else:
                    ligne_res.append(1 / x[i])
            res.append(ligne_res)
        return res

    # Contraintes sur les montees et descentes
    normalized_m, normalized_v = normalisaiton_vecteurs(m, v)
    montes_descentes = normalized_m + normalized_v
    matrice_contraintes = generation_matrice_contraintes(n)
    contraintes_montes_descentes = scipy.optimize.LinearConstraint(matrice_contraintes, montes_descentes,
                                                                   montes_descentes)

    # Contraintes sur les valeurs positives
    zeros = [0 for _ in range(4 * n)]
    bnds = [(0, None) for _ in range(d)]

    # Vecteur initial : produit des marginales
    x0 = vecteur_initial(normalized_m, normalized_v)

    # Minimisation par la méhtode de scipy
    resultat = scipy.optimize.minimize(entropie, x0, jac=jacobian_entropie, hess=hessian_entropie,
                                       method='trust-constr', bounds=bnds,
                                       constraints=contraintes_montes_descentes)
    return resultat


# Methode de penalisaiton (algorithme personnel)

def gradient_pas_fixe(x0, pas, itmax, erreur, fct_gradient, m, v):
    # On pose l'initialisation
    res = [x0]
    iteration = 0

    while iteration < itmax:
        # Si on a pas dépassé le nombre maximal d'itéreations, alors on applique l'algorithme
        xk = res[iteration]
        xk1 = np.array(xk - pas * fct_gradient(xk))
        res.append(xk1)  # On ajoute l'itération en cours à la liste des itérés.

        # On regarde si le critère d'arrêt d'être suffisammment proche de la solution est vérifié
        erreurk1 = qualite_resultat(xk1, m, v)  # On calcule l'erreur
        if erreurk1 <= erreur:
            # On est suffisament proche de la solution, on arrête l'algorithme.
            iteration = itmax
        else:
            # On est pas encore assez proche, on va faire une autre itération.
            iteration += 1

    return res


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


def penalisation(m, v, eps):
    """
    :param m: liste des montees (contrainte)
    :param v: liste des descentes (contrainte)
    :param eps: valeur de la penalisation
    :return: la matrice OD minimisant l'entropie par la methode de penalisation
    """
    n = len(m)
    inv_esp = 1 / eps
    normalized_m, normalized_v = normalisaiton_vecteurs(m, v)

    # Definition de la fonction a minimiser avec l'ajout des contraintes
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
            i, j = index_ligne_colonne(index, matrice_numeros)
            liste_numeros_ligne = liste_numeros_meme_ligne(i, matrice_numeros)
            somme_ligne = 0
            for k in liste_numeros_ligne:
                somme_ligne += x[k]
            somme_ligne = 2 * inv_esp * (somme_ligne - normalized_m[i])

            liste_numeros_colonne = liste_numeros_meme_colonne(j, matrice_numeros)
            somme_colonne = 0
            for k in liste_numeros_colonne:
                somme_colonne += x[k]
            somme_colonne = 2 * inv_esp * (somme_colonne - normalized_v[j])
            res.append(x_i * np.log(
                x_i) + somme_ligne + somme_colonne)
            index += 1
        return res

    # Definition de la hessienne
    def hessian_entropie_et_contraintes(x):
        res = []
        matrice_numeros = generation_matrice_numeros(n)
        for i in range(0, len(x)):
            res_i = []
            liste_numeros_ligne = liste_numeros_meme_ligne(i, matrice_numeros)
            liste_numeros_colonne = liste_numeros_meme_colonne(i, matrice_numeros)
            for j in range(0, len(x)):
                temp = 0
                if i == j:
                    temp += 1 / x[i]
                if j in liste_numeros_ligne:
                    temp += 2 * inv_esp
                if j in liste_numeros_colonne:
                    temp += 2 * inv_esp
                res_i.append(temp)
            res.append(res_i)

    # Fonction scipy
    x0 = vecteur_initial(normalized_m, normalized_v)
    resultat = scipy.optimize.minimize(entropie_et_contraintes, x0, jac=jacobian_entropie_et_contraintes,
                                       hess=hessian_entropie_et_contraintes)

    return resultat


def variation_epsilon(m, d):
    res_trouve = True
    best_vector = []
    best_qualite = 100000
    eps = 1
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


# Fonction affichage des resultats


def affichage_resultat_opti(m, v, type='scipy'):
    time_start = time.time()
    if type == 'scipy':
        resultat = optimisation_scipy(m, v)
        vect_resultat = resultat.x
        qual_resultat = qualite_resultat(vect_resultat, m, v)
    else:
        vect_resultat, qual_resultat = variation_epsilon(m, v)
    if len(vect_resultat) > 0:
        matrice_resultat = initialise_matrice_from_vect(vect_resultat, len(m))
        affiche_matrice_propre(matrice_resultat)
        print("La qualite du resultat est de :" + '\n' + str(qual_resultat) + '\n')
        print("Duree du traitement (secondes) :")
        print(time.time() - time_start)
    else:
        print("Pas de resultat")


if __name__ == "__main__":
    m5 = [2, 3, 1, 2, 0]
    v5 = [0, 1, 2, 2, 3]

    print("Variation epsilon vecteur 5")
    affichage_resultat_opti(m5, v5, type='penalisation')
    print("Optimisation scipy 5 :" + '\n')
    affichage_resultat_opti(m5, v5, type='scipy')
    print("__________________________________________________________________________________________________________")
    m6 = [5, 4, 6, 3, 1, 0]
    v6 = [0, 2, 4, 3, 5, 5]
    print("Variation epsilon vecteur 6")
    affichage_resultat_opti(m6, v6, type='penalisation')
    print("Optimisation scipy 6 :" + '\n')
    affichage_resultat_opti(m6, v6, type='scipy')
    print("__________________________________________________________________________________________________________")
    mA = [40, 37, 38, 39, 45, 36, 35, 50, 38, 50, 55, 35, 35, 32, 0]
    vA = [0, 33, 35, 38, 42, 35, 33, 52, 40, 47, 49, 38, 40, 45, 38]
    mA = [494, 292, 403, 176, 670, 358, 242, 1268, 152, 535, 693, 118, 10, 43, 0]
    vA = [0, 7, 35, 21, 157, 70, 76, 726, 330, 820, 927, 309, 386, 1128, 470]

    print("Variation epsilon ligne A")
    affichage_resultat_opti(mA, vA, type='penalisation')

    print("Optimisation scipy ligne A :" + '\n')
    affichage_resultat_opti(mA, vA, type='scipy')
