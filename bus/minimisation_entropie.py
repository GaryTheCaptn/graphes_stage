import scipy.optimize
import time
import random
import numpy as np
import extraction_donnees
import matplotlib.pyplot as plt
from bus import somme_colonne, somme_ligne, lagrange_to_euler


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
    """
    :param m: un liste d'entiers (montees)
    :param v: une liste d'entiers (descentes)
    :return: les vecteurs m et v normalises.
    """
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
    """
    :param m: liste d'entier (montees)
    :param v: liste d'entier (descentes)
    :return: le vecteur initial pour la minimisation qui correspond au produit des marginales (m_i * v_j = gamma_ij)
    """
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
    """
    :param x: un vecteur d'entiers de taille d = n*(n-1)/2
    :param n: la taille de la matrice
    :return: la matrice de taille n dont les coefficients sur la partie superieure (stricte) droite
    correspond au vecteur x.
    """
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
    """
    :param n: la taille de la matrice
    :return: la matrice de contraintes A tel que Ax = m++v
    """
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
    """
    :param m: liste d'entiers de montees (non-normalise)
    :param v: listen d'entier de descentes (non-normalise)
    :return: le resultat de l'optimization par la mehtode de Scipy avec contraintes.
    """
    n = len(m)
    d = int((n - 1) * n / 2)

    # Definition de la fonction a minimiser
    def entropie(x):
        res = 0
        # Entropie
        for x_i in x:
            if x_i > 0:
                res += x_i * np.log(x_i)
        return res

    # Definition de la jacobienne associee
    def jacobian_entropie(x):
        res = []
        for x_i in x:
            if x_i > 0:
                res += [np.log(x_i) + 1]
            else:
                res += [0]
        return res

    # Definition de la hessienne associee
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

    # On recupere les informations qui nous interesse
    vect_resultat = resultat.x
    qual_resultat = qualite_resultat(vect_resultat, normalized_m, normalized_v)

    return vect_resultat, qual_resultat


# Methode de penalisaiton (algorithme perso)
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

    # Fonction scipy
    x0 = vecteur_initial(normalized_m, normalized_v)
    # resultat = scipy.optimize.minimize(entropie_et_contraintes, x0, jac=jacobian_entropie_et_contraintes)
    resultat = scipy.optimize.minimize(entropie_et_contraintes, x0)

    return resultat


def variation_epsilon(m, d):
    """
    :param m: liste d'entiers des montees
    :param d: liste d'entiers des descentes
    :return: On teste tous les epsilons de 0.1 jusqu'a ce qu'il n'y ait plus de resultat en divisant par 10 a chaque iteration.
    On renvoie le meilleur vecteur et sa qualite.
    """
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


# Gradient à pas fixe
def gradient_pas_fixe(x0, pas, itmax, erreur, fct_gradient, m, v):
    """
    :param x0: vecteur initial (taille d)
    :param pas: float
    :param itmax: nombre maximal d'iterations
    :param erreur: seuil d'erreur
    :param fct_gradient: fonction gradient associee a la fonction a minimiser
    :param m: liste d'entiers des montees
    :param v: lsite d'entiers des descentes
    :return: le vecteur resultat apres itmax iterations ou si l'erreur seuil a ete atteinte et la qualite du resutlat
    """
    # On pose l'initialisation
    res = [x0]
    iteration = 0
    qual = 0

    while iteration < itmax:
        # Si on a pas dépassé le nombre maximal d'itéreations, alors on applique l'algorithme
        xk = res[iteration]
        xk1 = np.array(xk - pas * fct_gradient(xk))
        res.append(xk1)  # On ajoute l'itération en cours à la liste des itérés.

        # On regarde si le critère d'arrêt d'être suffisammment proche de la solution est vérifié
        qual = qualite_resultat(xk1, m, v)
        erreurk1 = qual  # On calcule l'erreur
        if erreurk1 <= erreur:
            # On est suffisament proche de la solution, on arrête l'algorithme.
            iteration = itmax
        else:
            # On est pas encore assez proche, on va faire une autre itération.
            iteration += 1

    return res, qual


# Fonctions pour les tests :

# Test 1 : Comparaison qualite et temps entre les deux methodes a partir de donnees euleriennes
def generation_vecteurs_euleriens_aleatoires(nbr_voyageurs, nbr_arrets):
    """
    :param nbr_voyageurs: nombre cumule de voyageurs sur la ligne
    :param nbr_arrets: nombre d'arrets sur la ligne
    :return: une liste correspondant aux montees, une liste correspondant aux descentes
    """
    montees = [0 * nbr_arrets]
    descentes = [0 * nbr_arrets]
    arret_courant = 0
    nbr_voyageurs_courant = 0
    personnes_restantes = nbr_voyageurs

    for i in range(0, nbr_arrets - 1):

        # Si on est au dernier arret, personne ne monte et tout le monde doit descendre
        if i == nbr_arrets - 2:
            montees_i = 0
            descentes_i = nbr_voyageurs_courant

        # Si on est à l'avant-dernier arret, il faut faire monter toutes les personnes qui manquent
        elif i == nbr_arrets - 3:
            montees_i = personnes_restantes
            descentes_i = random.randint(0, nbr_voyageurs_courant)

        # Sinon, on fait monter un nombre aleatoire de personnes (parmi les nombres de personnes restantes)
        # On fait descendre un nombre aleatoire de personnes (parmi les voyageurs qui etaient dans le bus)
        else:
            montees_i = random.randint(0, personnes_restantes)
            descentes_i = random.randint(0, nbr_voyageurs_courant)

        montees.append(montees_i)
        descentes.append(descentes_i)
        personnes_restantes += -montees_i  # On enleve les personnes montees
        nbr_voyageurs_courant = nbr_voyageurs_courant + montees_i - descentes_i  # On met a jour le nombre de voyageurs

    return montees, descentes


def comparaison_methodes_qualite_temps_vect_aleatoires():
    liste_arrets = np.linspace(5, 25, 21)
    liste_arrets = list(map(int, liste_arrets))

    qualite_methode_epsilon = []
    qualite_scipy = []

    temps_epsilon = []
    temps_scipy = []

    for nbr_arrets in liste_arrets:
        print(nbr_arrets)
        m, v = generation_vecteurs_euleriens_aleatoires(nbr_arrets * 8, nbr_arrets)

        # On teste la methode scipy
        time_start_scipy = time.time()
        vect_scipy, qual_scipy = optimisation_scipy(m, v)
        qualite_scipy.append(qual_scipy)
        temps_scipy.append(time.time() - time_start_scipy)

        # On teste la methode eps
        time_start_eps = time.time()
        vect_eps, qual_eps = variation_epsilon(m, v)
        qualite_methode_epsilon.append(qual_eps)
        temps_epsilon.append(time.time() - time_start_eps)

    # Temps de calcul
    plt.scatter(liste_arrets, temps_epsilon, label='temps epsilon')
    plt.scatter(liste_arrets, temps_scipy, label='temps scipy')
    plt.title("Comparaison temps de calcul en fonction du nombre d'arrets")
    plt.legend()
    plt.show()

    # Qualite du resultat
    plt.scatter(liste_arrets, qualite_methode_epsilon, label='qualite epsilon')
    plt.scatter(liste_arrets, qualite_scipy, label='qualite scipy')
    plt.title("Comparaison qualite en fonction du nombre d'arrets")
    plt.legend()
    plt.show()


# Test 2, 3 : Comparaison (moindres carres puis entropie relative) matrices OD et matrices trouvees par les deux methodes.

def distance_moindres_carres(matrice1, matrice2):
    """
    :param matrice1: Une matrice OD
    :param matrice2: Une matrice OD
    :return: la distance entre les deux matrices coefficient par coefficient
    """
    res = 0
    n = len(matrice1)
    for i in range(n):
        for j in range(i + 1, n):  # On a pas besoin de regarder les cases en dessous de la diagonale.
            res += (matrice1 - matrice2) ** 2
    return res


def distance_entropie_relative(matriceOD, matrice2):
    """
    :param matriceOD: matrice origine destination (avec des 0 potentiellement)
    :param matrice2: matrice trouvee par optimisation
    :return: l'entropie relative S_matrice2(matrice_OD)
    """
    res = 0
    for i in range(1, len(matriceOD)):
        for j in range(i + 1, len(matriceOD)):
            if matrice2[i][j] != 0:
                res += matriceOD[i][j] * np.log(matriceOD[i][j] / matrice2[i][j])
    return res


def comparaison_mc_entropie(matriceOD):
    m, v = lagrange_to_euler(matriceOD)
    n = len(m)

    # Test methode penalisation
    time_start = time.time()
    vect_eps, qual_eps = variation_epsilon(m, v)
    time_eps = time.time() - time_start
    matrice_eps = initialise_matrice_from_vect(vect_eps, n)
    entropie_relative_eps = distance_entropie_relative(matriceOD, matrice_eps)
    distance_mc_eps = distance_moindres_carres(matriceOD, matrice_eps)
    print("L'entropie relative pour eps est " + str(
        entropie_relative_eps) + "et la distance au sens des moindres carres est de " + str(
        distance_mc_eps) + " et l'operation a pris " + str(
        time_eps) + " secondes.")
    affiche_matrice_propre(matrice_eps)

    # Test sur la methode scipy
    time_start = time.time()
    vect_scipy, qual_scipy = optimisation_scipy(m, v)
    time_scipy = time.time() - time_start
    matrice_scipy = initialise_matrice_from_vect(vect_scipy, n)
    entropie_relative_scipy = distance_entropie_relative(matriceOD, matrice_scipy)
    distance_mc_scipy = distance_moindres_carres(matriceOD, matrice_scipy)
    print("L'entropie relative pour scipy est " + str(
        entropie_relative_scipy) + "et la distance au sens des moindres carres est de " + str(
        distance_mc_scipy) + " et l'operation a pris " + str(
        time_scipy) + " secondes.")
    affiche_matrice_propre(matrice_scipy)

    return entropie_relative_eps, entropie_relative_scipy


# Fonction affichage des resultats d'optimisation
def affichage_resultat_opti(m, v, type='scipy'):
    """
    Affiche dans le terminal le resultat de l'optimisation avec la methode choisie.
    :param m: liste d'entiers montees
    :param v: liste d'entiers descentes
    :param type: scipy ou epsilon
    :return: None
    """
    time_start = time.time()
    # On recupere le resultat
    # scipy
    if type == 'scipy':
        vect_resultat, qual_resultat = optimisation_scipy(m, v)
    # epsilon
    else:
        vect_resultat, qual_resultat = variation_epsilon(m, v)

    # S'il y a un resultat
    if len(vect_resultat) > 0:
        matrice_resultat = initialise_matrice_from_vect(vect_resultat, len(m))
        affiche_matrice_propre(matrice_resultat)
        print("La qualite du resultat est de :" + '\n' + str(qual_resultat) + '\n')
        print("Duree du traitement (secondes) :")
        print(time.time() - time_start)
    else:
        print("Pas de resultat")


if __name__ == "__main__":
    # Donnees :
    m5 = [2, 3, 1, 2, 0]
    v5 = [0, 1, 2, 2, 3]
    m6 = [5, 4, 6, 3, 1, 0]
    v6 = [0, 2, 4, 3, 5, 5]

    # Test 0 : Resultats optimisation pour les vecteurs 5 et 6 / A comparer avec les resultats de la recherche exhaustive
    test0 = False
    if test0:
        print("Variation epsilon vecteur 5")
        affichage_resultat_opti(m5, v5, type='penalisation')
        print("Optimisation scipy 5 :" + '\n')
        affichage_resultat_opti(m5, v5, type='scipy')
        print(
            "__________________________________________________________________________________________________________")
        print("Variation epsilon vecteur 6")
        affichage_resultat_opti(m6, v6, type='penalisation')
        print("Optimisation scipy 6 :" + '\n')
        affichage_resultat_opti(m6, v6, type='scipy')

    # Test 1 : Comparaison de la qualite des resultats et du temps pour des vecteurs aleatoires.
    test1 = True
    if test1:
        comparaison_methodes_qualite_temps_vect_aleatoires()
        # hi

    # Test 2 : Comparaison par entropie relative et moindres carres avec donnes reelles
    test2 = False
    if test2:
        print("Hello")
        # hi
