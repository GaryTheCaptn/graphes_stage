import scipy.optimize
import time
import random
import pickle
import os
import numpy as np
from extraction_donnees import extraction_donnees
import matplotlib.pyplot as plt
from bus import lagrange_to_euler


def affiche_matrice_propre(M):
    """
    Affiche la matrice M dans la console Python
    :param M: une matrice
    """
    # Determiner la largeur maximale d'un element du tableau pour l'alignement
    largeur_max = 9
    for ligne in M:
        # Joindre les elements de la ligne avec un espace et les aligner a droite selon la largeur maximale
        ligne_formatee = " ".join(f"{str(round(item, 5)):>{largeur_max}}" for item in ligne)
        print(ligne_formatee)
    print('\n')
    return None


def normalisation_vecteurs(m, v):
    """
    :param m: un liste d'entiers (montees)
    :param v: une liste d'entiers (descentes)
    :return: les vecteurs m et v normalises.
    """
    K = sum(m)
    invK = 1 / K
    normalized_m = [m_i * invK for m_i in m]
    normalized_v = [v_i * invK for v_i in v]
    return normalized_m, normalized_v


def normalisation_matrice(matrice):
    """
    :param matrice: une matrice de FLOAT
    :return: matrice normalisee
    """
    K = np.sum(matrice)
    invK = 1 / K

    # Copie la partie superieure de la matrice
    matrice_normalisee = np.triu(matrice, 1)

    # Normalisation de la partie superieure de la matrice
    matrice_normalisee *= invK

    return matrice_normalisee


def generation_matrice_numeros(n):
    """
    :param n: entier, taille de la matrice
    :return: une matrice avec les index de chaque case pour le vecteur colonne associe et des -1 pour les autres
    """
    m = np.full((n, n), -1, dtype=int)
    index = 0
    for i in range(n):
        for j in range(i + 1, n):
            m[i, j] = index
            index += 1
    return m


def vecteur_initial(m, v, matrice_numeros, n):
    """
    :param m: liste d'entier (montees)
    :param v: liste d'entier (descentes)
    :param matrice_numeros: la matrice avec les indices de vecteur
    :param n: taille des listes
    :return: le vecteur initial pour la minimisation qui correspond au produit des marginales (m_i * v_j = gamma_ij)
    """
    d = int((n - 1) * n / 2)
    x0 = [0] * d
    for i in range(n - 1):
        for j in range(i + 1, n):
            index = matrice_numeros[i][j]
            x0[index] = m[i] * v[j]
    return x0


def initialise_matrice_from_vect(x, n):
    """
    :param x: un vecteur d'entiers de taille d = n*(n-1)/2
    :param n: la taille de la matrice
    :return: la matrice de taille n dont les coefficients sur la partie superieure (stricte) droite
    correspondent au vecteur x.
    """
    matrice = np.zeros((n, n), dtype=float)
    index = 0
    for i in range(n):
        for j in range(i + 1, n):
            matrice[i][j] = x[index]
            index += 1
    return matrice


def qualite_resultat(vect_resultat, m, v, n):
    """
    :param vect_resultat: un vecteur avec les resultats d'une optimisation
    :param m: les montees normalisees
    :param v: les descentes normalisees
    :param n: la taille de la matrice
    :return: la qualite du resultat vis-a-vis du respect des contraintes
    """
    matrice_resultat = initialise_matrice_from_vect(vect_resultat, n)
    dist = 0

    # Convertir les listes m et v en tableaux NumPy pour les operations vectorisees
    m = np.array(m)
    v = np.array(v)

    # Calcul de la distance pour le respect des sommes sur les lignes et les colonnes
    somme_lignes = np.sum(matrice_resultat, axis=1)
    somme_colonnes = np.sum(matrice_resultat, axis=0)
    dist += np.sum((somme_lignes - m) ** 2) + np.sum((somme_colonnes - v) ** 2)

    # Distance pour le respect des valeurs >= 0
    vect_resultat = np.array(vect_resultat)
    dist += np.sum(vect_resultat[vect_resultat < 0] ** 2)

    return dist


# Methode Trust-Region Constrained Algorithm de Scipy

def generation_matrice_contraintes(n, matrice_numeros):
    """
    :param n: la taille de la matrice
    :param matrice_numeros: la matrice avec les numeros vecteur
    :return: la matrice de contraintes A tel que Ax = m++v
    """
    d = (n - 1) * n // 2

    # Matrice vierge pour les contraintes
    a = np.zeros((2 * n, d), dtype=int)

    # Contraintes de montees
    for i in range(n):
        for j in range(i + 1, n):
            temp_index = matrice_numeros[i][j]
            if temp_index != -1:
                a[i, temp_index] = 1

    # Contraintes de descentes
    for j in range(n):
        for i in range(j):
            temp_index = matrice_numeros[i][j]
            if temp_index != -1:
                a[n + j, temp_index] = 1

    return a


def optimisation_scipy(m, v, n):
    """
    :param m: liste des montees
    :param v: liste des descentes
    :param n: la taille des listes
    :return: le resultat de la minimisation de l'entropie pour la methode scipy avec contraintes.
    """
    d = (n - 1) * n // 2

    def entropie(x):
        res = np.sum(x[x > 0] * np.log(x[x > 0]))
        return res

    def jacobian_entropie(x):
        res = np.zeros_like(x)
        positive_indices = x > 0
        res[positive_indices] = np.log(x[positive_indices]) + 1
        return res

    def hessian_entropie(x):
        d = len(x)
        res = np.zeros((d, d))
        positive_indices = x > 0
        res[np.diag_indices_from(res)] = np.where(positive_indices, 1 / x, 0)
        return res

    # Matrice avec les numeros d'indices pour le passage de la vue vectorielle a la vue matricielle
    matrice_numeros = generation_matrice_numeros(n)

    # On cree la matrice de contraintes.
    montes_descentes = np.concatenate((m, v))
    matrice_contraintes = generation_matrice_contraintes(n, matrice_numeros)
    contraintes_montes_descentes = scipy.optimize.LinearConstraint(matrice_contraintes, montes_descentes,
                                                                   montes_descentes)

    # On defini les bornes des valeurs ([0 ; +inf[)
    bnds = [(0, None) for _ in range(d)]

    # Vecteur initial, croisement des marginales
    x0 = vecteur_initial(m, v, matrice_numeros, n)

    # Calcul du resultat par Scipy
    resultat = scipy.optimize.minimize(entropie, x0, jac=jacobian_entropie, hess=hessian_entropie,
                                       method='trust-constr', bounds=bnds,
                                       constraints=contraintes_montes_descentes)
    vect_resultat = resultat.x
    qual_resultat = qualite_resultat(vect_resultat, m, v, n)

    return vect_resultat, qual_resultat


# Methode de penalisaiton
def index_ligne_colonne(index, matrice_numeros, n):
    """
    :param index: l'index d'un x_k dans le vecteur x
    :param matrice_numeros: matrice avec les index de x, -1 sinon.
    :param n: taille de la matrice
    :return: les indices i(ligne) et j (colonne) ou se trouve l'element x_i dans la matrice numeros
    """
    # Convertir la matrice en un array NumPy
    matrice_numeros = np.array(matrice_numeros)

    # Trouver les indices ou la valeur est egale a l'index
    result = np.where(matrice_numeros == index)

    if result[0].size > 0 and result[1].size > 0:
        return result[0][0], result[1][0]
    else:
        return -1, -1


def liste_numeros_meme_ligne(i, matrice_numeros, n):
    """
    :param i: un numero de ligne
    :param matrice_numeros: matrice avec les index de x, -1 sinon.
    :param n: taille de la matrice
    :return: la liste des indices qui sont sur la ligne i.
    """
    # Convertir la matrice en un array NumPy
    matrice_numeros = np.array(matrice_numeros)

    # Selectionner la ligne i a partir de la matrice et ignorer les valeurs -1
    ligne_i = matrice_numeros[i, i + 1:n]

    # Filtrer les valeurs -1
    res = ligne_i[ligne_i != -1]

    return list(res)


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


def penalisation(m, v, eps, n):
    """
    :param m: liste des montees (normalise)
    :param v: liste des descentes (normalise)
    :param eps: valeur de la penalisation
    :param n: longueur des listes
    :return: la matrice OD minimisant l'entropie par la methode de penalisation
    """
    inv_eps = 1 / eps
    matrice_numeros = generation_matrice_numeros(n)

    # Definition de la fonction a minimiser avec l'ajout des contraintes
    def entropie_et_contraintes(x):
        res = 0
        x_pos = x[x > 0]
        res += np.sum(x_pos * np.log(x_pos))

        matrice = initialise_matrice_from_vect(x, n)
        for i in range(n - 1):
            indices_ligne = matrice_numeros[i, i + 1:n]
            sum_ligne = np.sum(x[indices_ligne[indices_ligne != -1]])
            res += inv_eps * (sum_ligne - m[i]) ** 2

        for j in range(1, n):
            indices_colonne = matrice_numeros[0:j, j]
            sum_colonne = np.sum(x[indices_colonne[indices_colonne != -1]])
            res += inv_eps * (sum_colonne - v[j]) ** 2

        res += np.sum(np.maximum(-inv_eps * x, 0) ** 2)
        return res

    # Definition de la jacobienne
    def jacobian_entropie_et_contraintes(x):
        res = []
        index = 0
        for x_i in x:
            val = 0
            if x_i > 0:
                val += np.log(x_i) + 1
            i, j = index_ligne_colonne(index, matrice_numeros, n)
            liste_numeros_ligne = liste_numeros_meme_ligne(i, matrice_numeros, n)
            somme_ligne = 0
            for k in liste_numeros_ligne:
                somme_ligne += x[k]
            somme_ligne = 2 * inv_eps * (somme_ligne - m[i])
            val += somme_ligne

            liste_numeros_colonne = liste_numeros_meme_colonne(j, matrice_numeros)
            somme_colonne = 0
            for k in liste_numeros_colonne:
                somme_colonne += x[k]
            somme_colonne = 2 * inv_eps * (somme_colonne - v[j])
            val += somme_colonne
            res.append(val)
            index += 1
        return res

    # Fonction scipy
    x0 = vecteur_initial(m, v, matrice_numeros, n)
    # resultat = scipy.optimize.minimize(entropie_et_contraintes, x0, jac=jacobian_entropie_et_contraintes)
    resultat = scipy.optimize.minimize(entropie_et_contraintes, x0)

    return resultat


def variation_epsilon(m, d, n):
    """
    :param m: liste d'entiers des montees (normalise)
    :param d: liste d'entiers des descentes (normalise
    :param n: la longueur des listes
    :return: On renvoie le vecteur resultat et sa qualite. Si pas de resultat on renvoie un vecteur vide et une qualite
     = -1
    """
    qualite = -1
    eps = 0.01
    resultat = penalisation(m, d, eps, n)
    res_trouve = resultat.success
    vector = resultat.x
    if res_trouve:
        qualite = qualite_resultat(vector, m, d, n)
    return vector, qualite


# Gradient a pas fixe (pas utilise)
def gradient_pas_fixe(x0, pas, itmax, erreur, fct_gradient, m, v):
    """
    :param x0: vecteur initial (taille d)
    :param pas: float
    :param itmax: nombre maximal d'iterations
    :param erreur: seuil d'erreur
    :param fct_gradient: fonction gradient associee a la fonction a minimiser
    :param m: liste d'entiers des montees (normalise)
    :param v: lsite d'entiers des descentes (normalise)
    :return: le vecteur resultat apres itmax iterations ou si l'erreur seuil a ete atteinte et la qualite du resutlat
    """
    # On pose l'initialisation
    res = [x0]
    iteration = 0
    qual = 0
    n = len(m)

    while iteration < itmax:
        # Si on a pas depasse le nombre maximal d'itereations, alors on applique l'algorithme
        xk = res[iteration]
        xk1 = np.array(xk - pas * fct_gradient(xk))
        res.append(xk1)  # On ajoute l'iteration en cours a la liste des iteres.

        # On regarde si le critere d'arret d'etre suffisammment proche de la solution est verifie
        qual = qualite_resultat(xk1, m, v, n)
        erreurk1 = qual  # On calcule l'erreur
        if erreurk1 <= erreur:
            # On est suffisament proche de la solution, on arrete l'algorithme.
            iteration = itmax
        else:
            # On est pas encore assez proche, on va faire une autre iteration.
            iteration += 1

    return res, qual


# Fonctions pour les tests :
def distance_moindres_carres(matrice1, matrice2, n):
    """
    :param matrice1: Une matrice OD
    :param matrice2: Une matrice OD
    :param n: taille des matrices
    :return: la distance entre les deux matrices coefficient par coefficient
    """
    matrice1 = np.array(matrice1)
    matrice2 = np.array(matrice2)

    # Calcul des differences
    differences = matrice1 - matrice2

    # Calcul de la somme des carres des differences au-dessus de la diagonale principale
    res = np.sum(differences[np.triu_indices(n, k=1)] ** 2)

    return res


def distance_entropie_relative(matriceOD, matrice2, n):
    """
    :param matriceOD: matrice origine destination (avec des 0 potentiellement)
    :param matrice2: matrice trouvee par optimisation
    :param n: taille des matrices
    :return: l'entropie relative S_matrice2(matrice_OD)
    """
    res = 0
    for i in range(n):
        for j in range(i + 1, n):
            if matrice2[i][j] > 0 and matriceOD[i][j] > 0:
                res += matriceOD[i][j] * np.log(matriceOD[i][j] / matrice2[i][j])
    return res


# Test 1 : Comparaison qualite et temps entre les deux methodes a partir de donnees euleriennes
def generation_vecteurs_euleriens_aleatoires(nbr_voyageurs, nbr_arrets):
    """
    :param nbr_voyageurs: nombre cumule de voyageurs sur la ligne
    :param nbr_arrets: nombre d'arrets sur la ligne
    :return: une liste correspondant aux montees, une liste correspondant aux descentes
    """
    montees = []
    descentes = []
    arret_courant = 0
    nbr_voyageurs_courant = 0
    personnes_restantes = nbr_voyageurs

    for i in range(0, nbr_arrets):

        # Si on est au dernier arret, personne ne monte et tout le monde doit descendre
        if i == nbr_arrets - 1:
            montees_i = 0
            descentes_i = nbr_voyageurs_courant

        # Si on est a l'avant-dernier arret, il faut faire monter toutes les personnes qui manquent
        elif i == nbr_arrets - 2:
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
    liste_arrets = np.linspace(8, 10, 3)
    liste_arrets = list(map(int, liste_arrets))
    temps_eps = []
    qualite_eps = []

    temps_scipy = []
    qualite_scipy = []

    for nbr_arrets in liste_arrets:

        # On defini les listes temporaires pour faire la moyenne pour nbr_arrets
        temps_eps_temp = []
        qualite_eps_temp = []

        temps_scipy_temp = []
        qualite_scipy_temp = []

        for i in range(5):
            print(str(nbr_arrets) + "." + str(i))
            m, v = generation_vecteurs_euleriens_aleatoires(nbr_arrets * 8, nbr_arrets)
            m, v = normalisation_vecteurs(m, v)

            # On teste la methode eps
            time_start_eps = time.time()
            vect_eps, qual_eps = variation_epsilon(m, v, nbr_arrets)
            qualite_eps_temp.append(qual_eps)
            temps_eps_temp.append(time.time() - time_start_eps)

            # On teste la methode scipy
            time_start_scipy = time.time()
            vect_scipy, qual_scipy = optimisation_scipy(m, v, nbr_arrets)
            qualite_scipy_temp.append(qual_scipy)
            temps_scipy_temp.append(time.time() - time_start_scipy)

        temps_eps.append(temps_eps_temp)
        qualite_eps.append(qualite_eps_temp)

        temps_scipy.append(temps_scipy_temp)
        qualite_scipy.append(qualite_scipy_temp)

    # Temps de calcul
    plt.figure(figsize=(12, 6))
    positions_group1 = np.array(range(len(liste_arrets))) * 2.0 - 0.3
    positions_group2 = np.array(range(len(liste_arrets))) * 2.0 + 0.3

    # Boxplots pour le premier groupe
    plt.boxplot(temps_eps, positions=positions_group1, widths=0.4, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='blue'),
                medianprops=dict(color='blue'),
                whiskerprops=dict(color='blue'),
                capprops=dict(color='blue'),
                showfliers=False)

    # Boxplots pour le deuxieme groupe
    plt.boxplot(temps_scipy, positions=positions_group2, widths=0.4, patch_artist=True,
                boxprops=dict(facecolor='lightgreen', color='green'),
                medianprops=dict(color='green'),
                whiskerprops=dict(color='green'),
                capprops=dict(color='green'),
                showfliers=False)

    # Ajustement des labels et des positions des axes
    plt.xticks(range(0, len(liste_arrets) * 2, 2), liste_arrets)
    plt.xlabel('Nombre arrets')
    plt.ylabel('Temps de traitement')
    plt.title('Boxplot pour le temps de traitement en fonction du nombre d arrets et de la methode')
    plt.grid(True)
    plt.legend([plt.Line2D([0], [0], color='lightblue', lw=4),
                plt.Line2D([0], [0], color='lightgreen', lw=4)],
               ['Penalisation', 'Scipy'])

    plt.show()

    # Temps de calcul
    plt.figure(figsize=(12, 6))
    positions_group1 = np.array(range(len(liste_arrets))) * 2.0 - 0.3
    positions_group2 = np.array(range(len(liste_arrets))) * 2.0 + 0.3

    # Boxplots pour le premier groupe
    plt.boxplot(qualite_eps, positions=positions_group1, widths=0.4, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='blue'),
                medianprops=dict(color='blue'),
                whiskerprops=dict(color='blue'),
                capprops=dict(color='blue'),
                showfliers=False)

    # Boxplots pour le deuxieme groupe
    plt.boxplot(qualite_scipy, positions=positions_group2, widths=0.4, patch_artist=True,
                boxprops=dict(facecolor='lightgreen', color='green'),
                medianprops=dict(color='green'),
                whiskerprops=dict(color='green'),
                capprops=dict(color='green'),
                showfliers=False)

    # Ajustement des labels et des positions des axes
    plt.xticks(range(0, len(liste_arrets) * 2, 2), liste_arrets)
    plt.xlabel('Nombre arrets')
    plt.ylabel('Qualite resultat')
    plt.title('Boxplot pour la qualite du resultat en fonction du nombre d arrets et de la methode')
    plt.grid(True)
    plt.legend([plt.Line2D([0], [0], color='lightblue', lw=4),
                plt.Line2D([0], [0], color='lightgreen', lw=4)],
               ['Penalisation', 'Scipy'])

    plt.show()


# Test 2, 3 : Comparaison (moindres carres puis entropie relative) matrices OD et matrices trouvees par les deux methodes.
def comparaison_mc_entropie(matriceOD, name):
    matriceOD = normalisation_matrice(matriceOD)
    m, v = lagrange_to_euler(matriceOD)
    n = len(m)
    dossier = "C:/Users/garan/Documents/Stage L3/Code/bus/resultats_minimisation/"

    # Test sur le vecteur marginales croisees
    x0 = vecteur_initial(m, v, generation_matrice_numeros(n), n)
    matrice_vecteur_marginale = initialise_matrice_from_vect(x0, n)
    entropie_relative_vecteur_marginales = distance_entropie_relative(matriceOD, matrice_vecteur_marginale, n)
    mc_vecteur_marginales = distance_moindres_carres(matriceOD, matrice_vecteur_marginale, n)
    print("Pour le vecteur croisement des marginales :")
    print("Entropie relative : " + str(entropie_relative_vecteur_marginales))
    print("Distance moindres carres : " + str(mc_vecteur_marginales))
    print("------" + '\n')

    # Test des methodes de minimalisation den l'entropie

    # Test methode penalisation
    time_start = time.time()
    vect_eps, qual_eps = variation_epsilon(m, v, n)
    time_eps = time.time() - time_start

    # On verifie s'il y a bien un resultat
    if len(vect_eps) > 0:
        # On initialise la matrice a partir du vecteur trouve par la minimisation
        matrice_eps = initialise_matrice_from_vect(vect_eps, n)

        # On sauvegarde la matrice dans un pickle
        # Sauvegarder la matrices
        nom_fichier = name + '_eps' + '.pkl'
        with open(os.path.join(dossier, nom_fichier), 'wb') as f:
            pickle.dump(matrice_eps, f)

        # Calcul des distances (entropie relative et distance moindres carres)
        entropie_relative_eps = distance_entropie_relative(matriceOD, matrice_eps, n)
        distance_mc_eps = distance_moindres_carres(matriceOD, matrice_eps, n)

        # Affichage du resultat
        print("Resultats methode penalisation :")
        print("Entropie relative : " + str(entropie_relative_eps))
        print("Distance moindres carres " + str(distance_mc_eps))
        print("Duree : " + str(time_eps))
        affiche_matrice_propre(matrice_eps)
    else:
        print("Pas de resultats")
        print("Duree traitement : " + str(time_eps))

    # Test sur la methode scipy
    time_start = time.time()
    vect_scipy, qual_scipy = optimisation_scipy(m, v, n)
    time_scipy = time.time() - time_start

    # On verifie s'il y a un resultat
    if len(vect_scipy) > 0:
        # On initialise la matrice a partir du vecteur trouve
        matrice_scipy = initialise_matrice_from_vect(vect_scipy, n)

        # Sauvegarder la matrices
        nom_fichier = name + '_scipy' + '.pkl'
        with open(os.path.join(dossier, nom_fichier), 'wb') as f:
            pickle.dump(matrice_scipy, f)

        # On calcule les distances (entropie relative et moindres carres)
        entropie_relative_scipy = distance_entropie_relative(matriceOD, matrice_scipy, n)
        distance_mc_scipy = distance_moindres_carres(matriceOD, matrice_scipy, n)

        # Affichage du resultat
        print("Resultats methode Scipy :")
        print("Entropie relative : " + str(entropie_relative_scipy))
        print("Distance moindres carres " + str(distance_mc_scipy))
        print("Duree : " + str(time_scipy))
        affiche_matrice_propre(matrice_scipy)
    else:
        print("Pas de resultats")
        print("Duree traitement : " + str(time_scipy))

    return None


# Fonction affichage des resultats d'optimisation
def affichage_resultat_opti(m, v, type='scipy'):
    """
    Affiche dans le terminal le resultat de l'optimisation avec la methode choisie (normalise si les vect en entree sont
    normalises)
    :param m: liste d'entiers montees (normalise)
    :param v: liste d'entiers descentes (normalise)
    :param type: scipy ou epsilon
    :return: None
    """
    time_start = time.time()
    n = len(m)  # Ok
    # On recupere le resultat
    # scipy
    if type == 'scipy':
        vect_resultat, qual_resultat = optimisation_scipy(m, v, n)
    # epsilon
    else:
        vect_resultat, qual_resultat = variation_epsilon(m, v, n)

    # S'il y a un resultat
    if len(vect_resultat) > 0:
        matrice_resultat = initialise_matrice_from_vect(vect_resultat, n)
        affiche_matrice_propre(matrice_resultat)
        print("La qualite du resultat est de :" + '\n' + str(qual_resultat) + '\n')
        print("Duree du traitement (secondes) :")
        print(time.time() - time_start)
    else:
        print("Pas de resultat")


if __name__ == "__main__":
    # Donnees :

    # Test 0 : Resultats optimisation pour les vecteurs 5 et 6 / A comparer avec les resultats de la recherche exhaustive
    test0 = False
    if test0:
        matrice_entropie5 = [[0.0, 1.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0, 1.0, 1.0],
                             [0.0, 0.0, 0.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0, 2.0],
                             [0.0, 0.0, 0.0, 0.0, 0.0]]
        print("Resultats ligne a 5 arrets")
        comparaison_mc_entropie(matrice_entropie5, name='matrice_entropie5')
        print(
            "__________________________________________________________________________________________________________")
        matrice_entropie6 = [[0.0, 2.0, 2.0, 0.0, 0.0, 1.0],
                             [0.0, 0.0, 2.0, 1.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                             [0.0, 0.0, 0.0, 0.0, 2.0, 1.0],
                             [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        print("Resultats ligne a 6 arrets")
        comparaison_mc_entropie(matrice_entropie6, name='matrice_entropie6')
        print(
            "__________________________________________________________________________________________________________")

    # Test 1 : Comparaison de la qualite des resultats et du temps pour des vecteurs aleatoires.
    test1 = True
    if test1:
        comparaison_methodes_qualite_temps_vect_aleatoires()

    # Test 2 : Comparaison par entropie relative et moindres carres avec donnes reelles
    test2 = False
    if test2:
        print("Resultats ligne A JOB 8h45-8h59" + "\n")
        path_AJOB = 'C:/Users/garan/Documents/Stage L3/Code/bus/donnees/LA_JOB.xlsx'
        sheet_AJOB = 'LAS2_trhor15=t_0845-0859'
        usecols_AJOB = 'C:Q'
        firstrow_AJOB = 7
        matrice_AJOB, noms_AJOB = extraction_donnees(path_AJOB, sheet_AJOB, usecols_AJOB, firstrow_AJOB)
        comparaison_mc_entropie(matrice_AJOB, name='matrice_AJOB')
        print("-------------------------------------------------------------------------------------------------------")
        print("Resultats ligne A Samedi 8h45-8h59" + "\n")
        path_ASam = 'C:/Users/garan/Documents/Stage L3/Code/bus/donnees/LA_SAMEDI.xlsx'
        sheet_ASam = 'LAS2_trhor15=t_0845-0859'
        usecols_ASam = 'C:Q'
        firstrow_ASam = 7
        matrice_ASam, noms_ASam = extraction_donnees(path_ASam, sheet_ASam, usecols_ASam, firstrow_ASam)
        comparaison_mc_entropie(matrice_ASam, name='matrice_ASam')
        print("-------------------------------------------------------------------------------------------------------")
        print("Resultats ligne B JOB 8h45-8h59" + " \n")
        path_BJOB = 'C:/Users/garan/Documents/Stage L3/Code/bus/donnees/LB_JOB.xlsx'
        sheet_BJOB = 'LBS2_trhor15=t_0845-0859'
        usecols_BJOB = 'C:Q'
        firstrow_BJOB = 7
        matrice_BJOB, noms_BJOB = extraction_donnees(path_BJOB, sheet_BJOB, usecols_BJOB, firstrow_BJOB)
        comparaison_mc_entropie(matrice_BJOB, name='matrice_BJOB')
        print("-------------------------------------------------------------------------------------------------------")
        print("Resultats ligne C4 JOB 8h45-8h59" + " \n")
        path_C4JOB = 'C:/Users/garan/Documents/Stage L3/Code/bus/donnees/LC4_JOB.xlsx'
        sheet_C4JOB = 'LC4S2_trhor15=t_0845-0859'
        usecols_C4JOB = 'C:AK'
        firstrow_C4JOB = 7
        matrice_C4JOB, noms_C4JOB = extraction_donnees(path_C4JOB, sheet_C4JOB, usecols_C4JOB, firstrow_C4JOB)
        comparaison_mc_entropie(matrice_C4JOB, name='matrice_C4JOB')
        print("-------------------------------------------------------------------------------------------------------")
        print("Resultats ligne C7 JOB 8h45-8h59" + " \n")
        path_C7JOB = 'C:/Users/garan/Documents/Stage L3/Code/bus/donnees/LC7_JOB.xlsx'
        sheet_C7JOB = 'LC7S1_trhor15=t_0845-0859'
        usecols_C7JOB = 'C:U'
        firstrow_C7JOB = 7
        matrice_C7JOB, noms_C7JOB = extraction_donnees(path_C7JOB, sheet_C7JOB, usecols_C7JOB, firstrow_C7JOB)
        comparaison_mc_entropie(matrice_C7JOB, name='matrice_C7JOB')
        print("-------------------------------------------------------------------------------------------------------")
