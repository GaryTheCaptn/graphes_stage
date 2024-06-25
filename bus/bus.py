import numpy as np


# Passage de donnees lagrangiennes a euleriennes
def somme_ligne(M, i):
    """
    :param M: Une matrice d'entiers
    :param i: une entier correspondant a une ligne
    :return: la somme des valeurs de toutes les coefficients de la ligne i
    """
    res = 0
    for j in range(0, len(M)):
        res += M[i][j]
    return res


def somme_colonne(M, j):
    """
    :param M: Une matrice d'entiers
    :param j: Un entier correspondant a une colonne
    :return: La somme des coefficients de la colonne j
    """
    res = 0
    for i in range(0, len(M)):
        res += M[i][j]
    return res


def lagrange_to_euler(M):
    """
    :param M: matrice OD carree d'entiers ou m_ij = nombre de personnes montees en i allant en j
    :return: m liste d'entiers des montees a chaque arret
            v liste d'entiers des descentes a chaque arret
    """
    m = [somme_ligne(M, i) for i in range(0, len(M))]
    v = [somme_colonne(M, j) for j in range(0, len(M))]
    return m, v


# Passage de donnees euleriennes a donnees lagrangiennes
def euler_to_lagrange(m, v):
    """
    :param m: une liste d'entiers, les montees a chaque arret
    :param v: une liste d'entiers, les descentes a chaque arret
    :return: une liste contenant toutes les matrices OD correspondant aux donnees de montees et de descentes
    """
    n = len(m)
    resultats = []

    # Fonction recursive pour trouver les grilles
    def backtrack(grille, m_rest, v_rest, ligne, col):
        # Cas 1 : On a rempli la derniere cellule. On verifie si la grille est valide
        #         Si elle est valide, on la sauvegarde dans les resultats.
        if ligne == n:
            if all(sum(grille[i]) == m[i] for i in range(n)) and all(
                    sum(grille[i][j] for i in range(n)) == v[j] for j in range(n)):
                resultats.append([row[:] for row in grille])
            return

        # Cas 2 : On a rempli la derniere colonne de la ligne actuelle. On passe a la ligne suivante
        if col == n:
            # On appelle la fonction backtrack pour continuer a remplir la matrice
            # en se placant sur la ligne +1 et sur la premiere colonne
            backtrack(grille, m_rest, v_rest, ligne + 1, 0)
            return

        # Cas 3 : On est sur la diagonale ou la partie inferieure gauche, la valeur doit etre 0.
        if ligne >= col:
            grille[ligne][col] = 0  # On met le coef a 0
            backtrack(grille, m_rest, v_rest, ligne, col + 1)  # On continue a remplir la grille.

        # Autre cas : On appelle la fonction backtrack pour chaque valeur que peut prendre la cellule
        else:
            # On definit la valeur maximale que peut prendre la cellule a partir de m_rest et v_rest.
            max_val = min(m_rest[ligne], v_rest[col])
            for val in range(max_val + 1):
                grille[ligne][col] = val
                # On modifie les valeurs de m_rest et v_rest pour remplir les cellules restantes
                # Et respecter les conditions sur les sommes de lignes et de colonnes
                m_rest[ligne] -= val
                v_rest[col] -= val

                # On appelle la fonction backtrack recursive pour remplir la cellule suivante
                backtrack(grille, m_rest, v_rest, ligne, col + 1)

                # On retablie les valeurs de m_rest et v_rest pour tester la nouvelle possibilite
                m_rest[ligne] += val
                v_rest[col] += val
                grille[ligne][col] = 0

    # On initialise une grille vide
    grille_vide = [[0] * n for _ in range(n)]

    # On appelle la fonction recursive backtrack pour remplir la grille et tester les possibilites.
    backtrack(grille_vide, m.copy(), v.copy(), 0, 0)

    return resultats


def print_euler_to_lagrange(m, v):
    """
    Affiche dans la console Python toutes les matrices OD correspondants a m et v
    :param m: liste d'entiers des montees a chaque arret
    :param v: liste d'entiers des descentes a chaque arret
    """
    # On recupere les matrices
    grilles = euler_to_lagrange(m, v)

    # On les affiche
    print(str(len(grilles)) + " grilles valides trouvees")
    for i, grille in enumerate(grilles):
        print(f"Grille {i + 1} :")
        for row in grille:
            print(row)
        print()
    return


def minisation_entropie(grilles):
    """
    :param grilles: Une liste de matrices
    :return: Une matrice qui minimise l'entropie
    """

    # Calcul du nombre total de voyageurs (identique pour toutes les grilles)
    K = 0
    first_grille = grilles[0]
    for i in range(len(first_grille)):
        for j in range(len(first_grille[0])):
            K += first_grille[i][j]

    # Definition "thermodynamique" avec une entropie positive donc on veut minimiser l'entropie
    def calcul_entropie(grille):

        sum = 0.0
        for i in range(len(grille)):
            for j in range(len(grille[i])):
                if grille[i][j] != 0:
                    sum = sum + (grille[i][j] / K * np.log2(grille[i][j] / K))
        return sum

    entropies = [calcul_entropie(grille) for grille in grilles]
    indice_grille_min_entropie = entropies.index(min(entropies))
    return grilles[indice_grille_min_entropie]


def euler_to_best_lagrange(m, v):
    """
    :param m: Une liste d'entiers des montees
    :param v: Une liste d'entiers des descentes
    :return: Une matrice OD qui verifie m et v qui minimise l'entropie
    """
    grilles = euler_to_lagrange(m, v)
    best_grille = minisation_entropie(grilles)
    return best_grille


def affiche_matrice(M):
    """
    Affiche la matrice M dans la console Python
    :param M:
    """
    for ligne in M:
        print(str(ligne) + '\n')


if __name__ == "__main__":
    # Test sur la ligne a de metro
    noms_arrets_ligne_A = ["Poterie", "Blosne", "Triangle", "Italie", "HF", "Clem", "JC", "Gares", "CDG", "Repu",
                           "StAnne", "Anatole Fr", "PC", "Villejean", "Kenndy"]
    mA = [40, 37, 38, 39, 45, 36, 35, 50, 38, 50, 55, 35, 35, 32, 0]
    vA = [0, 33, 35, 38, 42, 35, 33, 52, 40, 47, 49, 38, 40, 45, 38]
    # print_euler_to_lagrange(mA, vA)

    # Test sur une ligne a 6 arrets
    noms_arrets6 = ["A1", "A2", "A3", "A4", "A5", "A6"]
    m6 = [5, 4, 6, 3, 1, 0]
    v6 = [0, 2, 4, 3, 5, 5]
    # print_euler_to_lagrange(m6, v6)
    best = minisation_entropie(euler_to_lagrange(m6, v6))
    # euler_to_graph(noms_arrets6, m6, v6)
    print("Matrice minimisant l'entropie pour ligne a 6 arrets")
    affiche_matrice(best)

    # Test sur une ligne a 5 arrets
    noms_arrets5 = ["A1", "A2", "A3", "A4", "A5"]
    m5 = [2, 3, 1, 2, 0]
    v5 = [0, 1, 2, 2, 3]
    # print_euler_to_lagrange(m5, v5)
    best2 = minisation_entropie(euler_to_lagrange(m5, v5))
    # euler_to_graph(noms_arrets5, m5, v5)
    print("Matrice minimisant l'entropie pour ligne a 5 arrets")
    affiche_matrice(best2)
