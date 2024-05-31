import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


# Fonctions de conversion
def mv_to_b(m, v):
    """
    :param m: liste d'entier des montees a chaque arret
    :param v: liste d'entier des descentes a chaque arret
    :return: liste d'entier du bilan des montees et descentes a chaque arret
    """
    return [m[i] - v[i] for i in range(0, len(m))]


def mv_to_p(m, v):
    """
    :param m: liste d'entier des montees a chaque arret
    :param v: liste d'entier des descentes a chaque arret
    :return: la liste du nombre de voyageurs au depart du point i
    """
    p = [m[0]]
    for i in range(1, len(m) - 1):
        p += [p[i - 1] + m[i] - v[i]]

    return p


def p_to_b(p):
    """
    :param p: liste d'entier du nombre de personnes transportees entre les arrets
              p[i] = personnes transportees entre l'arret i et l'arret i+1
    :return: liste d'entier du bilan des montees et descentes a chaque arret
    """
    return [p[0]] + [p[i] - p[i - 1] for i in range(1, len(p))] + [-p[len(p) - 1]]


def b_to_p(b):
    """
    :param b: liste d'entier du bilan des montees et descentes a chaque arret
    :return: liste d'entier du nombre de personnes transportees entre les arrets
              p[i] = personnes transportees entre l'arret i et l'arret i+1
    """
    p = [b[0]]
    for i in range(1, len(b) - 1):
        p = p + [p[i - 1] + b[i]]
    return p


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
                    sum = sum - (grille[i][j] / K * np.log2(grille[i][j] / K))
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


# Fonctions de generation des graphes
def lagrange_to_graph(noms, M):
    """
    :param M: matrice OD carree d'entiers ou m_ij = nombre de personnes montees en i allant en j
    :return: le graphe networkx associe
    """
    # Initialisation du graphe
    G = nx.DiGraph()

    # Creation des arrets comme noeuds
    arrets = noms
    G.add_nodes_from(arrets)

    # Dessiner le graphe sans les noeuds fictifs
    pos = nx.spring_layout(G)

    # Definition des positions des noeuds reels
    for i in range(len(arrets)):
        pos[arrets[i]] = (0.5 * i, 0)

    # Creation des arretes de la ligne de bus
    trajets_ligne = [(noms[i], noms[i + 1]) for i in range(0, len(noms) - 1)]
    G.add_edges_from(trajets_ligne)

    # Creation des arretes pour les trajets des voyageurs (gamma_ij) et des noeuds ficitfs pour ces arretes
    trajets_voyageurs_depart = []
    trajets_voyageurs_arrivee = []
    nbr_voyageurs = []
    compteur = 0
    for i in range(len(M[0])):
        for j in range(i + 1, len(M[0])):  # Matrice triangulaire superieure
            nb = M[i][j]
            if nb > 0:
                trajets_voyageurs_depart.append((arrets[i], f"edge_{compteur}"))
                trajets_voyageurs_arrivee.append((f"edge_{compteur}", arrets[j]))
                val = 0.04
                if compteur % 2 == 0:
                    val = -val
                pos[f"edge_{compteur}"] = ((pos[arrets[i]][0] + pos[arrets[j]][0]) / 2, pos[arrets[i]][1] + val)
                compteur += 1
                nbr_voyageurs.append(nb)

    G.add_edges_from(trajets_voyageurs_depart)
    G.add_edges_from(trajets_voyageurs_arrivee)

    # Creation des arretes rentrantes pour les voyageurs montants
    for arret in arrets:
        G.add_edge(f"start_{arret}", arret)

    # Creation des arretes sortante pour les voyageurs descendants
    for arret in arrets:
        G.add_edge(arret, f"end_{arret}")

    # Positionnement des noeuds fictifs de montees et descentes
    for arret in arrets:
        pos[f"start_{arret}"] = (pos[arret][0], pos[arret][1] + 0.1)
        pos[f"end_{arret}"] = (pos[arret][0], pos[arret][1] - 0.1)

    # Dessiner les noeuds reels
    nx.draw_networkx_nodes(G, pos, nodelist=arrets, node_color='grey', node_size=1000)

    # Ajouter les labels aux noeuds reels
    nx.draw_networkx_labels(G, pos, labels={arret: arret for arret in arrets}, font_size=6)

    # Dessiner les aretes incluant celles des noeuds fictifs
    ## Trajets de la ligne de bus
    nx.draw_networkx_edges(G, pos, edgelist=trajets_ligne, edge_color='black', arrows=True, arrowstyle='-|>')

    ## Trajets des voyageurs
    nx.draw_networkx_edges(G, pos, edgelist=trajets_voyageurs_depart, edge_color='blue', style='dotted', arrows=True,
                           arrowstyle='-|>')
    nx.draw_networkx_edges(G, pos, edgelist=trajets_voyageurs_arrivee, edge_color='blue', style='dotted', arrows=True,
                           arrowstyle='-|>')
    ## montees
    nx.draw_networkx_edges(G, pos, edgelist=[(f"start_{arret}", arret) for arret in arrets], edge_color='green',
                           style='dashed', arrows=True, arrowstyle='-|>')
    ## Descentes
    nx.draw_networkx_edges(G, pos, edgelist=[(arret, f"end_{arret}") for arret in arrets], edge_color='red',
                           style='dashed', arrows=True, arrowstyle='-|>')

    # Calcul de donnees
    m_M, v_M = lagrange_to_euler(M)
    p_M = mv_to_p(m_M, v_M)

    # Afficher les etiquettes des arretes
    # Afficher le nombre de voyageurs entre chaque arret de la ligne
    edge_labels = {(arrets[i], arrets[i + 1]): str(p_M[i]) for i in range(len(arrets) - 1)}

    # Afficher le nombre de voyageurs pour chaque trajet gamma_i,j
    edge_labels.update({trajets_voyageurs_depart[k]: nbr_voyageurs[k] for k in range(len(trajets_voyageurs_depart))})

    # Afficher les etiquettes des arretes rentrantes et sortantes
    edge_labels.update({("start_" + arrets[i], arrets[i]): "+" + str(m_M[i]) for i in range(len(arrets))})
    edge_labels.update({(arrets[i], "end_" + arrets[i]): "-" + str(v_M[i]) for i in range(len(arrets))})
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black')

    plt.title("Graphe lagrangien")
    plt.show()
    return None


def euler_to_graph(noms, m, v):
    """
        :param m: liste d'entiers des montees
        :param v: liste d'entiers des descentes
        :return: le graphe networkx associe
    """

    # Initialisation du graphe
    G = nx.DiGraph()

    # Creation des arrets comme noeuds
    arrets = noms
    G.add_nodes_from(arrets)

    # Creation des arretes pour les trajets entre stations
    trajets = [(noms[i], noms[i + 1]) for i in range(0, len(noms) - 1)]
    G.add_edges_from(trajets)

    # Creation des arretes rentrantes pour les voyageurs montants
    for arret in arrets:
        G.add_edge(f"start_{arret}", arret)

    # Creation des arretes sortante pour les voyageurs descendants
    for arret in arrets:
        G.add_edge(arret, f"end_{arret}")

    # Dessiner le graphe sans les noeuds fictifs
    pos = nx.spring_layout(G)

    # Definition des positions des noeuds reels
    for i in range(len(arrets)):
        pos[arrets[i]] = (len(m) * 2 * i, 0)

    # Positionnement des noeuds fictifs
    for arret in arrets:
        pos[f"start_{arret}"] = (pos[arret][0], pos[arret][1] + 0.05)
        pos[f"end_{arret}"] = (pos[arret][0], pos[arret][1] - 0.05)

    # Dessiner les noeuds reels
    nx.draw_networkx_nodes(G, pos, nodelist=arrets, node_color='grey', node_size=1000)

    # Dessiner les aretes incluant celles des noeuds fictifs
    nx.draw_networkx_edges(G, pos, edgelist=trajets, edge_color='black', arrows=True, arrowstyle='-|>')
    nx.draw_networkx_edges(G, pos, edgelist=[(f"start_{arret}", arret) for arret in arrets], edge_color='green',
                           style='dashed', arrows=True, arrowstyle='-|>')
    nx.draw_networkx_edges(G, pos, edgelist=[(arret, f"end_{arret}") for arret in arrets], edge_color='red',
                           style='dashed', arrows=True, arrowstyle='-|>')

    # Ajouter les labels aux noeuds reels
    nx.draw_networkx_labels(G, pos, labels={arret: arret for arret in arrets}, font_size=10)

    # Afficher les etiquettes des arretes entre arrets
    p = mv_to_p(m, v)
    edge_labels = {(arrets[i], arrets[i + 1]): str(p[i]) for i in range(len(arrets) - 1)}

    # Afficher les etiquettes des arretes rentrantes et sortantes
    edge_labels.update({("start_" + arrets[i], arrets[i]): "+" + str(m[i]) for i in range(len(arrets))})
    edge_labels.update({(arrets[i], "end_" + arrets[i]): "-" + str(v[i]) for i in range(len(arrets))})
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black')

    plt.title("Graphe eulerien")
    plt.show()

    return G


def affiche_matrice(M):
    """
    Affiche la matrice M dans la console Python
    :param M:
    """
    for ligne in M:
        print(str(ligne) + '\n')


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
# best = minisation_entropie(euler_to_lagrange(m6, v6))
# euler_to_graph(noms_arrets6, m6, v6)
# lagrange_to_graph(noms_arrets6, best)
# print("Matrice minimisant l'entropie pour ligne à 6 arrêts")
# affiche_matrice(best)

# Test sur une ligne a 5 arrets
noms_arrets5 = ["A1", "A2", "A3", "A4", "A5"]
m5 = [2, 3, 1, 2, 0]
v5 = [0, 1, 2, 2, 3]
# print_euler_to_lagrange(m5, v5)
# best2 = minisation_entropie(euler_to_lagrange(m5, v5))
# euler_to_graph(noms_arrets5, m5, v5)
# lagrange_to_graph(noms_arrets5, best2)
# print("Matrice minimisant l'entropie pour ligne à 5 arrêts")
# affiche_matrice(best2)
