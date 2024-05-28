import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


# Fonctions de conversion
def mv_to_b(m, v):
    """
    :param m: liste d'entier des montées à chaque arrêt
    :param v: liste d'entier des descentes à chaque arrêt
    :return: liste d'entier du bilan des montées et descentes à chaque arrêt
    """
    return [m[i] - v[i] for i in range(0, len(m))]


def mv_to_p(m, v):
    """
    :param m: liste d'entier des montées à chaque arrêt
    :param v: liste d'entier des descentes à chaque arrêt
    :return: la liste du nombre de voyageurs au départ du point i
    """
    p = [m[0]]
    for i in range(1, len(m) - 1):
        p += [p[i - 1] + m[i] - v[i]]

    return p


def p_to_b(p):
    """
    :param p: liste d'entier du nombre de personnes transportées entre les arrêts
              p[i] = personnes transportées entre l'arrêt i et l'arrêt i+1
    :return: liste d'entier du bilan des montées et descentes à chaque arrêt
    """
    return [p[0]] + [p[i] - p[i - 1] for i in range(1, len(p))] + [-p[len(p) - 1]]


def b_to_p(b):
    """

    :param b: liste d'entier du bilan des montées et descentes à chaque arrêt
    :return: liste d'entier du nombre de personnes transportées entre les arrêts
              p[i] = personnes transportées entre l'arrêt i et l'arrêt i+1
    """
    p = [b[0]]
    for i in range(1, len(b) - 1):
        p = p + [p[i - 1] + b[i]]
    return p


# Passage de données lagrangiennes à eulériennes
def somme_ligne(M, i):
    res = 0
    for j in range(0, len(M)):
        res += M[i][j]
    return res


def somme_colonne(M, j):
    res = 0
    for i in range(0, len(M)):
        res += M[i][j]
    return res


def lagrange_to_euler(M):
    """
    :param M: matrice OD carrée d'entiers où m_ij = nombre de personnes montées en i allant en j
    :return: m liste d'entiers des montées à chaque arrêt
            v liste d'entiers des descentes à chaque arrêt
    """
    m = [somme_ligne(M, i) for i in range(0, len(M))]
    v = [somme_colonne(M, j) for j in range(0, len(M))]
    return m, v


# Passage de données eulériennes à données lagrangiennes
def euler_to_lagrange(m, v):
    n = len(m)
    resultats = []

    def backtrack(grille, m_rest, v_rest, row, col):
        # Si nous sommes à la dernière cellule, vérifions si la grille est valide
        if row == n:
            if all(sum(grille[i]) == m[i] for i in range(n)) and all(
                    sum(grille[i][j] for i in range(n)) == v[j] for j in range(n)):
                resultats.append([row[:] for row in grille])
            return

        # Si nous avons dépassé la dernière colonne, passer à la ligne suivante
        if col == n:
            backtrack(grille, m_rest, v_rest, row + 1, 0)
            return

        # Si nous sommes sur la diagonale ou dans la partie inférieure gauche, la valeur doit être 0
        if row >= col:
            grille[row][col] = 0
            backtrack(grille, m_rest, v_rest, row, col + 1)
        else:
            # Essayer chaque valeur possible pour cette cellule
            max_val = min(m_rest[row], v_rest[col])
            for val in range(max_val + 1):
                grille[row][col] = val
                m_rest[row] -= val
                v_rest[col] -= val

                # Appel récursif pour la cellule suivante
                backtrack(grille, m_rest, v_rest, row, col + 1)

                # Rétablir les valeurs originales pour essayer la prochaine possibilité
                m_rest[row] += val
                v_rest[col] += val
                grille[row][col] = 0

    # Initialiser une grille vide
    grille_vide = [[0] * n for _ in range(n)]
    backtrack(grille_vide, m.copy(), v.copy(), 0, 0)

    return resultats


def print_euler_to_lagrange(m, v):
    grilles = euler_to_lagrange(m, v)
    print(str(len(grilles)) + " grilles valides trouvées")
    for i, grille in enumerate(grilles):
        print(f"Grille {i + 1} :")
        for row in grille:
            print(row)
        print()
    return


def minisation_entropie(grilles):
    # Définition "thermodynamique" avec une entropie positive donc on veut minimiser l'entropie
    def calcul_entropie(grille):
        sum = 0.0
        for i in range(len(grille)):
            for j in range(len(grille[i])):
                if grille[i][j] != 0:
                    sum = sum - (grille[i][j] * np.log2(grille[i][j]))
        return sum

    entropies = [calcul_entropie(grille) for grille in grilles]
    indice_grille_min_entropie = entropies.index(min(entropies))
    return grilles[indice_grille_min_entropie]


def euler_to_best_lagrange(m, v):
    grilles = euler_to_lagrange(m, v)
    best_grille = minisation_entropie(grilles)
    return best_grille


# Fonctions de génération des graphes
def lagrange_to_graph(noms, M):
    """
    :param M: matrice OD carrée d'entiers où m_ij = nombre de personnes montées en i allant en j
    :return: le graphe networkx associé
    """
    # Initialisation du graphe
    G = nx.DiGraph()

    # Création des arrêts comme noeuds
    arrets = noms
    G.add_nodes_from(arrets)

    # Dessiner le graphe sans les noeuds fictifs
    pos = nx.spring_layout(G)

    # Définition des positions des noeuds réels
    for i in range(len(arrets)):
        pos[arrets[i]] = (0.5 * i, 0)

    # Création des arrêtes de la ligne de bus
    trajets_ligne = [(noms[i], noms[i + 1]) for i in range(0, len(noms) - 1)]
    G.add_edges_from(trajets_ligne)

    # Création des arrêtes pour les trajets des voyageurs (gamma_ij) et des noeuds ficitfs pour ces arrêtes
    trajets_voyageurs_depart = []
    trajets_voyageurs_arrivee = []
    nbr_voyageurs = []
    compteur = 0
    for i in range(len(M[0])):
        for j in range(i + 1, len(M[0])):  # Matrice triangulaire supérieure
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

    # Création des arrêtes rentrantes pour les voyageurs montants
    for arret in arrets:
        G.add_edge(f"start_{arret}", arret)

    # Création des arrêtes sortante pour les voyageurs descendants
    for arret in arrets:
        G.add_edge(arret, f"end_{arret}")

    # Positionnement des noeuds fictifs de montées et descentes
    for arret in arrets:
        pos[f"start_{arret}"] = (pos[arret][0], pos[arret][1] + 0.1)
        pos[f"end_{arret}"] = (pos[arret][0], pos[arret][1] - 0.1)

    # Dessiner les noeuds réels
    nx.draw_networkx_nodes(G, pos, nodelist=arrets, node_color='grey', node_size=1000)

    # Ajouter les labels aux noeuds réels
    nx.draw_networkx_labels(G, pos, labels={arret: arret for arret in arrets}, font_size=6)

    # Dessiner les arêtes incluant celles des noeuds fictifs
    ## Trajets de la ligne de bus
    nx.draw_networkx_edges(G, pos, edgelist=trajets_ligne, edge_color='black', arrows=True, arrowstyle='-|>')

    ## Trajets des voyageurs
    nx.draw_networkx_edges(G, pos, edgelist=trajets_voyageurs_depart, edge_color='blue', style='dotted', arrows=True,
                           arrowstyle='-|>')
    nx.draw_networkx_edges(G, pos, edgelist=trajets_voyageurs_arrivee, edge_color='blue', style='dotted', arrows=True,
                           arrowstyle='-|>')
    ## Montées
    nx.draw_networkx_edges(G, pos, edgelist=[(f"start_{arret}", arret) for arret in arrets], edge_color='green',
                           style='dashed', arrows=True, arrowstyle='-|>')
    ## Descentes
    nx.draw_networkx_edges(G, pos, edgelist=[(arret, f"end_{arret}") for arret in arrets], edge_color='red',
                           style='dashed', arrows=True, arrowstyle='-|>')

    # Calcul de données
    m_M, v_M = lagrange_to_euler(M)
    p_M = mv_to_p(m_M, v_M)

    # Afficher les étiquettes des arrêtes
    # Afficher le nombre de voyageurs entre chaque arrêt de la ligne
    edge_labels = {(arrets[i], arrets[i + 1]): str(p_M[i]) for i in range(len(arrets) - 1)}

    # Afficher le nombre de voyageurs pour chaque trajet gamma_i,j
    edge_labels.update({trajets_voyageurs_depart[k]: nbr_voyageurs[k] for k in range(len(trajets_voyageurs_depart))})

    # Afficher les étiquettes des arrêtes rentrantes et sortantes
    edge_labels.update({("start_" + arrets[i], arrets[i]): "+" + str(m_M[i]) for i in range(len(arrets))})
    edge_labels.update({(arrets[i], "end_" + arrets[i]): "-" + str(v_M[i]) for i in range(len(arrets))})
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black')

    plt.title("Graphe lagrangien")
    plt.show()
    return None


def euler_to_graph(noms, m, v):
    """
        :param m: liste d'entiers des montées
        :param v: liste d'entiers des descentes
        :return: le graphe networkx associé
    """

    # Initialisation du graphe
    G = nx.DiGraph()
    plt.figure(figsize=(20, 10))

    # Création des arrêts comme noeuds
    arrets = noms
    G.add_nodes_from(arrets)

    # Création des arrêtes pour les trajets entre stations
    trajets = [(noms[i], noms[i + 1]) for i in range(0, len(noms) - 1)]
    G.add_edges_from(trajets)

    # Création des arrêtes rentrantes pour les voyageurs montants
    for arret in arrets:
        G.add_edge(f"start_{arret}", arret)

    # Création des arrêtes sortante pour les voyageurs descendants
    for arret in arrets:
        G.add_edge(arret, f"end_{arret}")

    # Dessiner le graphe sans les noeuds fictifs
    pos = nx.spring_layout(G)

    # Définition des positions des noeuds réels
    for i in range(len(arrets)):
        pos[arrets[i]] = (len(m) * 2 * i, 0)

    # Positionnement des noeuds fictifs
    for arret in arrets:
        pos[f"start_{arret}"] = (pos[arret][0], pos[arret][1] + 0.05)
        pos[f"end_{arret}"] = (pos[arret][0], pos[arret][1] - 0.05)

    # Dessiner les noeuds réels
    nx.draw_networkx_nodes(G, pos, nodelist=arrets, node_color='grey', node_size=2000)

    # Dessiner les arêtes incluant celles des noeuds fictifs
    nx.draw_networkx_edges(G, pos, edgelist=trajets, edge_color='black', arrows=True, arrowstyle='-|>')
    nx.draw_networkx_edges(G, pos, edgelist=[(f"start_{arret}", arret) for arret in arrets], edge_color='green',
                           style='dashed', arrows=True, arrowstyle='-|>')
    nx.draw_networkx_edges(G, pos, edgelist=[(arret, f"end_{arret}") for arret in arrets], edge_color='red',
                           style='dashed', arrows=True, arrowstyle='-|>')

    # Ajouter les labels aux noeuds réels
    nx.draw_networkx_labels(G, pos, labels={arret: arret for arret in arrets}, font_size=10)

    # Afficher les étiquettes des arrêtes entre arrêts
    p = mv_to_p(m, v)
    edge_labels = {(arrets[i], arrets[i + 1]): str(p[i]) for i in range(len(arrets) - 1)}

    # Afficher les étiquettes des arrêtes rentrantes et sortantes
    edge_labels.update({("start_" + arrets[i], arrets[i]): "+" + str(m[i]) for i in range(len(arrets))})
    edge_labels.update({(arrets[i], "end_" + arrets[i]): "-" + str(v[i]) for i in range(len(arrets))})
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black')

    plt.title("Graphe eulérien")
    plt.show()

    return G


def affiche_matrice(M):
    for ligne in M:
        print(str(ligne) + '\n')


# Test sur la ligne a de métro
noms_arrets_ligne_A = ["Poterie", "Blosne", "Triangle", "Italie", "HF", "Clem", "JC", "Gares", "CDG", "Répu",
                       "StAnne", "Anatole Fr", "PC", "Villejean", "Kenndy"]
mA = [10, 7, 8, 9, 15, 6, 5, 20, 8, 20, 25, 5, 5, 2, 0]
vA = [0, 3, 5, 8, 12, 5, 3, 22, 10, 17, 19, 8, 10, 15, 8]

# Test sur une ligne à 5 arrêts
noms_arrets5 = ["A1", "A2", "A3", "A4", "A5", "A6"]
m5 = [5, 4, 6, 3, 1, 0]
v5 = [0, 2, 4, 3, 5, 5]
# print_euler_to_lagrange(m5, v5)
lagrange_to_graph(noms_arrets5, euler_to_best_lagrange(m5, v5))
# affiche_matrice(minisation_entropie(euler_to_lagrange(m5, v5)))

# Test sur une ligne à 5 arrêts
noms_arrets4 = ["A1", "A2", "A3", "A4", "A5"]
m4 = [2, 3, 1, 2, 0]
v4 = [0, 1, 2, 2, 3]
# print_euler_to_lagrange(m4, v4)
# affiche_matrice(minisation_entropie(euler_to_lagrange(m4, v4)))
