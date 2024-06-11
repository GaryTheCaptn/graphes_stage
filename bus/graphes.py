import matplotlib.pyplot as plt
import networkx as nx
from bus import lagrange_to_euler


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


# Fonctions de conversion
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
