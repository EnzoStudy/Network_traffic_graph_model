"""
Module intended to provide helper functions to create graph and data visuals
"""

import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sn


def default_graph_options():
    """
    Returns:
        - Utils options dictionary to improve plotting the graph.
    """
    options = {
        'arrowstyle': '-|>',
    }
    return options


def plot_networkx_graph(G, colors, options=default_graph_options()):
    """
    Args:
        - G: networkX graph
        - colors: colors list (len == N of nodes in G)
        - options: default plot options
    """
    pos = nx.get_node_attributes(G, 'pos')
    if len(pos) == 0:
        pos = None
    _ = plt.figure(figsize=(40, 40))
    nx.draw_networkx(G, arrows=True, node_color=colors,
                     with_labels=False, pos=None, **options)
    plt.axis('equal')
    plt.show()


def plot_ce_loss_curve(n_epochs, loss_arr):
    """
    Args:
        - n_epochs: number of epochs (x axis)
        - loss_arr: array containing all registered losses for n_epochs
    """
    plt.plot(range(0, n_epochs), loss_arr)
    plt.show()


def plot_confusion_matrix(df_cm):
    """
    Args:
        - Dataframe containg confusion matrix.
    """
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.show()
