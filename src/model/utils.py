import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from src.constants import *
from src.api.clash_royale_api import ClashRoyaleAPI

def draw_adj_matrix_graph(adj_matrix, threshold:int = 1):
    # threshold_matrix = adj_matrix >= threshold
    # adj_matrix = adj_matrix * threshold_matrix

    nx_graph = nx.from_numpy_array(np.round(adj_matrix, 3))
    graph_dict = nx.spring_layout(
        nx_graph
    )
    plt.figure()
    nx.draw(
        nx_graph, 
        graph_dict
    )
    nx.draw_networkx_edge_labels(nx_graph, pos=graph_dict)
    plt.show()


def get_training_testing_data(test_percentage = 0.1):
    api = ClashRoyaleAPI()
    decks = api.read_player_decks("big_cards_data.json")
    adj_matrix = api.get_adjacency_matrix_for_decks(decks)
    adj_matrix = adj_matrix / adj_matrix.max()

    training_decks = []
    testing_decks = []

    indices = np.arange(0, len(decks))
    np.random.shuffle(indices)
    training_indices, testing_indices = indices[:int(indices.shape[0] * (1 - test_percentage))], indices[:int(indices.shape[0] * test_percentage)]
    for i in training_indices:
        training_decks.append(decks[i])
    for i in testing_indices:
        testing_decks.append(decks[i])

    return training_decks, testing_decks, api.cards, adj_matrix


def get_ranking_vectors():
    ranking_encodings = np.eye(NR_OF_BUCKETS, dtype=np.float32)
    ranking_vectors = np.zeros((NR_OF_BUCKETS, NR_OF_CARDS), dtype=np.float32)
    return np.concatenate((ranking_vectors, ranking_encodings), axis=1)

def compute_distances(deck_embedding: np.ndarray, ranking_embeddings: np.ndarray):
    distance = ranking_embeddings - deck_embedding
    distance = distance * distance
    distance = distance.sum(axis = 1)
    distance = np.sqrt(distance)
    bucket = np.argmax(distance)
    return bucket


def plot_loss_accuracy(losses, accuracies):
    plt.figure()

    plt.plot(range(len(losses)), losses)
    plt.title("Loss per Batch")
    plt.ylabel("Loss")
    plt.xlabel("Batch")

    plt.figure()
    plt.plot(range(len(accuracies)), accuracies)
    plt.title("Accuracy per Epoch")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")

    plt.show(block = True)