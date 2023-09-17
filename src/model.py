"""
Module to create MPNN GNN model
"""

import torch
import torch.nn as nn
from torch_scatter import scatter
from torch_geometric.data import HeteroData
from sklearn.metrics import confusion_matrix
import torch_geometric.transforms as T
from metrics import log_scalar_metrics_to_tensorboard, compute_metrics, log_ce_loss_to_tensorboard
from tensorboardX import SummaryWriter
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
from metrics import compute_average_metrics


class GNN(nn.Module):
    """
    GNN module mainly containing:
        - MPNN (Message Passing Neural Network):
            - Message function: Dense Layer.
            - Aggregation function: Mean.
            - Update function: Gate Recurrent NN
        - Readout function (node classification):
            - Group of Dense layers with final Softmax activation.
    """

    def __init__(self, input_channels, hidden_channels, output_channels, dropout):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.dropout = dropout

        self.message_func_ip = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(hidden_channels * 2, hidden_channels)
        )

        self.message_func_conn = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(hidden_channels * 2, hidden_channels)
        )
        self.ip_update = nn.GRU(hidden_channels, hidden_channels)
        self.connection_update = nn.GRU(hidden_channels, hidden_channels)

        self.readout_nn = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_channels, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, self.output_channels),
            nn.Softmax(dim=1)
        )

    def forward(self, data: HeteroData):

        # number of connections
        n_connections = data["connection"].x.shape[0]

        # number of iterations
        T = 8

        # adjacencies
        src_ip_to_port, dst_ip_to_port = data[(
            "host_ip", "to", "host_ip_port")]["edge_index"]
        src_port_to_connection, dst_port_to_connection = data[(
            "host_ip_port", "to", "connection")]["edge_index"]
        src_connection_to_port, dst_connection_to_port = data[(
            "connection", "to", "host_ip_port")]["edge_index"]
        src_port_to_ip, dst_port_to_ip = data[(
            "host_ip_port", "to", "host_ip")]["edge_index"]

        # ip node initialization
        h_ip = data["host_ip"].x
        # ip and port node initialization
        h_ip_port = data["host_ip_port"].x

        # connection node initialization
        h_conn = torch.cat((data["connection"].x, torch.zeros(
            n_connections, self.hidden_channels - self.input_channels)), dim=1)

        for _ in range(T):
            # PART 1
            # Ip to Port
            ip_gather = h_ip[src_ip_to_port]
            port_gather = h_ip_port[dst_ip_to_port]
            nn_input = torch.cat((ip_gather, port_gather), dim=1).float()
            ip_to_port_message = self.message_func_ip(nn_input)
            ip_to_port_mean = scatter(ip_to_port_message, dst_ip_to_port,
                                      dim=0, reduce="mean")

            # PART 2
            # Port to Ip
            port_gather = h_ip_port[src_port_to_ip]
            ip_gather = h_ip[dst_port_to_ip]
            nn_input = torch.cat((port_gather, ip_gather), dim=1).float()
            port_to_ip_message = self.message_func_ip(nn_input)
            port_to_ip_mean = scatter(port_to_ip_message, dst_port_to_ip,
                                      dim=0, reduce="mean")

            # PART 3
            # Port to connection
            port_gather = h_ip_port[src_port_to_connection]
            connection_gather = h_conn[dst_port_to_connection]
            nn_input = torch.cat(
                (port_gather, connection_gather), dim=1).float()
            port_to_connection_message = self.message_func_ip(nn_input)
            port_to_connection_mean = scatter(
                port_to_connection_message, dst_port_to_connection, dim=0, reduce="mean")

            # PART 4
            # Connection to port
            connection_gather = h_conn[src_connection_to_port]
            port_gather = h_ip_port[dst_connection_to_port]
            nn_input = torch.cat(
                (connection_gather, port_gather), dim=1).float()
            connection_to_port_message = self.message_func_ip(nn_input)
            connection_to_port_mean = scatter(
                connection_to_port_message, dst_connection_to_port, dim=0, reduce="mean")

            # PART 5
            # update nodes
            _, new_h_ip = self.ip_update(port_to_ip_mean.unsqueeze(
                0), h_ip.unsqueeze(0))  # (2, 128), (2, 128)
            h_ip = new_h_ip[0]
            _, new_h_conn = self.connection_update(
                port_to_connection_mean.unsqueeze(0), h_conn.unsqueeze(0))
            h_conn = new_h_conn[0]
            _, new_h_ip_port = self.ip_update(
                connection_to_port_mean.unsqueeze(0), h_ip_port.unsqueeze(0))
            h_ip_port = new_h_ip_port[0]
            _, new_h_ip_port = self.ip_update(
                ip_to_port_mean.unsqueeze(0), h_ip_port.unsqueeze(0))
            h_ip_port = new_h_ip_port[0]

        return self.readout_nn(h_conn)


def train(model, hetero_data, optimizer):
    
    """
    Args:
        - model: GNN model.
        - hetero_data: PyG hetero_data object.
        - optimizer.
    Returns:
        - Cross Entropy Loss.
    """

    crit = torch.nn.CrossEntropyLoss()
    model.train()
    optimizer.zero_grad()
    mask = hetero_data["connection"].train_mask
    out = model(hetero_data)
    loss = crit(out[mask], hetero_data["connection"].y[mask])
    loss.backward()
    optimizer.step()
    return float(loss)


def run_model(model, hetero_data, optimizer, train_epochs):
    """
    Args:
        - model: GNN model.
        - hetero_data: PyG hetero data object.
        - optimizer.
        - train_epochs: number of training iterations
    Returns:
        - loss_arr: loss array containing all loss values at each training epoch.
    """
    writer = SummaryWriter("runs")
    loss_arr = []
    for epoch in range(0, train_epochs):
        loss = train(model, hetero_data, optimizer)
        loss_arr.append(loss)
        metrics = test_model(model, hetero_data)
        # Log metrics to TensorBoard
        log_ce_loss_to_tensorboard(writer, epoch, loss)
        log_scalar_metrics_to_tensorboard(writer, epoch, metrics)
        print("Epoch " + str(epoch) + ":" +
              " {Loss: " + str(loss) + "}" + str(metrics))

    writer.flush()
    writer.close()
    return loss_arr


@ torch.no_grad()
def test_model(model, hetero_data):
    """
    Args:
        - model: GNN model.
        - hetero_data: PyG hetero_data object.
    Returns:
        - Dictionary containing evaluation metrics (f1, accuracy, precision, recall)
    """
    model.eval()
    mask = hetero_data["connection"].test_mask
    out = model(hetero_data)
    _, pred = torch.max(out[mask], 1)
    y = hetero_data["connection"].y[mask]
    return compute_metrics(y, pred)


@ torch.no_grad()
def val_model(model, hetero_data, classes):
    """
    Args:
        - model: GNN model
        - hetero_data: PyG HeteroData object.
    Returns:
        - Dictionary containing evaluation metrics on unseen data and t
            computation of the confusion matrix.
    """

    model.eval()
    out = model(hetero_data)
    _, pred = torch.max(out, 1)
    y = hetero_data["connection"].y

    # Compute Confusion Matrix
    cf_matrix = confusion_matrix(y.numpy(), pred.numpy())
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.show()
    return compute_metrics(y, pred)


def train_test_val_split(hetero_data):
    """
    Args:
        - hetero_data: PyG hetero_data object
    Returns:
        - hetero_data object with [train, test, val] masks which can be used 
        for training, testing and validation purposes.
    """
    transform = T.Compose([
        T.RandomNodeSplit(num_val=0.0, num_test=0.15)
    ])
    return transform(hetero_data)


@ torch.no_grad()
def predict_graph(model, hetero_data, classes):
    """
    Method created to predict the labels of a given Graph (HeteroData object)

    Args:
        - model: GNN model
        - hetero_data: Graph to predict
        - classes: list of labels (classes)
    Returns:
        - results_dict: returns dictionary mapping all node_ids to their predicted label
    """
    model.eval()
    out = model(hetero_data)
    _, pred = torch.max(out, 1)
    y = hetero_data["connection"].y
    results_dict = {}
    for index, label in enumerate(pred.tolist()):
        results_dict[index] = label
    # Compute Confusion Matrix
    cf_matrix = confusion_matrix(y.numpy(), pred.numpy())
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.show()
    print(compute_metrics(y, pred))
    return results_dict


def run_model_for_graphs_list(model, loader, optimizer, train_epochs, n_graphs):
    """
    Method created to train the model againts a list of graphs.

    Args:
        - model: GNN model
        - loader: DataLoader object containing graphs
        - optimizer
        - train_epochs
        - n_graphs: number of graphs in DataLoader
    Returns:
        - Losses array for all training epochs and graphs
    """

    loss_arr = np.zeros((n_graphs, train_epochs), dtype=float)
    for epoch in range(1, train_epochs+1):
        k = 0
        print("Training epoch: " + str(epoch))
        i = 0
        for data in loader:
            data = train_test_val_split(data)
            loss = train(model, data, optimizer)
            loss_arr[k, epoch-1] = loss
            print("Graph " + str(i) + ": " + str(loss))
            i = i + 1
            k += 1

    # Compute average metrics

    f1_scores = []
    accuracy_scores = []
    precision_scores = []
    recall_scores = []

    for data in loader:
        data = train_test_val_split(data)
        metrics = test_model(model, data)
        # Append metrics for averaging
        f1_scores.append(metrics["F1 Score"])
        accuracy_scores.append(metrics["Accuracy"])
        precision_scores.append(metrics["Precision"])
        recall_scores.append(metrics["Recall"])

    print(compute_average_metrics(
        f1_scores, accuracy_scores, recall_scores, precision_scores))

    return loss_arr


@ torch.no_grad()
def val_model_for_graphs_list(model, loader):
    """
    Method to validate the model against a list of graphs.

    Args:
        - model: GNN model
        - loader: DataLoader object containing all the graphs.
    Returns:
        - Average classification metrics among all graphs.
    """
    model.eval()
    # Compute average metrics

    f1_scores = []
    accuracy_scores = []
    precision_scores = []
    recall_scores = []

    i = 0
    for data in loader:
        # Predict Graph
        out = model(data)
        _, pred = torch.max(out, 1)
        y = data["connection"].y
        metrics = compute_metrics(y, pred)
        print("Graph " + str(i) + ": " + str(metrics))
        i = i + 1
        # Append metrics for averaging
        f1_scores.append(metrics["F1 Score"])
        accuracy_scores.append(metrics["Accuracy"])
        precision_scores.append(metrics["Precision"])
        recall_scores.append(metrics["Recall"])

    return compute_average_metrics(f1_scores, accuracy_scores, recall_scores, precision_scores)
