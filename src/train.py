"""
Module to train the GNN model. All config data must have been added to /src/config.ini file prior
to executing this script. A model.dat file will be created in the first run, and automatically 
loaded again in the following runs.
"""

from data import load_csv_to_df, sample_df
from graph import create_hetero_graph
from model import GNN, run_model_for_graphs_list
import torch
import configparser
from sklearn.utils import shuffle
from torch_geometric.loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    # Config parameters
    config = configparser.ConfigParser()
    config.read("./dev/src/config.ini")

    print('주소')
    print(os.getcwd())

    # Load training dataframe
    df = load_csv_to_df(
        config["PARAMETERS"]["DataFolderPath"] + config["PARAMETERS"]["TrainFile"])
    # df = sample_df(df, sample_rate=float(
    #     config["PARAMETERS"]["TrainSampleRate"]))


    # Shuffle df
    df = shuffle(df)
    df.reset_index(inplace=True)
    # df = df.drop(['Unnamed: 0'], axis=1)
    # df = df.drop(['index'], axis=1)
    n_nodes = int(config["PARAMETERS"]["NumberOfNodes"])
    total = len(df)

    graphs = []
    for i in range(int(total/n_nodes)):
        initial = i*n_nodes
        final = (i+1)*n_nodes
        temp = df[initial:final]
        graphs.append(create_hetero_graph(temp))

    loader = DataLoader(graphs, batch_size=1, shuffle=True)

    # Initialize model

    model = GNN(
        input_channels=int(config["PARAMETERS"]["NInputFeatures"]),
        hidden_channels=128,
        output_channels=int(config["PARAMETERS"]["NClasses"]),
        dropout=0)

    # Load state dict of saves model
    try:
        model.load_state_dict(torch.load(config["PARAMETERS"]["ModelPath"]))
        print("Loading model...")
    except FileNotFoundError:
        print("Creating new model...")

    # Model parameters
    train_epochs = int(config["PARAMETERS"]["NTrainEpochs"])
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3, weight_decay=5e-4)

    # Run model
    print("Starting training phase...")
    loss_arr = run_model_for_graphs_list(
        model, loader, optimizer, train_epochs, len(graphs))

    # Plot Loss curve
    for i in range(len(graphs)):
        plt.plot(np.arange(train_epochs), loss_arr[i, :])
    plt.show()

    # Save model state
    torch.save(model.state_dict(), config["PARAMETERS"]["ModelPath"])


if __name__ == "__main__":
    main()
