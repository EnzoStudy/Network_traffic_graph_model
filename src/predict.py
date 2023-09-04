"""
prediction을 위한 code 
저장된 model 경로를 config 로 연결


Module to validate the created model. The evaluation file path and the model.dat file path should have 
been previously added in the /src/config.ini file.
"""

from data import load_csv_to_df
from graph import create_hetero_graph
from model import GNN, val_model_for_graphs_list
import torch
import configparser
from torch_geometric.loader import DataLoader
from sklearn.utils import shuffle


def main():
    # Config parameters
    config = configparser.ConfigParser()
    config.read("config.ini")

    # Create model
    model = GNN(
        input_channels=int(config["PARAMETERS"]["NInputFeatures"]),
        hidden_channels=128,
        output_channels=int(config["PARAMETERS"]["NClasses"]),
        dropout=0)

    # Load state dict of saves model
    # model가 이미 학습되어 저장되어 있어야 load 가능
    model.load_state_dict(torch.load(config["PARAMETERS"]["ModelPath"]))

    # Validate model on unseen data
    print("Starting evaluation phase...")
    df = load_csv_to_df(
        config["PARAMETERS"]["DataFolderPath"] + config["PARAMETERS"]["EvalFile"])

    # Shuffle df
    df = shuffle(df)
    df.reset_index(inplace=True)
    n_nodes = int(config["PARAMETERS"]["NumberOfNodes"])
    total = len(df)

    graphs = []
    for i in range(int(total/n_nodes)):
        initial = i*n_nodes
        final = (i+1)*n_nodes
        temp = df[initial:final]
        graphs.append(create_hetero_graph(temp))

    loader = DataLoader(graphs, batch_size=1, shuffle=True)

    average_metrics = val_model_for_graphs_list(
        model, loader)
    print("Average classification metrics: " + str(average_metrics))


if __name__ == "__main__":
    main()
