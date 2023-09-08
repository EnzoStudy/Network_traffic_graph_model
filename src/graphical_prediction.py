"""
Module to analyze and predict individual graphs and create two plots:
    - Network graph without labeling.
    - Network graph after prediction (labeled). 
"""

from data import load_csv_to_df, sample_df
from graph import create_hetero_graph
from model import GNN, predict_graph
import torch
import configparser
from graph import create_networkx_graph
from plot import plot_networkx_graph


import os

def main():
    print('주소')
    print(os.getcwd())

    # Config parameters
    config = configparser.ConfigParser()
    config.read("./config.ini")

    # Create model
    model = GNN(
        input_channels=int(config["PARAMETERS"]["NInputFeatures"]),
        hidden_channels=128,
        output_channels=int(config["PARAMETERS"]["NClasses"]),
        dropout=0)

    # Load state dict of saves model
    model.load_state_dict(torch.load(config["PARAMETERS"]["ModelPath"]))

    # Validate model on unseen data
    df = load_csv_to_df(
        config["PARAMETERS"]["DataFolderPath"] + config["PARAMETERS"]["EvalFile"])

    # Create graph
    # Example: A graph representing some PortScan traces
    df = sample_df(df, 0.005)
    df = df.loc[df[" Label"] == "PortScan"]

    hetero_data = create_hetero_graph(df)
    predicted_flow_labels = predict_graph(
        model, hetero_data, ["BENIGN", "PortScan"])

    G, colors = create_networkx_graph(df, predicted_flow_labels)
    plot_networkx_graph(G, None)
    plot_networkx_graph(G, colors)


if __name__ == "__main__":
    main()
