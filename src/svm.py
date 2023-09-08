""""
Experiment to check how a Support Vector Machine model reacts against
modifications in the synthetic testing datasets.
"""

import torch
from data import load_csv_to_df
from graph import get_flow_features_values, get_encoded_label
from data import modify_portscan_attack_behavior
import configparser
from sklearn import svm
import os

source_code_dir = '/home/dev/Network_traffic_graph_model-main/src'

print(os.getcwd())

config = configparser.ConfigParser()
config.read(source_code_dir+"/config.ini")



model = svm.SVC()


train = load_csv_to_df(
    config["PARAMETERS"]["DataFolderPath"] + config["PARAMETERS"]["TrainFile"])

test = load_csv_to_df(
    config["PARAMETERS"]["DataFolderPath"] + config["PARAMETERS"]["EvalFile"])


labels_map = {"BENIGN": 0, "PortScan": 1}
labels = ["BENIGN", "MALIGN"]

flow_features = [" Flow Duration"]

flow_features = [
    " Average Packet Size",
    " Flow IAT Mean",
    " Flow Duration",
]


test = modify_portscan_attack_behavior(test)


# train vectors
x_train = []
y_train = []
for _, row in train.iterrows():
    x_train.append(get_flow_features_values(row, flow_features))
    y_train.append(get_encoded_label(
        row, labels_map))

x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)

# test vectors
x_test = []
y_test = []
for _, row in test.iterrows():
    x_test.append(get_flow_features_values(row, flow_features))
    y_test.append(get_encoded_label(
        row, labels_map))

x_test = torch.FloatTensor(x_test)
y_test = torch.LongTensor(y_test)

print("Training model...")
model.fit(x_train, y_train)
print("Evaluating model...")

print(model.score(x_test, y_test))
