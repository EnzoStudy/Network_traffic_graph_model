""""
Experiment to check if the features are being key in the classification problem,
or if the graph structure is having impact.
"""

import torch
import torch.nn as nn
from data import load_csv_to_df
from graph import get_flow_features_values, get_encoded_label
from metrics import compute_metrics
from plot import plot_ce_loss_curve
from metrics import compute_confusion_matrix
from data import modify_portscan_attack_behavior
import configparser

config = configparser.ConfigParser()
config.read("config.ini")

# Same NN used for readout in the GNN model


class Net(nn.Module):

    def __init__(self, input_channels, hidden_channels, output_channels):
        super(Net, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels

        self.readout_nn = nn.Sequential(
            nn.Linear(input_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_channels, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, self.output_channels),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.readout_nn(x)


train = load_csv_to_df(
    config["PARAMETERS"]["DataFolderPath"] + config["PARAMETERS"]["TrainFile"])

test = load_csv_to_df(
    config["PARAMETERS"]["DataFolderPath"] + config["PARAMETERS"]["EvalFile"])


labels_map = {"BENIGN": 0, "PortScan": 1}
labels = ["BENIGN", "MALIGN"]

flow_features = [
    " Average Packet Size",
    " Flow IAT Mean",
    " Flow Duration",
]
# Add a synthetic modification to the test dataset
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


def train(model, x, y, optimizer):
    crit = torch.nn.CrossEntropyLoss()
    model.train()
    optimizer.zero_grad()
    out = model(x)
    loss = crit(out, y)
    loss.backward()
    optimizer.step()
    return float(loss)


@ torch.no_grad()
def test(model, x, y):
    model.eval()
    out = model(x)
    _, pred = torch.max(out, 1)
    compute_confusion_matrix(y, pred, classes=labels)
    return compute_metrics(y, pred)


model = Net(
    input_channels=len(flow_features),
    hidden_channels=128,
    output_channels=len(labels)
)

# Load state dict of saves model
try:
    model.load_state_dict(torch.load(config["PARAMETERS"]["ModelPath"]))
    print("Loading model...")
except FileNotFoundError:
    print("Creating new model...")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

loss_arr = []
print("Starting training phase...")
for epoch in range(int(config["PARAMETERS"]["NTrainEpochs"])):
    loss = train(model, x_train, y_train, optimizer)
    loss_arr.append(loss)
    print("Epoch: " + str(epoch))


plot_ce_loss_curve(int(config["PARAMETERS"]["NTrainEpochs"]), loss_arr)

print("Starting validation phase...")
metrics = test(model, x_test, y_test)
print(metrics)

torch.save(model.state_dict(), config["PARAMETERS"]["ModelPath"])
