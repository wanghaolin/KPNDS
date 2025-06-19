import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from networkx.readwrite import json_graph
from torch_geometric.utils import from_networkx

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load PatientNetwork from JSON files
with open('Heterogeneous-PatientNetwork.json', 'r') as f:
    data = json.load(f)
# print(data.keys())
G = json_graph.node_link_graph(data)
for node, data in G.nodes(data=True):
    print(node, data)
    break
for node, attrs in G.nodes(data=True):
    if 'type' not in attrs:
        print(f"Node {node} has no 'type' attribute.")
# Data objects converted to PyTorch Geometric
def convert_to_pyg_data(G):
    node_to_index = {node: i for i, node in enumerate(G.nodes)}
    edge_index = []
    for u, v in G.edges:
        edge_index.append([node_to_index[u], node_to_index[v]])
        edge_index.append([node_to_index[v], node_to_index[u]])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Randomly initialize the feature matrix
    num_nodes = len(G.nodes)
    num_features = 128
    node_features = torch.rand((num_nodes, num_features), dtype=torch.float)

    data = Data(x=node_features, edge_index=edge_index)
    return data

data = convert_to_pyg_data(G)
data = data.to(device)

# Defining the GCN model
class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5, heads=8):
        super(GAT, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * heads, output_dim, heads=1, concat=False, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads)
        self.bn2 = nn.BatchNorm1d(output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.bn1(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        return x

input_dim = data.x.size(1)
hidden_dim = 256
output_dim = 128
learning_rate = 0.001
num_epochs = 500

model = GAT(input_dim, hidden_dim, output_dim, dropout=0.5, heads=8).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = torch.norm(out)
    loss.backward()
    optimizer.step()
    print(f'Epoch: {epoch + 1}, Loss: {loss.item()}')

# add embeddings to ADR
patients = [node for node in G.nodes() if G.nodes[node]['type'] == 'patient']
patient_indices = [list(G.nodes).index(patient) for patient in patients]
patient_embeddings = out[patient_indices].detach().cpu().numpy()

df = pd.read_csv('ADR.csv', encoding='gb2312', low_memory=False)
def add_embeddings_to_df(original_df, embeddings_array, patients):
    embeddings_dict = {patients[i]: embeddings_array[i] for i in range(len(patients))}
    original_df['embedding'] = original_df['PatientID'].apply(lambda x: embeddings_dict.get(str(x), np.zeros(embeddings_array.shape[1])))
    embedding_features = pd.DataFrame(original_df['embedding'].tolist(), index=original_df.index)
    merged_df = pd.concat([original_df, embedding_features], axis=1)
    merged_df.drop(['embedding'], axis=1, inplace=True)
    return merged_df

df = add_embeddings_to_df(df, patient_embeddings, patients)
df.to_csv('ADR-embedding-GAT.csv', encoding='gb2312', index=False)
