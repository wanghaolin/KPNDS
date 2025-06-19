'''GAT embedding'''
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from networkx.readwrite import json_graph

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 从JSON文件加载患者网络
with open('patient_drug_network.json','r') as f:
    data = json.load(f)
G = json_graph.node_link_graph(data)

# Transfer to PyTorch Geometric
def convert_to_pyg_data(G):
    node_to_index = {node: i for i, node in enumerate(G.nodes)}
    edge_index = []
    for u, v in G.edges:
        edge_index.append([node_to_index[u], node_to_index[v]])
        edge_index.append([node_to_index[v], node_to_index[u]])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # one-hot encoding
    node_types = ['patient' if G.nodes[node]['type'] == 'patient' else 'drug' for node in G.nodes]
    type_to_idx = {'patient': 0, 'drug': 1}
    node_features = torch.zeros((len(G.nodes), 2), dtype=torch.float)
    for i, node_type in enumerate(node_types):
        node_features[i, type_to_idx[node_type]] = 1.0
    data = Data(x=node_features, edge_index=edge_index)
    return data, node_to_index

data, node_to_index = convert_to_pyg_data(G)
data = data.to(device)
# GAT
class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5, heads=8):
        super(GAT, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout)
        self.conv3 = GATConv(hidden_dim * heads, output_dim, heads=1, concat=False, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads)
        self.bn2 = nn.BatchNorm1d(hidden_dim * heads)
        self.bn3 = nn.BatchNorm1d(output_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(self.bn1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(self.bn2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return self.bn3(x)
# loss
def graph_loss(embeddings, edge_index, margin=1.0):
    src, dst = edge_index
    pos_dists = torch.norm(embeddings[src] - embeddings[dst], dim=1)
    pos_loss = torch.mean(pos_dists)
    num_neg = min(edge_index.size(1), embeddings.size(0))
    neg_idx1 = torch.randint(0, embeddings.size(0), (num_neg,))
    neg_idx2 = torch.randint(0, embeddings.size(0), (num_neg,))

    # Make sure it's not self-looping
    mask = neg_idx1 != neg_idx2
    neg_idx1 = neg_idx1[mask]
    neg_idx2 = neg_idx2[mask]
    neg_dists = torch.norm(embeddings[neg_idx1] - embeddings[neg_idx2], dim=1)
    neg_loss = torch.mean(F.relu(margin - neg_dists))
    return pos_loss + neg_loss

input_dim = data.x.size(1)
hidden_dim = 256
output_dim = 128  # Embedding dimension
learning_rate = 0.001
num_epochs = 500

model = GAT(input_dim, hidden_dim, output_dim, dropout=0.3, heads=4).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)

# Train
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = graph_loss(out, data.edge_index)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch: {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index).cpu().numpy()

patients = [node for node in G.nodes if G.nodes[node]['type'] == 'patient']
patient_indices = [node_to_index[patient] for patient in patients]
patient_embeddings = out[patient_indices]

# Check embedding value
non_zero_count = np.count_nonzero(patient_embeddings)
total_elements = patient_embeddings.size
print(f"Non-zero embeddings: {non_zero_count}/{total_elements} ({non_zero_count / total_elements:.2%})")

# Add the embedding to the dat.csv
df_train = pd.read_csv(
    'dat.csv',
    encoding='gb2312',
    low_memory=False,
    dtype={'inpatient.number': str})

def add_embeddings_to_df(original_df, embeddings_array, patients, id_column='inpatient.number'):
    patients_str = [str(p) for p in patients]
    embeddings_dict = {patients_str[i]: embeddings_array[i] for i in range(len(patients_str))}

    original_df[id_column] = original_df[id_column].astype(str)
    original_df['embedding'] = original_df[id_column].apply(
        lambda x: embeddings_dict.get(x, np.zeros(embeddings_array.shape[1])))
    embedding_features = pd.DataFrame(original_df['embedding'].tolist(), index=original_df.index)
    merged_df = pd.concat([original_df, embedding_features], axis=1)
    merged_df.drop(['embedding'], axis=1, inplace=True)
    return merged_df


df_train = add_embeddings_to_df(df_train, patient_embeddings, patients)
df_train.to_csv('dat-embed.csv',encoding='gb2312',index=False)
print("Embeddings saved successfully.")