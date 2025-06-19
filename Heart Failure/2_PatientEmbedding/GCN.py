'''GCN embedding'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import pandas as pd
import json
from networkx.readwrite import json_graph
import numpy as np

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('patient_drug_network.json', 'r') as f:
    data = json.load(f)
G = json_graph.node_link_graph(data)
def convert_to_pyg_data(G):
    node_to_index = {node: i for i, node in enumerate(G.nodes)}
    edge_index = []
    for u, v in G.edges:
        edge_index.append([node_to_index[u], node_to_index[v]])
        edge_index.append([node_to_index[v], node_to_index[u]])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    num_nodes = len(G.nodes)
    num_features = 128
    node_features = torch.rand((num_nodes, num_features), dtype=torch.float)  # 随机初始化特征矩阵

    data = Data(x=node_features, edge_index=edge_index)
    return data

data = convert_to_pyg_data(G).to(device)

# GCN
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, dropout=0.5):
        super(GCN, self).__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.conv2 = GCNConv(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.conv3 = GCNConv(hidden_dim2, output_dim)
        self.bn3 = nn.BatchNorm1d(output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.bn1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.bn2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        return x
input_dim = data.x.size(1)
hidden_dim1 = 256
hidden_dim2 = 128
output_dim = 128
learning_rate = 0.001
num_epochs = 500
dropout = 0.5

model = GCN(input_dim, hidden_dim1, hidden_dim2, output_dim, dropout).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = torch.norm(out)
    loss.backward()
    optimizer.step()
    print(f'Epoch: {epoch + 1}, Loss: {loss.item()}')

patients = [node for node in G.nodes() if G.nodes[node]['type'] == 'patient']
patient_indices = [list(G.nodes).index(patient) for patient in patients]
patient_embeddings = out[patient_indices].detach().cpu().numpy()

df_train = pd.read_csv('dat.csv',encoding='gb2312', low_memory=False)
def add_embeddings_to_df(original_df, embeddings_tensor, patients):
    embeddings_dict = {str(patient): embeddings_tensor[i] for i, patient in enumerate(patients)}
    original_df['embedding'] = original_df['inpatient.number'].apply(
        lambda x: embeddings_dict.get(str(x), np.zeros(embeddings_tensor.shape[1])))
    embedding_features = pd.DataFrame(original_df['embedding'].tolist(), index=original_df.index)
    merged_df = pd.concat([original_df, embedding_features], axis=1).drop(['embedding'], axis=1)
    return merged_df


df_train = add_embeddings_to_df(df_train, patient_embeddings, patients)
df_train.to_csv(
    'dat-embed.csv', encoding='gb2312', index=False)