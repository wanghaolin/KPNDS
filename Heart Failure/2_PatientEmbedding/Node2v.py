import pandas as pd
import json
from networkx.readwrite import json_graph
from node2vec import Node2Vec
import numpy as np

df_train = pd.read_csv('dat.csv', encoding='gb2312', low_memory=False)


with open('patient_drug_network.json', 'r') as f:
    data = json.load(f)
patient_network = json_graph.node_link_graph(data)
nodes = list(patient_network.nodes())

# Node2Vec parameter
dimensions = 128
walk_length = 100
num_walks = 30
workers = 5
batch_size = 400

embeddings_dict = {}
for i in range(0, len(nodes), batch_size):
    batch_nodes = nodes[i:i + batch_size]
    subgraph = patient_network.subgraph(batch_nodes).copy()
    node2vec = Node2Vec(subgraph, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers)
    model = node2vec.fit(window=10, min_count=1, batch_words=8)
    for node in batch_nodes:
        if str(node) in model.wv:
            embeddings_dict[str(node)] = model.wv[str(node)]

def add_embeddings_to_df(original_df, embeddings_dict):
    original_df['embedding'] = original_df['inpatient.number'].apply(lambda x: embeddings_dict[str(x)] if str(x) in embeddings_dict else np.zeros(dimensions))
    embedding_features = pd.DataFrame(original_df['embedding'].tolist(), index=original_df.index)
    merged_df = pd.concat([original_df, embedding_features], axis=1)
    merged_df.drop(['embedding'], axis=1, inplace=True)
    return merged_df

df_train = add_embeddings_to_df(df_train, embeddings_dict)
df_train.to_csv('dat-embed.csv',encoding='gb2312', index=False)