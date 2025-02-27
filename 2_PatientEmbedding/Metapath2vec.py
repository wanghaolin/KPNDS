import json
from networkx.readwrite import json_graph
from stellargraph import StellarGraph
from stellargraph.data import UniformRandomMetaPathWalk
from gensim.models import Word2Vec
import numpy as np
import pandas as pd

df = pd.read_csv('ADR', encoding='gb2312', low_memory=False)

# Load Bipartite-PatientNetwork from JSON files
with open('Bipartite-PatientNetwork.json', 'r') as f:
    data = json.load(f)
patient_network = json_graph.node_link_graph(data)

# Convert NetworkX graphs to StellarGraph
G = StellarGraph.from_networkx(patient_network,node_type_attr='type')
print(G.info())

# Define meta-paths
metapaths = [
    ["patient", "drug", "patient"],
    ["patient", "symptom", "patient"],
    ["patient", "disease", "patient"],
    ["patient", "drug", "gene", "drug", "patient"],
    ["patient", "drug", "ADR", "drug", "patient"],
    ["patient", "drug", "gene", "phenotype", "gene", "drug", "patient"]
]

rw = UniformRandomMetaPathWalk(G)
walks = rw.run(
    nodes=list(G.nodes()),
    length=200,
    n=30,
    metapaths=metapaths,
)
print(f"Number of walks: {len(walks)}")
if walks:
    print(f"Example walk: {walks[0]}")
walks = [[str(node) for node in walk] for walk in walks]
if walks:
    print(f"First walk length and type: {len(walks[0])}, type of first node: {type(walks[0][0])}")
# Train node embeddings using the Word2Vec model of gensim.
model = Word2Vec(
    # walks,
    vector_size=128,
    window=5,
    min_count=0,
    sg=1,
    workers=4,
    # epochs=1,
)
model.build_vocab(walks)
print("Vocabulary size:", len(model.wv.key_to_index))

# Train
model.train(
    walks,
    total_examples=model.corpus_count,
    epochs=100,
)
print("Training completed. Model is ready.")

# add embeddings to ADR
patient_embeddings = {node: model.wv[node] for node in G.nodes() if G.node_type(node) == "patient"}
def add_embeddings_to_df(original_df, embeddings_dict, dimensions=64):
    original_df['embedding'] = original_df['PatientID'].apply(lambda x: embeddings_dict[str(x)] if str(x) in embeddings_dict else np.zeros(dimensions))
    embedding_features = pd.DataFrame(original_df['embedding'].tolist(), index=original_df.index)
    merged_df = pd.concat([original_df, embedding_features], axis=1)
    merged_df.drop(['embedding'], axis=1, inplace=True)
    return merged_df
df = add_embeddings_to_df(df, patient_embeddings)
df.to_csv('ADR-embedding-metapath2vec.csv', encoding='gb2312', index=False)
