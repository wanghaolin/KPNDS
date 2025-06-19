import json
from networkx.readwrite import json_graph
from stellargraph import StellarGraph
from stellargraph.data import UniformRandomMetaPathWalk
from gensim.models import Word2Vec
import numpy as np
import pandas as pd

df_train = pd.read_csv('dat.csv', encoding='gb2312', low_memory=False)

# 从JSON文件加载患者网络
with open('patient_drug_network.json', 'r') as f:
    data = json.load(f)
patient_network = json_graph.node_link_graph(data)

# Convert NetworkX graph to StellarGraph
G = StellarGraph.from_networkx(patient_network,node_type_attr='type')
print(G.info())

# define metapath
metapaths = [
    ["patient", "drug", "patient"],
]

nodes = [str(node) for node in G.nodes()]
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
# Train node embeddings using gensim's Word2Vec model
model = Word2Vec(
    # walks,
    vector_size=128,
    window=5,
    min_count=0,
    sg=1,
    workers=4,
    epochs=100,
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

patient_embeddings = {}
for node in G.nodes():
    if G.node_type(node) == "patient":
        node_str = str(node)
        try:
            patient_embeddings[node_str] = model.wv[node_str]
        except KeyError:
            print(f"Warning: Node {node_str} not in vocabulary, using zero vector")
            patient_embeddings[node_str] = np.zeros(model.vector_size)

def add_embeddings_to_df(original_df, embeddings_dict, id_column='inpatient.number', dimensions=128):
    original_df[id_column] = original_df[id_column].astype(str)
    original_df['embedding'] = original_df[id_column].apply(lambda x: embeddings_dict[str(x)] if str(x) in embeddings_dict else np.zeros(dimensions))
    embedding_features = pd.DataFrame(original_df['embedding'].tolist(), index=original_df.index)
    merged_df = pd.concat([original_df, embedding_features], axis=1)
    merged_df.drop(['embedding'], axis=1, inplace=True)
    return merged_df

df_train = add_embeddings_to_df(df_train, patient_embeddings)
df_train.to_csv('dat-embed.csv', encoding='gb2312', index=False)
