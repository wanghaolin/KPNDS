'''Building a Drug-Patient network for patients'''
import pandas as pd
import networkx as nx

new_df = pd.read_csv('heart-failure\\dat_md.csv', encoding='gb2312')
patient_drug_network = nx.Graph()


for patient_id, group in new_df.groupby('inpatient.number'):
    # patient node
    patient_drug_network.add_node(patient_id, type='patient')
    for drug_name in group['Drug_name'].unique():
        clean_drug = drug_name.strip().lower()

        # drug node
        patient_drug_network.add_node(clean_drug, type='drug')

        # patient-drug edge
        patient_drug_network.add_edge(patient_id, clean_drug)

# JSON format
import json
from networkx.readwrite import json_graph

data = json_graph.node_link_data(patient_drug_network)
with open('patient_drug_network.json', 'w') as f:
    json.dump(data, f)