import json
from networkx.readwrite import json_graph
import networkx as nx
import pandas as pd

'''Develop a Heterogeneous Knowledge Network for Patients'''

df = pd.read_csv('ADR.csv', encoding='gb2312', low_memory=False)

drugs_list = ['penicillin', 'cephalosporin', '...']
symptoms_list = ['fever', 'cough', '...']
diseases_list = ['pneumonia', 'respiratory infection', '...']

selected_columns = ['PatientID'] + drugs_list + symptoms_list + diseases_list
df_filtered = df[selected_columns]

# Initializing a heterogeneous network graph
B = nx.Graph()

# Add nodes and edges
for index, row in df_filtered.iterrows():
    patient_id = row['PatientID']
    B.add_node(patient_id, type='patient')

    for drug in drugs_list:
        if row[drug] == 1:
            B.add_node(drug, type='drug')
            B.add_edge(patient_id, drug)

    for symptom in symptoms_list:
        if row[symptom] == 1:
            B.add_node(symptom, type='symptom')
            B.add_edge(patient_id, symptom)

    for disease in diseases_list:
        if row[disease] == 1:
            B.add_node(disease, type='disease')
            B.add_edge(patient_id, disease)

# Gene,Phenotype,SIDER are additional introduced Domain Knowledge
# Gene data, Gene and Drug Connections
gene_file_path = 'CTD.csv'
gene_df = pd.read_csv(gene_file_path, encoding='gb2312', low_memory=False)
genes_list = gene_df.columns.tolist()[1:]
for _, row in gene_df.iterrows():
    drug_name = row['Chemical Names']
    if drug_name in drugs_list:
        for gene in genes_list:
            if row[gene] == 1:
                B.add_node(gene, type='gene')
                B.add_edge(drug_name, gene)

# Phenotype data, Phenotype and Gene Connections
phenotype_gene_path = 'CTD-phenotype.csv'
phenotype_gene_df = pd.read_csv(phenotype_gene_path, encoding='gb2312', low_memory=False)
phenotype_ids = phenotype_gene_df.iloc[:, 0]
genes_in_phenotype_df = phenotype_gene_df.columns[1:]

for _, row in phenotype_gene_df.iterrows():
    phenotype_id = row[0]
    B.add_node(phenotype_id, type='phenotype')
    for gene in genes_in_phenotype_df:
        if row[gene] == 1:
            if gene in B.nodes:
                B.add_edge(phenotype_id, gene)

# SIDER data to add drug --> Adverse Reaction Node
c_file_path = 'SIDER_table.csv'
c_df = pd.read_csv(c_file_path, encoding='gb2312')
adr_column = c_df.iloc[:, 0]
for adr in adr_column.unique():
    B.add_node(adr, type='ADR')
drug_cid_map = {
    'penicillin': ['CID100002349', 'CID100003040', 'CID100004834', 'CID100002171', 'CID100002173', 'CID100003364',
                 'CID100060706', 'CID100150610'],
    'cephalosporin': ['CID100002617', 'CID100002610', 'CID100002631', 'CID100047419', 'CID100002609', 'CID100002637',
                 'CID106398970', 'CID100002675', 'CID100002650', 'CID100002655', 'CID100444006', 'CID100054547',
                 'CID100002622', 'CID100002646'],
    'ibuprofen': ['CID100003672'],
    'drug': ['CID'],
}
# Add the edge between the drug and the ADR
for drug, cids in drug_cid_map.items():
    for cid in cids:
        if cid in c_df.columns:
            cid_index = c_df.columns.get_loc(cid)
            for index, row in c_df.iterrows():
                adr = row[0]
                if row[cid_index] == 1 and not B.has_edge(drug, adr):
                    B.add_edge(drug, adr)
# print(list(B.neighbors('ibuprofen')))
nx.write_gexf(B, "Heterogeneous-PatientNetwork.gexf")
# Convert to JSON format
data = json_graph.node_link_data(B)
with open('Heterogeneous-PatientNetwork.json', 'w') as f:
    json.dump(data, f)