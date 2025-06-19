# Knowledge-augmented Patient Network Embedding-based Dynamic Model Selection for Predictive Analysis of Pediatric Drug-induced Liver Injury

This repository contains the supplementary materials related to the implementation of our submission entitled "Knowledge-augmented Patient Network Embedding-based Dynamic Model Selection for Predictive Analysis of Pediatric Drug-induced Liver Injury". This file includes the code for verifying KPNDS in two datasets. The DILI dataset with ethical restrictions is not publicly available, and the Heart Failure dataset is from a published paper (https://doi.org/10.1038/s41597-021-00835-9).

## Folder Structure and Functionality

### Folder 1: DILI

#### 1. PatientNetwork
- **Network Composition**: The network we developed consists of 19,396 nodes, categorized into seven types:
  - Patient (12,353, 63.69%)
  - Phenotype (5,159, 26.6%)
  - ADR (1,492, 7.69%)
  - Gene (330, 1.7%)
  - Drug (48, 0.25%)
  - Disease (9, 0.05%)
  - Symptom (5, 0.03%)
- **Data Sources**: Data for Patient, Drug, Disease, and Symptom nodes were sourced from EHRs. Phenotype, ADR, and Gene data were obtained from the Comparative Toxicogenomics Database (CTD) ([https://ctdbase.org/](https://ctdbase.org/)) and the Side Effect Resource (SIDER) databases ([http://sideeffects.embl.de/drugs/](http://sideeffects.embl.de/drugs/)).

#### 2. PatientEmbedding
- **Knowledge Enrichment**: We enrich the network with extra relevant knowledge.
- **Embedding Methods**: Several graph embedding methods are explored, such as Node2Vec, Metapath2Vec, GCN, and GAT, to capture latent information within the Knowledge-augmented Patient Network (KPN).

#### 3. KPNDS
- **Framework**: A meta-learning based framework is adopted to dynamically select the optimal classifiers based on the latent patient representations to perform individualized risk prediction.
- **Meta-classifiers**: Multi-Layer Perceptron, Transformer, and Kolmogorov-Arnold Networks are used as meta-classifiers to enhance the selection of the optimal classifiers for each patient.

### Folder 2: Heart Failure

#### 1. PatientNetwork
- **Pre.py**: This script is used for the preprocessing of Heart Failure data, including numerical mapping, average value filling, and filtering columns with more than 20% of null values.
- **Network.py**: It includes the bipartite network we constructed based on the patient-medication relationship extracted from EHRs.

#### 2. PatientEmbedding
- **Embedding Methods**: Node2Vec, Metapath2Vec, GCN, and GAT are used to capture latent information within the PatientNetwork.

#### 3. KPNDS
- **Meta-classifiers**: Similar to the DILI dataset, Multi-Layer Perceptron, Transformer, and Kolmogorov-Arnold Networks are used as meta-classifiers to enhance the selection of the optimal classifiers for each patient.

## How to Use

1. **Data Preparation**:
   - For the Heart Failure dataset, download the data and place it in the appropriate directory.
2. **Running the Code**:
   - Navigate to the relevant folder (DILI or Heart Failure) and run the scripts in the order of data preprocessing, network construction, embedding generation, and finally the KPNDS framework.

## Dependencies
The code relies on several Python libraries, including but not limited to `pandas`, `networkx`, `torch`, `gensim`, `stellargraph`, `catboost`, `lightgbm`, `sklearn`, etc. Make sure these libraries are installed in your Python environment before running the code. You can install them using `pip` or `conda`. For example:
```bash
pip install pandas networkx torch gensim stellargraph catboost lightgbm scikit-learn
