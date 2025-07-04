'''MLP as meta-learner to select classifiers for every query instance'''
import warnings
from interpret.glassbox import ExplainableBoostingClassifier
from lightgbm import LGBMClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, \
    ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

class ClassifierWithThreshold:
    def __init__(self, classifier, threshold):
        self.classifier = classifier
        self.threshold = threshold
    def predict(self, X):
        probabilities = self.classifier.predict_proba(X)[:, 1]
        return (probabilities >= self.threshold).astype(int)
    def predict_proba(self, X):
        return self.classifier.predict_proba(X)

rng = np.random.RandomState(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
df = pd.read_csv('dat-embed.csv', encoding='gb2312', low_memory=False)

# base classifier pool
models = [
    (LogisticRegression(C=0.3, random_state=rng), 'LogisticRegression'),
    (ExtraTreesClassifier(n_estimators=300, random_state=rng), 'ExtraTrees'),
    (RandomForestClassifier(n_estimators=300, max_depth=15, max_features='sqrt', criterion='entropy', bootstrap=False, random_state=rng), 'RandomForest'),
    (LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'), 'LDA'),
    (GradientBoostingClassifier(n_estimators=300, max_depth=10, learning_rate=0.1, subsample=0.8, random_state=rng), 'GradientBoosting'),
    (SVC(C=3, kernel='rbf', probability=True, gamma='auto', random_state=rng), 'SVM'),
    (DecisionTreeClassifier(random_state=rng, max_depth=5, criterion='entropy', max_features=0.5), 'DecisionTree'),
    (AdaBoostClassifier(n_estimators=300,learning_rate=1, random_state=rng, algorithm='SAMME'), 'AdaBoost'),
    (QuadraticDiscriminantAnalysis(reg_param=0.3), 'QDA'),
    (LGBMClassifier(n_estimators=300, random_state=rng, min_child_samples=5, max_depth=-1,is_unbalance=True, verbose=-1, min_child_weight=0.001), 'LGBM'),
    (xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                       reg_alpha=0.5, reg_lambda=1, random_state=rng, eval_metric='logloss'), 'XGBoost'),
    (TabNetClassifier(n_d=128, n_a=128, n_steps=10, gamma=2, seed=42, mask_type='entmax'), 'TabNet'),
    (ExplainableBoostingClassifier(interactions=0, max_rounds=200, learning_rate=0.01, random_state=42), 'EBM')
]

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rng)
scaler = StandardScaler()
scaler_embed = StandardScaler()
columns = [f'col_{i}' for i in range(115, 243)]  # Embedding column

model_mcc = []
model_precision = []
model_recall = []
model_f1 = []
model_roc_auc = []
for fold, (train_index, test_index) in enumerate(skf.split(df, df['re.admission.within.28.days'])):
    print(f"Fold {fold + 1}")
    # Divide the training set and test set
    train_data = df.iloc[train_index]
    test_data = df.iloc[test_index]

    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=rng)

    X_train_embed = train_data.iloc[:, 115:243]
    embed_columns = list(X_train_embed.columns)
    print('X_train_embed：', embed_columns)
    X_train_global = train_data.drop(columns=['inpatient.number', 'death.within.28.days', 'death.within.3.months', 're.admission.within.3.months', 'death.within.6.months', 're.admission.within.6.months', 're.admission.within.28.days'] + embed_columns)
    y_train_global = train_data['re.admission.within.28.days']
    X_train_global = scaler.fit_transform(X_train_global)
    X_train_embed = scaler_embed.fit_transform(X_train_embed)

    X_val = scaler.transform(val_data.drop(columns=['inpatient.number', 'death.within.28.days', 'death.within.3.months', 're.admission.within.3.months', 'death.within.6.months', 're.admission.within.6.months', 're.admission.within.28.days'] + embed_columns))
    X_val_embed = val_data.iloc[:, 115:243]
    X_val_embed = scaler_embed.transform(X_val_embed)
    y_val = val_data['re.admission.within.28.days']

    X_test_embed = test_data.iloc[:, 115:243]
    print('X_test_embed:', X_test_embed.columns)
    X_test_embed = scaler_embed.transform(X_test_embed)
    X_test = scaler.transform(test_data.drop(columns=['inpatient.number', 'death.within.28.days', 'death.within.3.months', 're.admission.within.3.months', 'death.within.6.months', 're.admission.within.6.months', 're.admission.within.28.days'] + embed_columns))  # + disease_conditions
    y_test = test_data['re.admission.within.28.days']

    MIN_MCC_1 = 0.30
    MIN_RECALL_1 = 0.65
    MIN_F1_1 = 0.76
    global_classifier_pool = []
    for model, name in models:
        #  shelter classifier in the pool
        global_model = clone(model)
        if name == 'TabNet':
            global_model.fit(
                X_train_global, y_train_global,
                eval_set=[(X_val, y_val)],
                eval_metric=['auc'],
                patience=10,
                max_epochs=100,
                batch_size=512,
            )
        else:
            global_model.fit(X_train_global, y_train_global)
        y_pred_proba_val = global_model.predict_proba(X_val)[:, 1]
        best_threshold, best_mcc, best_f1 = 0.5, -np.inf, -np.inf
        for threshold in np.arange(0.1, 1.0, 0.05):
            y_pred_val = (y_pred_proba_val >= threshold).astype(int)
            mcc_val = matthews_corrcoef(y_val, y_pred_val)
            f1_val = f1_score(y_val, y_pred_val, pos_label=1, zero_division=0)
            # record best_threshold
            if mcc_val > best_mcc and f1_val > best_f1:
                best_threshold, best_mcc, best_f1 = threshold, mcc_val, f1_val
        while best_f1 < MIN_F1_1 and MIN_F1_1 > 0.12:
            MIN_F1_1 -= 0.02
        if best_mcc >= MIN_MCC_1 or best_f1 >= MIN_F1_1:
            global_classifier_pool.append(ClassifierWithThreshold(global_model, best_threshold))
    print("global classifier number：", len(global_classifier_pool))

    final_classifier_pool = global_classifier_pool
    if len(final_classifier_pool) == 0:
        raise ValueError("please reduce MIN_MCC or MIN_RECALL threshold")
    classifier_sources = ["Global"] * len(global_classifier_pool)
    print('Classifier pool constructed successfully')
    y_classifier_train = np.zeros((X_train_embed.shape[0], len(final_classifier_pool)))
    for i in range(X_train_embed.shape[0]):
        true_label = y_train_global.iloc[i]
        patient = X_train_global[i]
        for classifier_idx, classifier_with_threshold in enumerate(final_classifier_pool):
            pred = classifier_with_threshold .predict(patient.reshape(1, -1))
            if pred == true_label:
                y_classifier_train[i, classifier_idx] = 1  # Multi label format

    # X_val_embed multi-label: y_classifier_val
    y_classifier_val = np.zeros((X_val_embed.shape[0], len(final_classifier_pool)))
    for i in range(X_val_embed.shape[0]):
        true_label = y_val.iloc[i]
        patient = X_val[i]
        for classifier_idx, classifier in enumerate(final_classifier_pool):
            pred = classifier.predict(patient.reshape(1, -1))
            if pred == true_label:
                y_classifier_val[i, classifier_idx] = 1

    X_train_embed_torch = torch.tensor(X_train_embed, dtype=torch.float32).to(device)
    y_classifier_train_torch = torch.tensor(y_classifier_train, dtype=torch.float32).to(device)
    X_val_embed_torch = torch.tensor(X_val_embed, dtype=torch.float32).to(device)
    y_val_torch = torch.tensor(y_val.values, dtype=torch.float32).to(device)
    y_classifier_val_torch = torch.tensor(y_classifier_val, dtype=torch.float32).to(device)
    print("X_train_embed shape:", X_train_embed.shape)
    print("y_classifier_train shape:", y_classifier_train.shape)
    print("y_classifier_val shape:", y_classifier_val.shape)
    print("X_test_embed shape:", X_test_embed.shape)

    # MLP
    class MLP(nn.Module):
        def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size1)
            self.bn1 = nn.BatchNorm1d(hidden_size1)
            self.fc2 = nn.Linear(hidden_size1, hidden_size2)
            self.bn2 = nn.BatchNorm1d(hidden_size2)
            self.fc3 = nn.Linear(hidden_size2, output_size)
            self.dropout = nn.Dropout(p=0.5)

        def forward(self, x):
            x = torch.relu(self.bn1(self.fc1(x)))
            x = self.dropout(x)
            x = torch.relu(self.bn2(self.fc2(x)))
            x = self.dropout(x)
            x = self.fc3(x)
            return x

    # Train the MLP model to select the base classifier
    input_size = X_train_embed.shape[1]
    hidden_size1 = 256
    hidden_size2 = 128
    output_size = len(final_classifier_pool)
    mlp_selector = MLP(input_size=input_size, hidden_size1=hidden_size1, hidden_size2=hidden_size2, output_size=output_size)

    mlp_selector.to(device)
    criterion = nn.BCEWithLogitsLoss()  # Loss
    optimizer = optim.Adam(mlp_selector.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    for epoch in range(300):
        mlp_selector.train()
        optimizer.zero_grad()
        outputs = mlp_selector(X_train_embed_torch)
        loss = criterion(outputs, y_classifier_train_torch)
        loss.backward()
        optimizer.step()
        # valid
        mlp_selector.eval()
        with torch.no_grad():
            val_outputs = mlp_selector(X_val_embed_torch)
            val_loss = criterion(val_outputs, y_classifier_val_torch)
        scheduler.step(val_loss)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/100], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')
    X_test_embed_torch = torch.tensor(X_test_embed, dtype=torch.float32).to(device)
    threshold = 0.8
    # Use MLP to output the propensity of each classifier
    mlp_selector.eval()
    with torch.no_grad():
        classifier_probs = mlp_selector(X_test_embed_torch)
    classifier_probs = torch.sigmoid(classifier_probs)
    print("Classifier probabilities distribution:", classifier_probs)
    # Majority voting
    y_pred_votes = []
    y_pred_proba = []
    for i in range(len(X_test)):
        sample = X_test[i].reshape(1, -1)
        sample_probs = classifier_probs[i]
        selected_classifiers_for_sample = (sample_probs > threshold).nonzero(as_tuple=False).squeeze()
        if selected_classifiers_for_sample.numel() == 0:
            selected_classifiers_for_sample = torch.argmax(sample_probs).unsqueeze(0)
        if isinstance(selected_classifiers_for_sample, int):
            selected_classifiers_for_sample = [selected_classifiers_for_sample]
        elif isinstance(selected_classifiers_for_sample, torch.Tensor):
            if selected_classifiers_for_sample.dim() == 0:
                selected_classifiers_for_sample = [selected_classifiers_for_sample.item()]
            else:
                selected_classifiers_for_sample = selected_classifiers_for_sample.tolist()

        selected_classifiers_info = [(classifier_idx, classifier_sources[classifier_idx])
                                 for classifier_idx in selected_classifiers_for_sample]
        print(f"Sample {i} selected classifiers: {selected_classifiers_info}")

        votes = []
        proba = []
        for classifier_idx in selected_classifiers_for_sample:
            classifier_with_threshold = final_classifier_pool[classifier_idx]
            probabilities = classifier_with_threshold.predict_proba(sample.reshape(1, -1))[:, 1]
            pred = classifier_with_threshold.predict(sample.reshape(1, -1))
            votes.append(pred[0])
            proba.append(probabilities.ravel())
        votes = np.array(votes).ravel()
        final_prediction = np.bincount(votes).argmax()
        y_pred_votes.append(final_prediction)
        if proba:
            avg_prob = np.mean(proba, axis=0)
            y_pred_proba.append(avg_prob)
    y_pred = np.array(y_pred_votes)
    y_pred_proba = np.array(y_pred_proba)

    mcc = matthews_corrcoef(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    model_mcc.append(mcc)
    model_precision.append(precision)
    model_recall.append(recall)
    model_f1.append(f1)
    model_roc_auc.append(roc_auc)
avg_mcc = np.mean(model_mcc)
avg_precision = np.mean(model_precision)
avg_recall = np.mean(model_recall)
avg_f1 = np.mean(model_f1)
avg_roc_auc = np.mean(model_roc_auc)
# Std
mcc_std = np.std(model_mcc)
precision_std = np.std(model_precision)
recall_std = np.std(model_recall)
f1_std = np.std(model_f1)
roc_auc_std = np.std(model_roc_auc)
print(f'Average MCC: {avg_mcc:.3f}(±{mcc_std:.3f})')
print(f'Average Precision: {avg_precision:.3f}(±{precision_std:.3f})')
print(f'Average Recall: {avg_recall:.3f}(±{recall_std:.3f})')
print(f'Average F1 Score: {avg_f1:.3f}(±{f1_std:.3f})')
print(f'Average ROC-AUC: {avg_roc_auc:.3f}(±{roc_auc_std:.3f})')