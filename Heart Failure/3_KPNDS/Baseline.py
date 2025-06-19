import warnings

import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from interpret.glassbox import ExplainableBoostingClassifier
import shap
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, \
    ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    accuracy_score
from sklearn.tree import DecisionTreeClassifier

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

rng = np.random.RandomState(42)
df = pd.read_csv('dat-embed.csv', encoding='gb2312', low_memory=False)

models = [
    (LogisticRegression(C=0.3, random_state=rng), 'LogisticRegression'),
    (ExtraTreesClassifier(n_estimators=300, random_state=rng), 'ExtraTrees'),
    (RandomForestClassifier(n_estimators=300, max_depth=15, max_features='sqrt', criterion='entropy', bootstrap=False,
                            random_state=rng), 'RandomForest'),
    (LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'), 'LDA'),
    (GradientBoostingClassifier(n_estimators=300, max_depth=10, learning_rate=0.1, subsample=0.8, random_state=rng),
     'GradientBoosting'),
    (SVC(C=3, kernel='rbf', probability=True, gamma='auto', random_state=rng), 'SVM'),
    (DecisionTreeClassifier(random_state=rng, max_depth=5, criterion='entropy', max_features=0.5), 'DecisionTree'),
    (AdaBoostClassifier(n_estimators=300, learning_rate=1, random_state=rng, algorithm='SAMME'), 'AdaBoost'),
    (QuadraticDiscriminantAnalysis(reg_param=0.3), 'QDA'),
    (
    LGBMClassifier(n_estimators=300, random_state=rng, min_child_samples=5, max_depth=-1, is_unbalance=True, verbose=-1,
                   min_child_weight=0.001), 'LGBM'),
    (xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                       reg_alpha=0.5, reg_lambda=1, random_state=rng, eval_metric='logloss'), 'XGBoost')
    ]
models += [
    (TabNetClassifier(n_d=128, n_a=128, n_steps=10, gamma=2, seed=42, mask_type='entmax'), 'TabNet'),
    (ExplainableBoostingClassifier(interactions=0,max_rounds=200,learning_rate=0.01,random_state=42), 'EBM-GAM'),

]

ds_metrics = {name: [] for _, name in models}
shap_data = {}
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rng)
scaler = StandardScaler()
for fold, (train_index, test_index) in enumerate(skf.split(df, df['re.admission.within.3.months'])):
    print(f"Fold {fold + 1}")
    train_data = df.iloc[train_index]
    test_data = df.iloc[test_index]
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=rng)

    X_train = train_data.drop(columns=['inpatient.number', 'death.within.28.days', 'death.within.3.months', 're.admission.within.3.months', 'death.within.6.months', 're.admission.within.6.months', 're.admission.within.28.days'])
    y_train = train_data.iloc[:, -1]
    X_train = scaler.fit_transform(X_train)

    X_val = scaler.transform(val_data.drop(columns=['inpatient.number', 'death.within.28.days', 'death.within.3.months', 're.admission.within.3.months', 'death.within.6.months', 're.admission.within.6.months', 're.admission.within.28.days']))
    y_val = val_data['re.admission.within.3.months']

    X_test = scaler.transform(test_data.drop(columns=['inpatient.number', 'death.within.28.days', 'death.within.3.months', 're.admission.within.3.months', 'death.within.6.months', 're.admission.within.6.months', 're.admission.within.28.days']))
    y_test = test_data['re.admission.within.3.months']
    for model, name in models:
        print(name)
        if name == 'TabNet':
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric=['auc'],
                patience=10,
                max_epochs=100,
                batch_size=512,
            )
        else:
            model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        mcc = matthews_corrcoef(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        print(cm)
        ds_metrics[name].append({
            'mcc': mcc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        })
for name, metrics in ds_metrics.items():
    print(f"\n{name} Average Results:")
    print(f"MCC: {np.mean([m['mcc'] for m in metrics]):.4f}, {np.std([m['mcc'] for m in metrics]):.4f}")
    print(f"Precision: {np.mean([m['precision'] for m in metrics]):.4f}, {np.std([m['precision'] for m in metrics]):.4f}")
    print(f"Recall: {np.mean([m['recall'] for m in metrics]):.4f}, {np.std([m['recall'] for m in metrics]):.4f}")
    print(f"F1 Score: {np.mean([m['f1'] for m in metrics]):.4f}, {np.std([m['f1'] for m in metrics]):.4f}")
    print(f"ROC-AUC: {np.mean([m['roc_auc'] for m in metrics]):.4f}, {np.std([m['roc_auc'] for m in metrics]):.4f}")