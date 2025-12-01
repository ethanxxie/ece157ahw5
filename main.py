import functions

import numpy as np
import pandas as pd

# Seaborn and Matplotlib for plotting
import seaborn as sns
import matplotlib.pyplot as plt

# Functions for data preprocessing 
from functions import prepare_data, get_salient_region, create_feature_columns, int2string, add_salient_region, process_wafer_map
from sklearn.preprocessing import StandardScaler


# Functons for model validation
from functions import print_accuracies, print_confusion_matrix

# Scikit-Learn and helper functions
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier




FAIL = 2 # failing die
PASS = 1 # passing die
NO_DIE = 0 # no die
RANDOM_SEED = 67

# ---------------------------------------------------------------------------------
#
#                                   Data Setup
#
# ---------------------------------------------------------------------------------

# Load the numpy array from the .npy file
data = np.load('data/wafermap_train.npy', allow_pickle = True)

# Create a DataFrame from the numpy array
df = pd.DataFrame(data)

# Rehsapes maps to 64 x 64
df_prepped = prepare_data(df)

# Adds a column for the salient region.
df_prepped['salientRegion'] = df_prepped.apply(get_salient_region, axis=1)

# create_feature_columns() creates the feature columns for the supervised model to learn
df_train = create_feature_columns(df_prepped)

# Define the features and target variable
feature_columns = ['areaRatio', 'perimeterRatio', 'maxDistFromCenter',
                   'minDistFromCenter', 'majorAxisRatio', 'minorAxisRatio',
                   'solidity', 'eccentricity', 'yieldLoss', 'edgeYieldLoss']
target_column = 'failureTypeNumber'

# Extract features (X) and target variable (y)
X = df_train[feature_columns]
y = df_train[target_column]

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

# scale the feature columns to improve model performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# ---------------------------------------------------------------------------------
#
#                           Model Fitting & Prediction
#
# ---------------------------------------------------------------------------------


print("----- Hyperparameter Tuning Results -----")

# # hyperparameter tuning using GridSearchCV for Random Forest and SVC
# param_grid_rf = {
#     'max_depth': [10, 20, None],          # max depth of trees
#     'min_samples_split': [2, 5, 10],      # min samples to split a node
#     'min_samples_leaf': [1, 2, 4],        # min samples in a leaf
#     'max_features': ['sqrt', 'log2', None], # features considered for split
# }

rf_model = RandomForestClassifier(
    n_estimators=200, 
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=RANDOM_SEED, 
    class_weight='balanced'
    )

# best_rf = GridSearchCV(rf_model, param_grid_rf, cv=5, scoring='accuracy')
rf_model.fit(X_train_scaled, y_train)
y_pred_val_rf = rf_model.predict(X_val_scaled)

# print(rf_model.best_params_)
# print(rf_model.best_score_)

# param_grid_svc = {
#     'C': [0.1, 1, 10],                   # regularization
#     'kernel': ['linear', 'rbf', 'poly'], # kernel type
#     'gamma': ['scale', 'auto'],           # kernel coefficient
#     'degree': [3, 4, 5]                   # degree for poly kernel
# }

svc_model = SVC(
    probability=True,
    C=1,
    degree=3,
    gamma='auto',
    kernel='poly',
    random_state=RANDOM_SEED
    )

# best_svc = GridSearchCV(svc_model, param_grid_svc, cv=5, scoring='accuracy')
svc_model.fit(X_train_scaled, y_train)
y_pred_val_best_svc = svc_model.predict(X_val_scaled)

# print(best_svc.best_params_)
# print(best_svc.best_score_)


# param_grid_mlp = {
#     'hidden_layer_sizes': [
#         (128, 128, 64),
#         (256, 128, 64),
#         (128, 256, 128),
#         (256, 256, 128),
#         (128, 256, 128, 64),
#         (256, 256, 128, 64)
#     ],
#     'activation': ['relu', 'tanh'],
#     'solver': ['adam'],
#     'alpha': np.logspace(-5, -2, 5),  # 1e-5, 1e-4, 1e-3, 1e-2
#     'learning_rate': ['constant', 'adaptive'],
#     'learning_rate_init': [0.001, 0.01]
# }
# 
# grid_mlp = GridSearchCV(
#     estimator=MLPClassifier(max_iter=5000, random_state=67),
#     param_grid=param_grid_mlp,
#     scoring='accuracy',
#     cv=5,
#     n_jobs=-1,
#     verbose=2
# )

mlp_model = MLPClassifier(
    max_iter=10000,
    random_state=67,
    hidden_layer_sizes=(128, 128, 64),
    activation='relu',
    solver='adam',
    alpha=0.00001,
    learning_rate='constant',
    learning_rate_init=0.001
)

# grid_mlp.fit(X_train_scaled, y_train)
mlp_model.fit(X_train_scaled, y_train)
y_pred_val_mlp = mlp_model.predict(X_val_scaled)

# print("Best MLP Parameters:", grid_mlp.best_params_)
# print("Best MLP CV Accuracy:", grid_mlp.best_score_)

from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier(
    estimators=[('rf', rf_model), ('svc', svc_model), ('mlp', mlp_model)],
    voting='soft'
)
ensemble.fit(X_train_scaled, y_train)
cv_scores_ensemble = cross_val_score(ensemble, X_train_scaled, y_train, cv=10)
print(f"Ensemble CV Accuracy: {cv_scores_ensemble.mean():.4f}")

# predictions = {
#     "SVC": y_pred_val_best_svc,
#     "Random Forest": y_pred_val_rf,
#     "MLP": y_pred_val_mlp
# }
# 
# accuracies = functions.evaluate_models(predictions, y_val)
# print("Accuracies:", accuracies)
# 
# best = functions.find_best_model(accuracies)
# print("Best model:", best)

# rf_scores = cross_val_score(rf_model, X, y, cv=10)
# svc_scores = cross_val_score(svc_model, X, y, cv=10)
# mlp_scores = cross_val_score(mlp_model, X, y, cv=10)
# print(f"Random Forest: {rf_scores.mean()}")
# print(f"SVC: {svc_scores.mean()}")
# print(f"MLP: {mlp_scores.mean()}")

print("----- Test Dataset Predictions -----")

test_data = np.load('data/wafermap_test.npy', allow_pickle = True)
df_test = pd.DataFrame(test_data)
df_test = process_wafer_map(df_test)
df_test = add_salient_region(df_test)
df_test = create_feature_columns(df_test)
test_X_scaled = scaler.transform(df_test[feature_columns])

#rf_predictions = rf_model.predict(df_test_final_features)
ensemble_predictions = ensemble.predict(test_X_scaled)

# rf_predictions_labels = [int2string[pred] for pred in rf_predictions]
ensemble_predictions_labels = [int2string[pred] for pred in ensemble_predictions]

# rf_series = pd.Series(rf_predictions_labels, name='failureType')
ensemble_series = pd.Series(ensemble_predictions_labels, name='failureType')

# rf_series.to_csv('rf_scores.csv', index=False)   # header included automatically
ensemble_series.to_csv('scores.csv', index=False)
