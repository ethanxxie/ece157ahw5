# Wafer Defect Classification Model

This project implements a machine learning model for classifying wafer defects using the WM-811K dataset. The goal is to predict failure types (Center, Edge-Loc, Scratch, Donut, Near-full) from wafer maps.

## Data Preparation

The data preparation process involves the following steps:

1. **Data Loading**: Load the training data from `wafermap_train.npy` using `np.load()` and create a pandas DataFrame with the loaded array.

2. **Wafer Map Processing**: Resize all wafer maps to a uniform 64x64 shape using `skimage.transform.resize()` with order=0 to preserve discrete values. Convert failure type strings to integer labels using a predefined dictionary (e.g., "Center" → 0, "Edge-Loc" → 1, etc.).

3. **Salient Region Detection**: For each wafer map, detect connected regions of failing dies (value 2) using `skimage.measure.label()`. Select the salient region as the connected component with the largest area.

4. **Feature Extraction**: Compute 10 engineered features based on the salient regions and wafer properties.

5. **Feature Scaling**: Apply `sklearn.preprocessing.StandardScaler` to standardize the feature columns.

6. **Data Splitting**: Split the processed dataset into training (80%) and validation (20%) sets using `train_test_split()`.

This results in a cleaned dataset ready for model training.

## Feature Engineering

The following 10 features are extracted from each wafer map, all derived from the salient region (largest connected area of failing dies):

- **areaRatio**: Ratio of the salient region's area to the total die count.
- **perimeterRatio**: Ratio of the salient region's perimeter to the wafer map's radius.
- **maxDistFromCenter**: Maximum distance from any failing die in the salient region to the wafer center.
- **minDistFromCenter**: Minimum distance from any failing die in the salient region to the wafer center.
- **majorAxisRatio**: Ratio of the major axis length of the fitted ellipse around the salient region to the wafer radius.
- **minorAxisRatio**: Ratio of the minor axis length of the fitted ellipse around the salient region to the wafer radius.
- **solidity**: Proportion of failing dies within the convex hull of the salient region.
- **eccentricity**: Measure of how elliptical the salient region is (0 = circle, 1 = line).
- **yieldLoss**: Overall ratio of failing dies to total dies on the wafer.
- **edgeYieldLoss**: Ratio of failing dies on the outermost two rings (within 2 pixels of the edge) to total dies in those rings.

These features capture geometric, positional, and density characteristics of defect patterns using techniques from `scikit-image`.

## Model and Hyperparameter Choice

Three models were evaluated: Random Forest, Support Vector Classifier (SVC), and Multi-Layer Perceptron (MLP). A soft-voting ensemble combining all three was also tested.

### Ensemble (Chosen Model for Submission)
- **Rationale**: Combines strengths of individual models to improve robustness and accuracy, using soft voting based on predicted probabilities.
- **Components**:
  - Random Forest with n_estimators=200, max_depth=10, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', random_state=67, class_weight='balanced'
  - SVC with probability=True, C=1, degree=3, gamma='auto', kernel='poly', random_state=67
  - MLP with max_iter=10000, random_state=67, hidden_layer_sizes=(128, 128, 64), activation='relu', solver='adam', alpha=1e-5, learning_rate='constant', learning_rate_init=0.001

### Individual Model Details

#### Random Forest
- Evaluated as the best individual model with 97.27% validation accuracy.
- Hyperparameters: as above.

#### SVC
- Kernel: Poly
- Hyperparameters: C=1, degree=3, gamma='auto', random_state=67

#### MLP
- Architecture: Hidden layers (128, 128, 64), activation='relu', solver='adam'
- Hyperparameters: alpha=1e-5, learning_rate_init=0.001, max_iter=10000

### Selection Process
Models were trained on scaled features and evaluated using validation set accuracy and cross-validation. The ensemble was selected for final test predictions to leverage combined performance. GridSearchCV was considered but manual tuning was used.

## Training and Validation Accuracy

Model performance was evaluated using overall and per-class accuracy on the validation set, plus 10-fold cross-validation.

### Validation Accuracies (on 20% held-out set):
- SVC: 0.9527
- Random Forest: 0.9727 (Best)
- MLP: 0.9709

### Cross-Validation Accuracies (10-fold):
- Random Forest: 0.9487
- SVC: 0.9519
- MLP: 0.9377
- Ensemble: 0.9649

Random Forest achieved the highest individual validation accuracy. The ensemble model was selected for final test predictions to leverage combined performance. Per-class accuracies were also computed (available in code output) to ensure balanced performance across failure types.

The models demonstrate strong classification performance, with the ensemble providing additional robustness through averaging predictions across the individual models.
