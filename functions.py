import os
import typing
from typing import Tuple
import shutil

import numpy as np
import pandas as pd

# OpenCV and Scikit-Image for image manipulation
# This code prefers skimage for resize but you can also import and use cv2
from skimage.transform import resize
from skimage.measure import label, regionprops, perimeter
from skimage.morphology import convex_hull_image, binary_dilation, square

# Seaborn and Matplotlib for plotting
import seaborn as sns
import matplotlib.pyplot as plt

# Scikit-Learn and helper functions
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix, homogeneity_score, completeness_score, ConfusionMatrixDisplay

FAIL = 2 # failing die
PASS = 1 # passing die
NO_DIE = 0 # no die
RANDOM_SEED = 10

# ------------------------------------------------------------------------------------------------
#
#                                      Data Preprocessing Functions
#
# ------------------------------------------------------------------------------------------------
string2int = {
    "Center": 0,
    "Edge-Loc": 1,
    "Scratch": 2,
    "Donut": 3,
    "Near-full": 4
}

int2string = {value: key for key, value in string2int.items()}

# convert_failure_type(): Uses the string2int dictionary to change each failure type to an integer
def convert_failure_type(failure_type: str) -> int:
    return string2int[str(failure_type)]

# resise_wafer_map(): Changes the size of all wafermaps to 64 x 64
def resize_wafer_map(wafer_map: np.ndarray, output_shape: tuple=(64, 64)) -> np.ndarray:
    return resize(wafer_map, output_shape, order=0, anti_aliasing=False, preserve_range=True)


# prepare_data(): creates dataframe columns to store the return values of the two tasks below using convert_failure_type and resize_wafer_map:
# (1) reshapes the wafer maps as a numpy array of shape (64, 64)
# (2) converts the failureType into numeric values
def prepare_data(df: pd.DataFrame, has_labels: bool=True) -> Tuple[np.ndarray, list]:
    
    # Check if DataFrame has the necessary columns
    if 'waferMap' not in df.columns:
        raise ValueError("DataFrame must contain 'waferMap' column")
    
    # Create a new DataFrame to store the processed data
    processed_data = df.copy()

    # Apply the reshape function to the wafer_map column
    processed_data['reshapedMap'] = processed_data['waferMap'].apply(lambda x: resize_wafer_map(np.array(x)))

    # Apply the convert function to the failure_type column
    processed_data['failureTypeNumber'] = processed_data['failureType'].apply(convert_failure_type)

    return processed_data

# process_wafer_map(): processes the wafer maps in the dataframe by resizing them to 64 x 64
def process_wafer_map(df):
    df['reshapedMap'] = df['waferMap'].apply(resize_wafer_map)
    return df

# add_salient_region(): adds a column for the salient region to the dataframe
def add_salient_region(df):
    df['salientRegion'] = df.apply(get_salient_region, axis=1)
    return df

# output_all_wafer_maps(): outputs all wafer maps to corresponding failure type directories
def output_all_wafer_maps(df: pd.DataFrame) -> None:
    # List of failure types for directory creation
    failure_types = ["Center", "Edge-Loc", "Scratch", "Donut", "Near-full"]

    # Iterate through each failure type to create directories
    for failure_type in failure_types:
        # Create a directory for the failure type if it doesn't exist
        os.makedirs(failure_type, exist_ok=True)
    
    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Extract the wafer map and failure type from the current row
        wafer_map = row['waferMap']  # this is expected to be a numpy array
        failure_type = row['failureType']
        
        # Construct a filename based on the index
        # You can customize this name according to what you want.
        filename = f"wafer_map_{index}.npy"  # This will save the profile with index
        
        # Define the path to save the file
        file_path = os.path.join(failure_type, filename)
        
        # Save the wafer map as a .npy file
        np.save(file_path, wafer_map)
    return


# ------------------------------------------------------------------------------------------------
#
#                                      Feature Column Functions
#
# ------------------------------------------------------------------------------------------------


# create_feature_columns(): creates 10 feature columns in given dataframe
def create_feature_columns(df: pd.DataFrame) -> pd.DataFrame:

    df['areaRatio'] = df.apply(get_area_ratio, axis=1)
    df['perimeterRatio'] = df.apply(get_perimeter_ratio, axis=1)
    df['maxDistFromCenter'] = df.apply(get_max_dist_from_center, axis=1)
    df['minDistFromCenter'] = df.apply(get_min_dist_from_center, axis=1)
    df['majorAxisRatio'] = df.apply(get_major_axis_ratio, axis=1)
    df['minorAxisRatio'] = df.apply(get_minor_axis_ratio, axis=1)
    df['solidity'] = df.apply(get_solidity, axis=1)
    df['eccentricity'] = df.apply(get_eccentricity, axis=1)
    df['yieldLoss'] = df.apply(get_yield_loss, axis=1)
    df['edgeYieldLoss'] = df.apply(get_edge_yield_loss, axis=1)

    return df


# get_salient_region(): detects connected failing dies using skimage and selects "one with the largest area(=salient region)" for each wafer map
def get_salient_region(row: pd.Series) -> np.ndarray:
    # Extract the wafer map from the input row
    wafer_map = row['reshapedMap']
    
    # Check that the wafer map is in the expected format
    if wafer_map.shape != (64, 64):
        raise ValueError("Wafer map must be a 64x64 numpy array.")
    
    # Label connected regions of '2's (failing dies) using connectivity=2
    labeled_map = label(wafer_map == 2, connectivity=2)
    
    # Extract properties of labeled regions
    regions = regionprops(labeled_map)
    
    # If there are no failing dies, return an empty array
    if not regions:
        return np.zeros_like(wafer_map)
    
    # Find the region with the largest area
    largest_region = max(regions, key=lambda r: r.area)
    
    # Create a blank array to hold the salient region
    salient_region = np.zeros_like(wafer_map)
    
    # Fill in the salient region with '2's
    for coords in largest_region.coords:
        salient_region[coords[0], coords[1]] = 2
    
    return salient_region


# get_area_ratio(): returns the ratio of the area of the salient region to the area of the wafer map
def get_area_ratio(row: pd.Series) -> float:
    # Get the salient region numpy array from the row
    salient_region = row['salientRegion']
    
    # Calculate the area of the salient region (non-zero elements)
    area_salient_region = np.count_nonzero(salient_region)  # Count of non-zero elements
    
    # Get the area of the wafer map from the dieSize column
    area_wafer_map = row['dieSize']
    
    # Calculate the area ratio
    area_ratio = area_salient_region / area_wafer_map if area_wafer_map > 0 else 0.0
    
    return area_ratio


# get_perimeter_ratio(): returns the ratio of the perimeter of the salient region to the radius of the wafer map
def get_perimeter_ratio(row: pd.Series) -> float:
    # Extract the salient region
    salient_region = row['salientRegion']
    
    # Calculate the perimeter of the salient region
    # The perimeter function requires a binary image, so we will use the salient region
    # If "2" indicates the salient region, we will convert it to a binary format suitable for perimeter calculation
    binary_region = (salient_region == 2).astype(int)
    # perimeter_value = perimeter(binary_region, connectivity=2) # for older version fo skimage
    perimeter_value = perimeter(binary_region)
    
    # Calculate the radius
    # The radius for a 64 x 64 array
    radius = np.sqrt(2) * 32  # or 32 * np.sqrt(2)
    
    # Calculate the ratio of the perimeter to the radius
    perimeter_ratio = perimeter_value / radius
    
    return perimeter_ratio


# get_max_dist_from_center(): returns the maximal distance between the salient region and the center of the wafer map
def get_max_dist_from_center(row: pd.Series) -> float:
    # Get the salient region array from the row
    salient_region = row['salientRegion']
    
    # Center of the wafer map
    center_x, center_y = 31.5, 31.5
    
    # Find indices where the salient region is marked
    salient_indices = np.argwhere(salient_region == 2)
    
    # If there are no salient regions, return a large number or a suitable value
    if len(salient_indices) == 0:
        return 0.0  # or some other sentinel value

    # Calculate distances from center to each salient region
    distances = np.sqrt((salient_indices[:, 0] - center_x) ** 2 + (salient_indices[:, 1] - center_y) ** 2)
    
    # Return the maximum distance
    return np.max(distances)


# get_min_dist_from_center(): returns the minimal distance between the salient region and the center of the wafer map
def get_min_dist_from_center(row: pd.Series) -> float:
    # Get the salient region array from the row
    salient_region = row['salientRegion']
    
    # Center of the wafer map
    center_x, center_y = 31.5, 31.5
    
    # Find indices where the salient region is marked
    salient_indices = np.argwhere(salient_region == 2)
    
    # If there are no salient regions, return a large number or a suitable value
    if len(salient_indices) == 0:
        return float('inf')  # or some other sentinel value

    # Calculate distances from center to each salient region
    distances = np.sqrt((salient_indices[:, 0] - center_x) ** 2 + (salient_indices[:, 1] - center_y) ** 2)
    
    # Return the minimum distance
    return np.min(distances)


# get_major_axis_ratio(): returns the ratio of the length of the major axis of the estimated ellipse surrounding the salient region to the radius of the wafer map
def get_major_axis_ratio(row: pd.Series) -> float:
    # Extract the salient region from the row
    salient_region = np.array(row['salientRegion'])

    # Check if there are any salient regions (2's) in the array
    if np.any(salient_region == 2):
        # Create a labeled image from the salient region; assume that 2 indicates the region
        labeled_region = label(salient_region == 2)  # This will create regions labeled as 1, 2, etc.
        
        # Get properties of the labeled regions
        regions = regionprops(labeled_region)

        if regions:
            # We can assume we're interested in the first region
            region = regions[0]
            # The major axis length of the fitted ellipse
            major_axis_length = region.major_axis_length
            
            # The radius of the wafer map (half the diagonal of 64x64)
            wafer_size = 64
            map_radius = np.sqrt(2 * (wafer_size / 2)**2)  # Diagonal / 2 = sqrt(2) * (side/2)
            
            # Calculate the ratio of major axis length to map radius
            major_axis_ratio = major_axis_length / map_radius
            
            return major_axis_ratio
    else:
        # If there is no salient region, return 0 or some indicator value
        return 0.0
    

# get_minor_axis_ratio(): returns the ratio of the length of the minor axis of the estimated ellipse surrounding the salient region to the radius of the wafer map
def get_minor_axis_ratio(row: pd.Series) -> float:
    # Extract the salient region array from the row
    salient_region_array = np.array(row['salientRegion'])

    # Convert the salient region (np array) to a binary image (0s and 1s) for region analysis
    binary_mask = (salient_region_array == 2).astype(int)

    # Label the connected components in the binary mask
    labeled_mask = label(binary_mask)

    # Extract properties of the labeled region
    props = regionprops(labeled_mask)

    # We assume there's only one salient region to analyze, if there are multiple you might need to loop or select one
    if not props:
        return 0.0  # if no regions found, return 0
    
    # Get the major axis and minor axis lengths of the first region
    minor_axis_length = props[0].minor_axis_length
    
    # Calculate the radius of the wafer map
    # The radius r = diagonal_length / 2, where diagonal_length = sqrt(2) * side_length
    # For a 64 x 64 image (side_length = 64), diagonal_length = sqrt(2) * 64
    diagonal_length = np.sqrt(2) * 64
    radius = diagonal_length / 2
    
    # Calculate the ratio of the minor axis to the radius
    minor_axis_ratio = minor_axis_length / radius
    
    return minor_axis_ratio


# get_solidity(): returns the solidity, indicating the proportion of defective dies in the estimated convex hull of the salient region
def get_solidity(row: pd.Series) -> float:
    # Extract the salient region as a binary image
    salient_region = row['salientRegion']
    
    # Calculate the convex hull of the salient region
    convex_hull = convex_hull_image(salient_region)
    
    # Calculate the area of the salient region (sum of true values)
    area_salient = np.sum(salient_region)
    
    # Calculate the area of the convex hull (sum of true values in convex_hull)
    area_convex_hull = np.sum(convex_hull)
    
    # Calculate the solidity
    if area_convex_hull == 0:
        return 0.0  # To avoid division by zero
    
    solidity = area_salient / area_convex_hull
    return solidity


# get_eccentricity(): returns the eccentricity of the salient region
def get_eccentricity(row: pd.Series) -> float:
    # Assuming 'salientRegion' is a binary image (2D array) in the row
    salient_region = row['salientRegion']  # Get the salient region

    # Label the image to identify distinct regions
    labeled_region = label(salient_region)

    # Calculate properties for the labeled regions
    regions = regionprops(labeled_region)

    # If there are regions found, return the eccentricity of the first region (or handle accordingly)
    if regions:
        # Return the eccentricity of the first region
        return regions[0].eccentricity

    return 0.0


# get_yield_loss(): returns the ratio of the failed dies on the wafer map to the total number of dies on the wafer map
def get_yield_loss(row: pd.Series) -> float:
    # Access the waferMap and dieSize from the row
    wafer_map = row['reshapedMap']
    total_dies = row['dieSize']
    
    # Count the number of failed dies (2s)
    failed_dies = np.sum(wafer_map == 2)
    
    # Calculate the yield loss (ratio of failed to total dies)
    # We use max to avoid division by zero in case total_dies is 0
    yield_loss = failed_dies / total_dies if total_dies > 0 else 0.0
    
    return yield_loss


# ring_label_from_outside(): returns the ratio of the failed dies on the outermost two rings of the wafer map to the total number of dies on the outermost two rings of the wafer map
# Used in get_edge_yield_loss() function
def ring_label_from_outside(wafer_map: np.ndarray, ring_thickness: int = 2) -> np.ndarray:
    nrows, ncols = wafer_map.shape
    cy, cx = nrows / 2, ncols / 2

    Y, X = np.ogrid[:nrows, :ncols]
    dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)

    wafer_radius = nrows / 2  # or adjusted if needed
    ring_mask = (dist_from_center >= wafer_radius - ring_thickness) & (dist_from_center <= wafer_radius)

    # Convert boolean mask to int
    ring_labeled_map = np.zeros_like(wafer_map)
    ring_labeled_map[ring_mask] = wafer_map[ring_mask]

    return ring_labeled_map

# get_edge_yield_loss(): returns a numpy array highlighting the outermost two rings of the wafer map with nonzero value
def get_edge_yield_loss(row: pd.Series) -> float:
    wafer_map = row['reshapedMap']
    rings = ring_label_from_outside(wafer_map)

    # Count the number of failing dies (value 2) and total dies (value 1 and 2)
    failed_dies = np.sum(rings == 2)
    total_dies = np.sum((rings == 1) | (rings == 2))

    # Calculate and return the ratio of failed dies to total dies
    if total_dies == 0:
        return 0.0  # Avoid division by zero, return 0 if there are no dies
    edge_yield_loss = failed_dies / total_dies
    
    return edge_yield_loss


# ------------------------------------------------------------------------------------------------
#
#                                      Model Validation Functions
#
# ------------------------------------------------------------------------------------------------


# compute_accuracy(): calculates the prediction accuracy against expected
def compute_accuracy(prediction: np.ndarray, expected: np.ndarray) -> float:
    
    pred = np.asarray(prediction)
    exp = np.asarray(expected)

    if pred.shape != exp.shape:
        raise ValueError("The shape of predicted labels and true labels must match.")

    total = pred.size
    correct = np.sum(pred == exp)

    return float(correct) / float(total)

def calculate_accuracy(pred_labels: np.ndarray, true_labels: np.ndarray) -> float:
    # Backward-compatible wrapper delegating to compute_accuracy
    return compute_accuracy(pred_labels, true_labels)


# calculate_per_class_accuracy(): calculates the prediction accuracy for each failure type separately.
def calculate_per_class_accuracy(pred_labels: np.ndarray, true_labels: np.ndarray):
    # Get unique classes (failure types)
    unique_classes = np.unique(true_labels)
    
    # Initialize a dictionary to hold accuracies
    accuracies = {}

    # Calculate accuracy for each class
    for cls in unique_classes:
        # Get indices where true_labels are the current class
        class_indices = np.where(true_labels == cls)[0]

        # If there are no instances of this class in true_labels, accuracy is undefined
        if len(class_indices) == 0:
            accuracies[cls] = None
            continue

        # Get predictions for the current class
        class_pred_labels = pred_labels[class_indices]

        # Calculate the number of correct predictions for this class
        correct_predictions = np.sum(class_pred_labels == cls)

        # Calculate accuracy for this class
        accuracies[cls] = correct_predictions / len(class_indices)

    return accuracies


# print_accuracies(): uses the other two accuracy functions to print overall and class specific accuracies with the class label next to each accuracy
def print_accuracies(pred_labels: np.ndarray, true_labels: np.ndarray):
    # Calculate overall accuracy
    overall_accuracy = calculate_accuracy(pred_labels, true_labels)

    # Calculate per-class accuracies
    per_class_accuracies = calculate_per_class_accuracy(pred_labels, true_labels)

    # Print overall accuracy
    print(f"Overall Accuracy: {overall_accuracy:.4f}")

    # Print class-specific accuracies with labels
    print("Class-Specific Accuracies:")
    for cls, accuracy in per_class_accuracies.items():
        class_label = int2string.get(cls, f"Class {cls}")
        if accuracy is not None:
            print(f"{class_label}: {accuracy:.4f}")
        else:
            print(f"{class_label}: No instances in true labels")


# confunsion_matrix(predicted, expected): computes the confusino matrix for a models predicted values vs the expected values
def print_confusion_matrix(y_pred, y_expected):
    # Compute confusion matrix
    cm = sklearn_confusion_matrix(y_expected, y_pred)

    # Get labels and their string representations
    labels = np.unique(np.concatenate([y_expected, y_pred]))
    labels_strings = [int2string[l] for l in labels]

    # Print the confusion matrix
    print("Confusion Matrix:")
    print(cm)

    # Display as pandas DataFrame for better readability
    if hasattr(y_expected, '__len__') and len(labels) <= 10:  # Assuming small number of classes
        df_cm = pd.DataFrame(cm, index=labels_strings, columns=labels_strings)
        print("\nConfusion Matrix as DataFrame:")
        print(df_cm)

    # Plot the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_strings)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()



def evaluate_models(model_preds: dict, y_true: np.ndarray) -> dict:
    """
    Evaluate multiple models' predictions against the same ground truth.

    Args:
      model_preds: dict mapping model name -> predictions (array-like)
      y_true: ground truth labels (array-like)

    Returns:
      dict mapping model name -> accuracy (float)
    """
    if not isinstance(model_preds, dict):
        raise TypeError("model_preds must be a dict mapping model name to predictions")

    y_true_arr = np.asarray(y_true)
    results = {}

    for name, preds in model_preds.items():
        preds_arr = np.asarray(preds)
        # Align shapes if possible
        if preds_arr.shape != y_true_arr.shape:
            if preds_arr.size == y_true_arr.size:
                preds_arr = preds_arr.reshape(y_true_arr.shape)
            else:
                raise ValueError(f"Predictions for model '{name}' have incompatible shape {preds_arr.shape} vs {y_true_arr.shape}")
        results[name] = compute_accuracy(preds_arr, y_true_arr)

    return results

def find_best_model(accuracies: dict, return_all_max: bool = False):
    """
    Given a dict of {model_name: accuracy}, return the best model.

    Args:
      accuracies: dict of model name -> accuracy
      return_all_max: if True, return a list of all models tying for max accuracy

    Returns:
      - If return_all_max is False: the single best model name (lexicographically smallest in ties)
      - If return_all_max is True: a list of best model names
    """
    if not accuracies:
        raise ValueError("No accuracies provided")

    max_acc = max(accuracies.values())
    best_models = [name for name, acc in accuracies.items() if acc == max_acc]

    if return_all_max:
        return best_models
    else:
        # Tie-break by lexicographic order for determinism
        return min(best_models)

