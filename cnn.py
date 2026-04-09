import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from functions import prepare_data

df = pd.DataFrame(np.load('data/wafermap_train.npy', allow_pickle=True))

# 1. Run your existing preprocessing
df_prepped = prepare_data(df)

# 2. Extract and Stack the Reshaped Maps
# Since reshapedMap is a column of 64x64 arrays, we stack them into a 3D block
X_images = np.stack(df_prepped['reshapedMap'].values)

# 3. Add the channel dimension for the CNN (Samples, 64, 64, 1)
X_images = X_images.reshape(-1, 64, 64, 1)

# 4. Normalize the values
# WM-811K usually has 0, 1, 2. Dividing by 2 scales them to [0, 1]
X_images = X_images.astype('float32') / 2.0

# 5. Extract the target labels
y_labels = df_prepped['failureTypeNumber'].values

# 6. Train/Val Split
from sklearn.model_selection import train_test_split
X_train_img, X_val_img, y_train_img, y_val_img = train_test_split(
    X_images, y_labels, test_size=0.2, random_state=42
)

# 4. Normalize pixel values
# Typically WM-811K uses 0, 1, 2. We scale to [0, 1] for better convergence.
X_train_img = X_train_img.astype('float32') / 2.0
X_val_img = X_val_img.astype('float32') / 2.0


# 5. Build the CNN model
def build_wafer_cnn(input_shape=(64, 64, 1), num_classes=5):
    model = models.Sequential([
        # Block 1: Local defects
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Block 2: Shape Recognition (Scratch vs. Edge)
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Block 3: Global topology (Donut vs. Near-full)
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.GlobalAveragePooling2D(), # Reduces overfitting compared to Flatten
        
        # Output
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3), # Helps prevent the model from memorizing noise
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

cnn_model = build_wafer_cnn()
cnn_model.summary()

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

from sklearn.utils import class_weight

weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train_img),
    y=y_train_img
)
dict_weights = dict(enumerate(weights))

# Then in model.fit:
history = cnn_model.fit(
    X_train_img, y_train_img,
    epochs=100,
    validation_data=(X_val_img, y_val_img),
    class_weight=dict_weights, # Add this!
    callbacks=[early_stop]
)

from sklearn.metrics import classification_report

y_pred = np.argmax(cnn_model.predict(X_val_img), axis=1)
print(classification_report(y_val_img, y_pred, target_names=["Donut", "Center", "Edge-Loc", "Scratch", "Near-full"]))