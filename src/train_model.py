import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import class_weight

# Load datasets
X_train = np.load("processed/X_train.npy")
y_train = np.load("processed/y_train.npy")

X_val = np.load("processed/X_val.npy")
y_val = np.load("processed/y_val.npy")

# 🔥 Class weights (handle imbalance)
weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weights = {0: weights[0], 1: weights[1]}

print("Class weights:", class_weights)

print("Train shape:", X_train.shape)
print("Val shape:", X_val.shape)

# Add channel dimension
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]

# 🔥 Model (balanced for your data)
model = models.Sequential([
    layers.Input(shape=(64,128,128,1)),

    layers.Conv3D(16, 3, activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling3D(2),

    layers.Conv3D(32, 3, activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling3D(2),

    layers.Conv3D(64, 3, activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling3D(2),

    layers.GlobalAveragePooling3D(),

    layers.Dense(64, activation='relu'),
    layers.Dropout(0.6),

    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Early stopping
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=50,
    restore_best_weights=True
)

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=2,
    callbacks=[early_stop],
    class_weight=class_weights
)

# Save model
model.save("parkinson_3d_model.keras")

print("\nModel training complete!")

# 🔥 Evaluation
y_pred = (model.predict(X_val) > 0.5).astype(int)

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_pred))

print("\nClassification Report:")
print(classification_report(y_val, y_pred))