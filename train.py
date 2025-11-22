import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# ============================
# 📌 1. PATHS
# ============================
train_path = r"C:\Users\akhil\OneDrive\Documents\brain_tumer_project\brain_tumor_dataset_structure\Training"
test_path  = r"C:\Users\akhil\OneDrive\Documents\brain_tumer_project\brain_tumor_dataset_structure\Testing"

img_size = (224, 224)
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_path,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_path,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_path,
    shuffle=False,
    image_size=img_size,
    batch_size=batch_size
)

class_names = train_ds.class_names
print("Classes found:", class_names)

# Prefetching for performance
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# ============================
# 📌 3. BUILD MODEL (ResNet50)
# ============================
base_model = tf.keras.applications.ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False  # Freeze ResNet layers

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(4, activation="softmax")
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================
# 📌 4. TRAIN MODEL
# ============================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# ============================
# 📌 5. SAVE MODEL
# ============================
model.save("brain_tumor_model.h5")
print("\nModel saved as brain_tumor_model.h5")

# ============================
# 📌 6. EVALUATE ON TEST DATA
# ============================
test_loss, test_acc = model.evaluate(test_ds)
print("\nTest Accuracy:", test_acc)

# ============================
# 📌 7. PLOT ACCURACY & LOSS
# ============================
plt.figure(figsize=(10,5))

# Accuracy plot
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label="train_acc")
plt.plot(history.history['val_accuracy'], label="val_acc")
plt.title("Accuracy")
plt.legend()

# Loss plot
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label="train_loss")
plt.plot(history.history['val_loss'], label="val_loss")
plt.title("Loss")
plt.legend()

plt.savefig("training_plots.png")
plt.show()

print("\nTraining plots saved as training_plots.png")
