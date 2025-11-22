import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load your trained model
model = tf.keras.models.load_model("brain_tumor_model.h5")

# Class names (update if needed)
class_names = ["glioma", "meningioma", "no_tumor", "pituitary"]

# Your test image
image_path = r"C:\Users\akhil\OneDrive\Documents\brain_tumer_project\brain_tumor_dataset_structure\Testing\glioma\Te-gl_0012.jpg"

# Preprocess the image
img = image.load_img(image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Prediction
predictions = model.predict(img_array)
predicted_class = class_names[np.argmax(predictions)]
confidence = np.max(predictions) * 100

print("Predicted Class:", predicted_class)
print("Confidence:", confidence)
