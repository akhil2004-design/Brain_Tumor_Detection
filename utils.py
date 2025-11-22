import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


def create_generators(data_dir, img_size=(224,224), batch_size=32, val_split=0.15, test_split=0.10, seed=42):
# Assumes data_dir has class subfolders
datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.1,
height_shift_range=0.1, zoom_range=0.15, horizontal_flip=True)


# We'll manually create train/val/test splits by file lists
classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir,d))]
filepaths = []
labels = []
for cls in classes:
for fname in os.listdir(os.path.join(data_dir, cls)):
if fname.lower().endswith(('.png','.jpg','.jpeg')):
filepaths.append(os.path.join(data_dir, cls, fname))
labels.append(cls)


X_train, X_temp, y_train, y_temp = train_test_split(filepaths, labels, test_size=(val_split+test_split), stratify=labels, random_state=seed)
rel_val = val_split/(val_split+test_split)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(1-rel_val), stratify=y_temp, random_state=seed)


def generator_from_lists(filepaths, labels, batch_size, shuffle=True, augment=False):
gen = ImageDataGenerator(rescale=1./255)
if augment:
gen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.15, horizontal_flip=True)
while True:
idxs = np.arange(len(filepaths))
if shuffle:
np.random.shuffle(idxs)
for i in range(0, len(filepaths), batch_size):
batch_idxs = idxs[i:i+batch_size]
batch_x = []
batch_y = []
for j in batch_idxs:
img = image_load_and_preprocess(filepaths[j], target_size=img_size)
batch_x.append(img)
batch_y.append(labels[j])
yield np.array(batch_x), np.array(batch_y)


return (X_train, y_train), (X_val, y_val), (X_test, y_test)


from tensorflow.keras.preprocessing import image


def image_load_and_preprocess(path, target_size=(224,224)):
img = image.load_img(path, target_size=target_size)
arr = image.img_to_array(img)/255.0
return arr