# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout , BatchNormalization , AveragePooling2D ,Dropout,GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16 ,ResNet50 , DenseNet121, MobileNetV2
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import precision_recall_curve, classification_report,f1_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics import accuracy_score 
from tensorflow.keras import models,layers
from PIL import Image
import os

image_folder = "/kaggle/input/plant-pathology-2021-fgvc8/train_images"
sizes = set()

for img_name in os.listdir(image_folder):
    img_path = os.path.join(image_folder, img_name)
    try:
        img = Image.open(img_path)
        sizes.add(img.size)  # (width, height)
    except:
        pass

print("Unique image sizes found:", sizes)
df = pd.read_csv("/kaggle/input/plant-pathology-2021-fgvc8/train.csv")
df.head()
X_train = []
y_train = []

img_dir = "/kaggle/input/plant-pathology-2021-fgvc8/train_images"

for image, label in zip(df["image"], df["labels"]):

    path_image = os.path.join(img_dir, image)

    if not os.path.exists(path_image):
        continue  # skip broken/missing files

    im = tf.keras.preprocessing.image.load_img(path_image, target_size=(224, 224))
    im = tf.keras.preprocessing.image.img_to_array(im)

    X_train.append(im)
    y_train.append(label)

    if len(X_train) == 7000:
        break
x_train, x_valid, y_train, y_valid = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, shuffle=True
)
x_train = np.array(x_train, dtype="float32")
x_valid = np.array(x_valid, dtype="float32")
# y_train = np.array(y_train)
# y_valid = np.array(y_valid)
x_train = x_train/255.0
x_valid = x_valid/255.0
IMG_WIDTH=224
IMG_HEIGHT=224
from sklearn.preprocessing import MultiLabelBinarizer

# Ensure they are Series
y_train = pd.Series(y_train)
y_valid = pd.Series(y_valid)

# Convert strings → lists
y_train = y_train.apply(lambda x: x.split())
y_valid = y_valid.apply(lambda x: x.split())

# Fit MLb on training ONLY
mlb = MultiLabelBinarizer()
y_train_enc = mlb.fit_transform(y_train)
y_valid_enc = mlb.transform(y_valid)

print(mlb.classes_)
# DenseNet
base_model_dense = DenseNet121(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
model1 = models.Sequential([
    base_model_dense,
    layers.Flatten(),
    layers.Dense(1024, activation='relu'),
    layers.Dense(128,activation='relu'),
    layers.Dense(6, activation='sigmoid')
])
base_model_dense.trainable = True
model1.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary
model1.summary()
history = model1.fit(x_train, y_train_enc,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_valid, y_valid_enc))
pred_dense=model1.predict(x_valid)
# y_pred_bin = (pred_dense > 0.5).astype(int)



best_thresh1 = []
for i in range(pred_dense.shape[1]):
    p, r, t = precision_recall_curve(y_valid_enc[:, i], pred_dense[:, i])
    f1 = 2 * p * r / (p + r + 1e-9)
    bi = np.argmax(f1)
    best_thresh1.append(t[bi] if bi < len(t) else 0.5)
best_thresh1 = np.array(best_thresh1)

# 3) Convert to binary predictions
y_pred_bin = (pred_dense >= best_thresh1).astype(int)

# 4) Report
print(classification_report(y_valid_enc, y_pred_bin, target_names=mlb.classes_, zero_division=0))

mean_f1 = f1_score(y_valid_enc, y_pred_bin, average='macro')
print("Mean F1 Score (Macro):", mean_f1)
# Base MobileNetV2
base_model_mobilenet = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
)

# Build model
model_mobilenet = models.Sequential([
    base_model_mobilenet,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1024, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(6, activation='sigmoid')   # <-- 6 classes
])

# Freeze backbone
base_model_mobilenet.trainable = True
# Print the model summary
model_mobilenet.summary()
# Compile
model_mobilenet.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
history = model_mobilenet.fit(x_train, y_train_enc,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_valid, y_valid_enc))
pred_mobile=model_mobilenet.predict(x_valid)

best_thresh = []
for i in range(pred_mobile.shape[1]):
    p, r, t = precision_recall_curve(y_valid_enc[:, i], pred_mobile[:, i])
    f1 = 2 * p * r / (p + r + 1e-9)
    bi = np.argmax(f1)
    best_thresh.append(t[bi] if bi < len(t) else 0.5)
best_thresh = np.array(best_thresh)

# 3) Convert to binary predictions
y_pred2_bin = (pred_mobile >= best_thresh).astype(int)

# 4) Report
print(classification_report(y_valid_enc, y_pred2_bin, target_names=mlb.classes_, zero_division=0))

mean_f1 = f1_score(y_valid_enc, y_pred2_bin, average='macro')
print("Mean F1 Score (Macro):", mean_f1)
from tensorflow.keras.applications import EfficientNetV2B0

# Load EfficientNetB0 base model
base_model_eff = EfficientNetV2B0(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)


# Build model
model_eff = models.Sequential([
    
    base_model_eff,
    layers.GlobalAveragePooling2D(),         
    layers.Dense(1024, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(6, activation='sigmoid')   
])

# Freeze base model weights
base_model_eff.trainable = True

# Compile model
model_eff.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model_eff.summary()
history = model_eff.fit(x_train, y_train_enc,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_valid, y_valid_enc))
pred_eff=model_eff.predict(x_valid)

best_thresh = []
for i in range(pred_eff.shape[1]):
    p, r, t = precision_recall_curve(y_valid_enc[:, i], pred_eff[:, i])
    f1 = 2 * p * r / (p + r + 1e-9)
    bi = np.argmax(f1)
    best_thresh.append(t[bi] if bi < len(t) else 0.5)
best_thresh = np.array(best_thresh)

# 3) Convert to binary predictions
y_pred3_bin = (pred_eff >= best_thresh).astype(int)

# 4) Report
print(classification_report(y_valid_enc, y_pred3_bin, target_names=mlb.classes_, zero_division=0))

mean_f1 = f1_score(y_valid_enc, y_pred3_bin, average='macro')
print("Mean F1 Score (Macro):", mean_f1)
from tqdm import tqdm
from tensorflow.keras.preprocessing import image as kimage   # <--- renamed here

# ---------------------------
# Load sample submission
# ---------------------------
submission = pd.read_csv("/kaggle/input/plant-pathology-2021-fgvc8/sample_submission.csv")

# ---------------------------
# Function: load & preprocess test image
# ---------------------------
def load_test_image(path):
    img_obj = kimage.load_img(path, target_size=(IMG_WIDTH, IMG_HEIGHT,3))  # no conflict
    img_arr = kimage.img_to_array(img_obj)
    img_arr = img_arr / 255.0
    return img_arr


test_dir = "/kaggle/input/plant-pathology-2021-fgvc8/test_images"
test_images = submission["image"].values

# Create paths
test_paths = [os.path.join(test_dir, fname) for fname in test_images]

# Dataset with preprocessing
def preprocess(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])
    img = img / 255.0
    return img

test_dataset = tf.data.Dataset.from_tensor_slices(test_paths)
test_dataset = test_dataset.map(preprocess).batch(32)


# ---------------------------
# Predict probabilities
# ---------------------------
pred_probs = model1.predict(test_dataset)

# ---------------------------
# Convert probabilities → binary predictions
# ---------------------------
y_pred_bin = (pred_probs >= best_thresh1).astype(int)

# ---------------------------
# Convert binary vectors → text labels
# ---------------------------
pred_labels = mlb.inverse_transform(y_pred_bin)

# Convert list of labels → space-separated strings
pred_strings = [" ".join(lbl) if len(lbl) > 0 else "healthy" 
                for lbl in pred_labels]

# ---------------------------
# Build submission file
# ---------------------------
submission["labels"] = pred_strings

submission.to_csv("submission.csv", index=False)

print("Submission file saved as submission.csv!")
