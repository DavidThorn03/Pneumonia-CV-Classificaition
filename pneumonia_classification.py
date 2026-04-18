from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Rescaling, BatchNormalization, RandomFlip, RandomRotation, RandomZoom, GlobalAvgPool2D
from tensorflow.keras.applications import EfficientNetB0
from keras.optimizers import RMSprop,Adam
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight


import warnings
warnings.filterwarnings("ignore")

tf.get_logger().setLevel('ERROR')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

AUTOTUNE = tf.data.AUTOTUNE

batch_size = 16
num_classes = 3
epochs = 30
img_width = 224
img_height = 224
img_channels = 3
fit = True #make fit false if you do not want to train the network again
stage = 'Transfer/'
os.makedirs(stage, exist_ok=True)
os.makedirs(stage + 'plots', exist_ok=True)
os.makedirs(stage + 'reports', exist_ok=True)

train_dir = 'chest_xray/chest_xray/train/'
test_dir = 'chest_xray/chest_xray/test/'
    
#create training,validation and test datatsets 
train_ds,val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    seed=123,
    validation_split=0.2,
    subset='both',
    image_size=(img_height, img_width),
    batch_size=batch_size,
    labels='inferred',
    shuffle=True,
    verbose=False,
    color_mode='rgb')

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    seed=None,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    labels='inferred',
    shuffle=False,
    verbose=False,
    color_mode='rgb')

class_names = train_ds.class_names
print('Class Names: ',class_names)
num_classes = len(class_names)

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    RandomFlip('horizontal'),
    RandomRotation(0.01),
    RandomZoom(height_factor=(-0.05, 0.05), width_factor=(-0.05, 0.05))
])

base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(img_height, img_width, 3),
    pooling="avg"
)

base_model.trainable = False

y_train = np.concatenate([labels.numpy() for _, labels in train_ds])

class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weight = dict(enumerate(class_weights_array))

#create model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(img_height, img_width, 3)),
    data_augmentation,
    base_model,
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(num_classes, activation="softmax")
])

model.compile(loss='sparse_categorical_crossentropy',
                optimizer=Adam(),
                metrics=['accuracy'])

earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5)
save_callback = tf.keras.callbacks.ModelCheckpoint(stage + "pneumonia.keras",save_freq='epoch',save_best_only=True)

if fit:
    history = model.fit(
        train_ds,
        batch_size=batch_size,
        validation_data=val_ds,
        callbacks=[save_callback, earlystop_callback],
        epochs=epochs,
        class_weight=class_weight)
else:
    model = tf.keras.models.load_model(stage + "pneumonia.keras")

#if shuffle=True when creating the dataset, samples will be chosen randomly   
score = model.evaluate(test_ds, batch_size=batch_size)
print('Test accuracy:', score[1])

y_true = np.concatenate([labels.numpy() for _, labels in test_ds])
y_pred_probs = model.predict(test_ds, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
)

print(report)

with open(stage + "reports/classification_report.txt", "w") as f:
    f.write(report)

cm = confusion_matrix(y_true, y_pred)
per_class_accuracy = cm.diagonal() / cm.sum(axis=1)

with open(stage + "reports/per_class_accuracy.txt", "w") as f:
    for i, class_name in enumerate(class_names):
        f.write(f"{class_name}: {per_class_accuracy[i]:.4f}\n")

cm_norm = confusion_matrix(y_true, y_pred, normalize="true")

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix")
plt.savefig(stage + "plots/confusion_matrix.png", bbox_inches="tight")
plt.close()

disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_names)
disp.plot(cmap="Blues", values_format=".2f")
plt.title("Normalized Confusion Matrix")
plt.savefig(stage + "plots/confusion_matrix_normalized.png", bbox_inches="tight")
plt.close()

if fit:
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(stage + 'plots/training_accuracy.png')
    plt.close()