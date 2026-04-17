from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Rescaling, BatchNormalization
from keras.optimizers import RMSprop,Adam
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


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
epochs = 8
img_width = 128
img_height = 128
img_channels = 3
fit = True #make fit false if you do not want to train the network again

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
    verbose=False)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    seed=None,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    labels='inferred',
    shuffle=False,
    verbose=False)

class_names = train_ds.class_names
print('Class Names: ',class_names)
num_classes = len(class_names)

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

#create model
model = tf.keras.models.Sequential([
    Rescaling(1.0/255),
    Conv2D(16, (3,3), activation = 'relu', input_shape = (img_height,img_width, img_channels)),
    MaxPooling2D(2,2),
    Conv2D(32, (3,3), activation = 'relu'),
    MaxPooling2D(2,2),
    Conv2D(32, (3,3), activation = 'relu'),
    MaxPooling2D(2,2),
    Flatten(), # flatten multidimensional outputs into single dimension for input to dense fully connected layers
    Dense(512, activation = 'relu'),
    Dropout(0.2),
    Dense(num_classes, activation = 'softmax')
])

model.compile(loss='sparse_categorical_crossentropy',
                optimizer=Adam(),
                metrics=['accuracy'])

#earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5)
save_callback = tf.keras.callbacks.ModelCheckpoint("pneumonia.keras",save_freq='epoch',save_best_only=True)

if fit:
    history = model.fit(
        train_ds,
        batch_size=batch_size,
        validation_data=val_ds,
        callbacks=[save_callback],
        epochs=epochs)
else:
    model = tf.keras.models.load_model("pneumonia.keras")

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

with open("reports/classification_report.txt", "w") as f:
    f.write(report)

cm = confusion_matrix(y_true, y_pred)
per_class_accuracy = cm.diagonal() / cm.sum(axis=1)

with open("reports/per_class_accuracy.txt", "w") as f:
    for i, class_name in enumerate(class_names):
        f.write(f"{class_name}: {per_class_accuracy[i]:.4f}\n")

cm_norm = confusion_matrix(y_true, y_pred, normalize="true")

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix")
plt.savefig("plots/confusion_matrix.png", bbox_inches="tight")
plt.close()

disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_names)
disp.plot(cmap="Blues", values_format=".2f")
plt.title("Normalized Confusion Matrix")
plt.savefig("plots/confusion_matrix_normalized.png", bbox_inches="tight")
plt.close()

if fit:
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('plots/training_accuracy.png')
    plt.close()