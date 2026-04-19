import os

import keras
import tensorflow as tf
import matplotlib as mpl
import numpy as np

batch_size = 16
num_classes = 3
epochs = 30
img_width = 128
img_height = 128
img_channels = 1
stage = './'
os.makedirs(stage, exist_ok=True)
os.makedirs(stage + 'plots', exist_ok=True)
os.makedirs(stage + 'reports', exist_ok=True)

train_dir = 'chest_xray/chest_xray/train/'
test_dir = 'chest_xray/chest_xray/test/'

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
    color_mode='grayscale')

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    seed=None,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    labels='inferred',
    shuffle=False,
    verbose=False,
    color_mode='grayscale')

class_names = train_ds.class_names
print('Class Names: ',class_names)
num_classes = len(class_names)

model = tf.keras.models.load_model(stage + "pneumonia.keras")
_ = model(tf.zeros((1, img_height, img_width, img_channels)), training=False)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names, pred_index=None):
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = tf.keras.models.Model(model.inputs, last_conv_layer.output)

    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = tf.keras.models.Model(classifier_input, x)

    with tf.GradientTape() as tape:
        last_conv_output = last_conv_layer_model(img_array, training=False)
        tape.watch(last_conv_output)

        preds = classifier_model(last_conv_output, training=False)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_output)

    if grads is None:
        raise ValueError("Gradients are still None.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_output = last_conv_output[0]
    heatmap = tf.reduce_sum(last_conv_output * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(pred_index)

def save_gradcam(img_tensor, heatmap, cam_path="plots/gradcam.png", alpha=0.4):
    img = img_tensor.numpy()

    if img.shape[-1] == 1:
        base_img = np.repeat(img, 3, axis=-1)
    else:
        base_img = img.copy()

    heatmap = np.uint8(255 * heatmap)

    jet = mpl.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((base_img.shape[1], base_img.shape[0]))
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + base_img
    superimposed_img = tf.keras.utils.array_to_img(superimposed_img)
    superimposed_img.save(cam_path)

wanted_labels = set(range(len(class_names)))

for images, labels in test_ds:
    for i in range(len(labels)):
        true_label = int(labels[i].numpy())
        if true_label in wanted_labels:
            img = images[i]
            img_array = tf.expand_dims(img, axis=0)
            preds = model.predict(img_array, verbose=0)
            pred_label = int(np.argmax(preds[0]))

            if pred_label == true_label:
                print("True:", class_names[true_label])
                print("Pred:", class_names[pred_label])

                heatmap, pred_index = make_gradcam_heatmap(
                    img_array,
                    model,
                    last_conv_layer_name="conv2d_2",
                    classifier_layer_names=[
                        "max_pooling2d_2",
                        "flatten",
                        "dense",
                        "dropout",
                        "dense_1"
                    ]
                )
                save_gradcam(
                    img,
                    heatmap,
                    cam_path=f"plots/gradcam_{class_names[true_label]}.png",
                    alpha=0.4
                )
                wanted_labels.remove(true_label)
        if not wanted_labels:
            break
    if not wanted_labels:
        break