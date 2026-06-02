# Pneumonia X-Ray Classification

A convolutional neural network for classifying chest X-ray images into normal, bacterial pneumonia, or viral pneumonia. The project focuses on improving sick-patient recall, reducing overfitting, handling class imbalance, and adding explainability through Grad-CAM visualisations.

## Problem Statement

Pneumonia diagnosis from chest X-rays is a medical image classification problem where false negatives are especially important. The main objective of this project was to improve classification performance across three classes:

* Normal
* Bacterial pneumonia
* Viral pneumonia

Although the model predicts three separate classes, the project also considered the practical importance of identifying sick patients by focusing on recall for both bacterial and viral pneumonia.

## Dataset

* Training images: 5,419
* Test images: 438
* Classes: Normal, bacterial pneumonia, viral pneumonia

Class distribution:

| Dataset | Bacterial | Normal | Viral |
| ------- | --------: | -----: | ----: |
| Train   |     2,596 |  1,461 | 1,362 |
| Test    |       184 |    122 |   132 |

The dataset was moderately imbalanced, with bacterial pneumonia being the most common class in the training set.

## Baseline Model

The baseline CNN used:

* 3 convolutional layers
* Max pooling after each convolutional layer
* Flatten layer
* Dense layer with 512 neurons
* Dropout of 0.2
* 3-node output layer

Baseline test performance:

| Metric      | Score |
| ----------- | ----: |
| Accuracy    | 0.748 |
| Macro F1    | 0.737 |
| Weighted F1 | 0.735 |

The main limitation of the baseline model was poor viral recall, which was only 0.458. Training curves also showed clear overfitting after the first epoch.

## Methodology

### Data Augmentation

To reduce overfitting, augmentation was applied only to the training data. Tested augmentations included:

* Random horizontal flips
* Small rotations
* Small zoom adjustments

The best augmentation setup substantially reduced overfitting and improved generalisation, increasing test accuracy to 0.812.

### Greyscale Preprocessing

Because X-ray images are naturally greyscale, the images were converted from RGB to greyscale to reduce unnecessary input complexity. This maintained performance while reducing training time from around 10 seconds per epoch to around 8 seconds per epoch.

### Class Imbalance Handling

Class weights were introduced to improve performance on underrepresented classes, especially viral pneumonia. This improved viral recall from 0.580 to 0.802, helping the model identify more sick patients.

### CNN Architecture Tuning

Several model configurations were tested, including larger convolutional layers, global average pooling, different dense layer sizes, and batch normalisation.

The best custom CNN used:

* Input size: 128 × 128 greyscale images
* Convolutional layer with 32 filters
* Convolutional layer with 64 filters
* Convolutional layer with 128 filters
* Max pooling after each convolutional layer
* Dense layer with 512 neurons
* Dropout of 0.2
* 3-class output layer

### Transfer Learning

EfficientNetB0 was tested as a transfer learning approach. It underperformed compared to the custom CNN, likely because the model was too complex for the task and dataset setup.

### Explainability

Grad-CAM visualisations were generated for correctly classified examples from each class. These heatmaps helped inspect whether the CNN was focusing on relevant regions of the X-ray, such as the lungs and ribcage, or potentially noisy background areas.

## Results

| Model Stage              | Test Accuracy | Macro F1 |
| ------------------------ | ------------: | -------: |
| Baseline CNN             |         0.748 |    0.737 |
| Augmentation + greyscale |         0.817 |    0.813 |
| Class weighting          |         0.812 |    0.820 |
| Final custom CNN         |         0.826 |    0.829 |

Final model classification performance:

| Class     | Precision | Recall |    F1 |
| --------- | --------: | -----: | ----: |
| Bacterial |     0.830 |  0.793 | 0.811 |
| Normal    |     0.902 |  0.984 | 0.941 |
| Viral     |     0.742 |  0.725 | 0.734 |

## Key Achievement

Improved test accuracy from 74.8% to 82.6% and macro F1 from 0.737 to 0.829 through data augmentation, greyscale preprocessing, class imbalance handling, CNN architecture tuning, and model explainability.

## Technologies

Python • TensorFlow • Keras • scikit-learn • NumPy • matplotlib • seaborn • CNNs • Grad-CAM • EfficientNetB0
