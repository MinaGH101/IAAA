import tensorflow as tf
print("Available devices:")
for device in tf.config.list_physical_devices():
    print(device)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU is available: ", gpus)
else:
    print("No GPU found")

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
import tensorflow as tf
import torch
import torch.nn.functional as F
import keras
import os
import cv2
from sklearn import metrics
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Conv3D, Dense, Dropout, BatchNormalization, Flatten
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import LeakyReLU
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from collections import Counter


MAIN_PATHS = ['/home/ai/IAAA/Data/T1/',
              '/home/ai/IAAA/Data/T2/',
              '/home/ai/IAAA/Data/FLAIR/']



def center_crop(image, target_height=224, target_width=224):

    original_height, original_width = image.shape[1], image.shape[2]

    offset_height = (original_height - target_height) // 2
    offset_width = (original_width - target_width) // 2

    cropped_image = image[:,
                        offset_height:offset_height + target_height,
                        offset_width:offset_width + target_width]

    return cropped_image


def enhance_contrast(image, alpha=1.0, beta=0.0):

    image = image.astype(np.float32)
    def enhance_contrast_for_image(image):

        channels = [image[i] for i in range(image.shape[-1])]
        enhanced_channels = [cv2.convertScaleAbs(c, alpha=alpha, beta=beta) for c in channels]
        contrast_enhanced_image = np.stack(enhanced_channels, axis=-1)

        return contrast_enhanced_image

    enhanced = np.array([enhance_contrast_for_image(img) for img in image])
    enhanced = enhanced.reshape(enhanced.shape[0], enhanced.shape[1], enhanced.shape[1])
    enhanced = np.array([cv2.transpose(s) for s in enhanced])

    return enhanced


def resize_tensor(tensor, target_size=(9, 64, 64)):
    target_depth, target_height, target_width = target_size

    resized_tensor = F.interpolate(tensor.unsqueeze(0), size=(target_height, target_width), mode='bilinear', align_corners=False).squeeze(0)

    current_depth = resized_tensor.size(0)
    if current_depth < target_depth:
        padding = (0, 0, 0, 0, 0, target_depth - current_depth)
        resized_tensor = F.pad(resized_tensor, padding, mode='constant', value=0)
    elif current_depth > target_depth:
        resized_tensor = resized_tensor[:target_depth, :, :]

    # resized_tensor = resized_tensor[3:target_depth-4, :, :]

    return resized_tensor


def dataset(main_paths): # returns images and labels lists.
  images = []
  labels = []
  for main_path in main_paths:
    for npz_file in os.listdir(main_path)[:100]:
        if npz_file.endswith('.npz'):
            npz_file_path = os.path.join(main_path, npz_file)
            data = np.load(npz_file_path)
            image = data['x']
            image = image.astype(np.float32)
            # print([normalize(i) for i in image])
            image = center_crop(image)
            image = enhance_contrast(image)
            image = torch.tensor(image)
            image = resize_tensor(image)

            label = data['y']

            images.append(np.array(image))
            labels.append(label)

  return images, labels




images, labels = dataset(MAIN_PATHS)
X = np.array(images)
y = np.array(labels)
print(len(images))
print(len(labels))
print(X.shape)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=25, shuffle=True)


sample_weight = np.ones(shape=(len(y_train),))
sample_weight[y_train == 1] = 3.0

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train, sample_weight))

train_dataset = train_dataset.shuffle(buffer_size=25).batch(bs)


Lr = 0.0005
bs = 16
epochs = 5



checkpoint_filepath = '/tmp/ckpt/checkpoint.model.keras'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)



tf.random.set_seed(100)
model = models.Sequential([
    Conv3D(32, (3, 3, 3), activation='relu', kernel_regularizer=l2(0.001), input_shape=(9, 64, 64, 1)),
    BatchNormalization(),
    Dropout(0.3),
    Conv3D(16, (3, 3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),
    Conv3D(8, (3, 3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),
    Flatten(),
    Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    # Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    Dense(1, activation='sigmoid')
])

Adam = keras.optimizers.Adam(learning_rate=Lr)
model.compile(optimizer=Adam, loss='binary_crossentropy', metrics=['accuracy', 'auc', 'recall'])

model.fit(X_train, y_train, epochs=epochs, batch_size=bs, validation_data=(X_val, y_val), callbacks=[model_checkpoint_callback])

m = keras.models.load_model(checkpoint_filepath)
m.save('/home/ai/IAAA/model_V2_best_95acc_auc0.9.keras')

y_pred = m.predict(X_val)
y_pred = [np.round(i, 0) for i in y_pred]

print(m.evaluate(X_val, y_val))
print('precision score:', metrics.precision_score(y_val, y_pred))
print('recall score:', metrics.recall_score(y_val, y_pred))
print('f1 score:', metrics.f1_score(y_val, y_pred))

confusion_matrix = metrics.confusion_matrix(y_val, y_pred)
print(confusion_matrix)