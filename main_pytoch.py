# import tensorflow as tf
# print("Available devices:")
# for device in tf.config.list_physical_devices():
#     print(device)

# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     print("GPU is available: ", gpus)
# else:
#     print("No GPU found")

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import tensorflow as tf
import keras
import os
import cv2
from sklearn import metrics
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


MAIN_PATHS = ['/home/ai/IAAA/Data/T1/',
              '/home/ai/IAAA/Data/T2/',
              '/home/ai/IAAA/Data/FLAIR/']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

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
    for npz_file in os.listdir(main_path)[:]:
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

Lr = 0.001
bs = 32
epochs = 20

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

X_test_tensor = torch.tensor(X_val, dtype=torch.float32)
y_test_tensor = torch.tensor(y_val, dtype=torch.long)
X_train_tensor = X_train_tensor.unsqueeze(1)
X_test_tensor = X_test_tensor.unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)


# checkpoint_filepath = '/tmp/ckpt/checkpoint.model.keras'
# model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_filepath,
#     monitor='val_accuracy',
#     mode='max',
#     save_best_only=True)



class CNN3DModel(nn.Module):
    def __init__(self):
        super(CNN3DModel, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=(3, 3, 3), padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 16, kernel_size=(3, 3, 3), padding=1)
        self.bn2 = nn.BatchNorm3d(16)
        self.conv3 = nn.Conv3d(16, 8, kernel_size=(3, 3, 3), padding=1)
        self.bn3 = nn.BatchNorm3d(8)
        self.dropout = nn.Dropout(0.3)
        self.flattened_size = 8 * 9 * 64 * 64
        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

model = CNN3DModel().to(device)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
criterion = nn.BCELoss()


# dummy_input = torch.zeros(1, 1, 9, 64, 64)
# flattened_size = model._get_flattened_size(dummy_input)
# model.fc1 = nn.Linear(flattened_size, 256)



def train_model(model, criterion, optimizer, train_loader, test_loader, epochs, device):
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device) 
            optimizer.zero_grad() 
            outputs = model(inputs) 
            loss = criterion(outputs, labels.unsqueeze(1).float())
            loss.backward() 
            optimizer.step() 
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device) 
                outputs = model(inputs)  # Forward pass
                loss = criterion(outputs, labels.unsqueeze(1).float())  
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                correct += (predicted == labels.unsqueeze(1).float()).sum().item()
                total += labels.size(0)

        val_accuracy = correct / total
        print(f'Epoch {epoch+1}/{epochs}, Training Loss: {train_loss/len(train_loader):.4f}, '
              f'Validation Loss: {val_loss/len(test_loader):.4f}, Validation Accuracy: {val_accuracy:.4f}')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCELoss()

train_model(model, criterion, optimizer, train_loader, test_loader, epochs=epochs, device=device)








# m = keras.models.load_model(checkpoint_filepath)
# m.save('/home/ai/IAAA/model_V2_best_95acc_auc0.9.keras')

# y_pred = m.predict(X_val)
# y_pred = [np.round(i, 0) for i in y_pred]

# print(m.evaluate(X_val, y_val))
# print('precision score:', metrics.precision_score(y_val, y_pred))
# print('recall score:', metrics.recall_score(y_val, y_pred))
# print('f1 score:', metrics.f1_score(y_val, y_pred))

# confusion_matrix = metrics.confusion_matrix(y_val, y_pred)
# print(confusion_matrix)