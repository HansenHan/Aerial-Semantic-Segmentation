import cv2
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.utils import Sequence, to_categorical, plot_model
from keras.layers import Conv2D, Dropout, MaxPooling2D, UpSampling2D, concatenate, Input
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping

path = '/kaggle/input/semantic-drone-dataset/dataset/semantic_drone_dataset/'
#path='C:/STUDY/McMaster/SEP769/archive/dataset/semantic_drone_dataset/'
#path='F:/McMaster/SEP769/archive/dataset/semantic_drone_dataset'
image = cv2.imread(path + 'original_images/001.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mask = cv2.imread(path + 'label_images_semantic/001.png', cv2.IMREAD_GRAYSCALE)



# Data Preprocessing
X = []
for filename in sorted(os.listdir(path + 'original_images/')):
    img = cv2.imread(path + 'original_images/' + filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img = img / 255
    X.append(img)
    
X = np.array(X)


Y = []
for filename in sorted(os.listdir(path + 'label_images_semantic/')):
    img = cv2.imread(path + 'label_images_semantic/' + filename, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))
    Y.append(img)
    
Y = np.array(Y)
Yc = to_categorical(Y)



xTrain, xValidation, yTrain, yValidation = train_test_split(X[0:-40], Yc[0:-40], test_size = 0.15)


#build model
def Unetmodel(num_classes = 23, image_shape = (256, 256, 3)):
    # Input
    inputs = Input(image_shape)
    c1 = Conv2D(64, 3, activation='relu', kernel_initializer = 'he_normal', padding='same')(inputs)
    c1 = Conv2D(64, 3, activation='relu', kernel_initializer = 'he_normal', padding='same')(c1)
    pool1 = MaxPooling2D((2,2))(c1)
    
    c2 = Conv2D(128, 3, activation='relu', kernel_initializer = 'he_normal', padding='same')(pool1)
    c2 = Conv2D(128, 3, activation='relu', kernel_initializer = 'he_normal', padding='same')(c2)
    pool2 = MaxPooling2D((2,2))(c2)

    c3 = Conv2D(256, 3, activation='relu', kernel_initializer = 'he_normal', padding='same')(pool2)
    c3 = Conv2D(256, 3, activation='relu', kernel_initializer = 'he_normal', padding='same')(c3)
    pool3 = MaxPooling2D((2,2))(c3)
    
    c4 = Conv2D(512, 3, activation='relu', kernel_initializer = 'he_normal', padding='same')(pool3)
    c4 = Conv2D(512, 3, activation='relu', kernel_initializer = 'he_normal', padding='same')(c4)
    drop4 = Dropout(0.5)(c4)
    pool4 = MaxPooling2D((2,2))(drop4)
    
    c5 = Conv2D(1024, 3, activation='relu', kernel_initializer = 'he_normal', padding='same')(pool4)
    c5 = Conv2D(1024, 3, activation='relu', kernel_initializer = 'he_normal', padding='same')(c5)
    drop5 = Dropout(0.5)(c5)
    
    up6 = Conv2D(512, 2, activation='relu', kernel_initializer='he_normal', padding='same')(UpSampling2D(size=(2,2))(drop5))
    merge6 = concatenate([up6, c4], axis = 3)
    c6 = Conv2D(512, 3, activation='relu', kernel_initializer='he_normal', padding='same')(merge6)
    c6 = Conv2D(512, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    
    up7 = Conv2D(256, 2, activation='relu', kernel_initializer='he_normal', padding='same')(UpSampling2D(size=(2,2))(c6))
    merge7 = concatenate([up7, c3], axis = 3)
    c7 = Conv2D(256, 3, activation='relu', kernel_initializer='he_normal', padding='same')(merge7)
    c7 = Conv2D(256, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    
    up8 = Conv2D(128, 2, activation='relu', kernel_initializer='he_normal', padding='same')(UpSampling2D(size=(2,2))(c7))
    merge8 = concatenate([up8, c2], axis = 3)
    c8 = Conv2D(128, 3, activation='relu', kernel_initializer='he_normal', padding='same')(merge8)
    c8 = Conv2D(128, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    
    up9 = Conv2D(64, 2, activation='relu', kernel_initializer='he_normal', padding='same')(UpSampling2D(size=(2,2))(c8))
    merge9 = concatenate([up9, c1], axis = 3)
    c9 = Conv2D(64, 3, activation='relu', kernel_initializer='he_normal', padding='same')(merge9)
    c9 = Conv2D(64, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    
    c10 = Conv2D(num_classes, (1, 1), padding='same', activation='softmax')(c9)
    
    model = Model(inputs, c10)
    
    return model

model = Unetmodel()
model.summary()

plot_model(model)

model_checkpoint = ModelCheckpoint('unet_model.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
model_earlyStopping = EarlyStopping(min_delta= 0.001, patience=15)

model.compile(optimizer='adam', loss=['categorical_crossentropy'], metrics=['accuracy'])

#train model
history = model.fit(x=xTrain, y=yTrain,
              validation_data=(xValidation, yValidation),
              batch_size=4, epochs=100,
              callbacks=[model_checkpoint, model_earlyStopping])

#result
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Train Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Train Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training & Validation Loss plot')
plt.show()


test1 = X[-1]
test2 = X[-2]
test1mask = Yc[-1]
test2mask = Yc[-2]

image1 = test1
pred = model.predict(np.expand_dims(image1, 0))
pred_mask = np.argmax(pred, axis=-1)
print(pred_mask.shape)
pred_mask = pred_mask[0]
print(pred_mask.shape)



fig, axs = plt.subplots(1, 3, figsize=(20, 10))
axs[0].imshow(image1)
axs[0].set_title('Image')
axs[1].imshow(np.argmax(test1mask, axis=-1))
axs[1].set_title('Ground Truth')
axs[2].imshow(pred_mask)
axs[2].set_title('Prediction')



intersection = np.logical_and(np.argmax(test1mask, axis=-1), pred_mask)
union = np.logical_or(np.argmax(test1mask, axis=-1), pred_mask)
iou_score1 = np.sum(intersection) / np.sum(union)
print("iou_score1",iou_score1)


pix_acc1=np.sum(np.equal(np.argmax(test1mask, axis=-1), pred_mask))/(256*256)
print("pix_acc1",pix_acc1)

image2 = test2
pred2 = model.predict(np.expand_dims(image2, 0))
pred_mask2 = np.argmax(pred2, axis=-1)
print(pred_mask2.shape)
pred_mask2 = pred_mask2[0]
print(pred_mask2.shape)

fig, axs = plt.subplots(1, 3, figsize=(20, 10))
axs[0].imshow(image2)
axs[0].set_title('Image')
axs[1].imshow(np.argmax(test2mask, axis=-1))
axs[1].set_title('Ground Truth')
axs[2].imshow(pred_mask2)
axs[2].set_title('Prediction')

intersection1 = np.logical_and(np.argmax(test2mask, axis=-1), np.argmax(pred2, axis=-1))
union1 = np.logical_or(np.argmax(test2mask, axis=-1), np.argmax(pred2, axis=-1))
iou_score2 = np.sum(intersection1) / np.sum(union1)
print("iou_score2",iou_score2)

pix_acc2=np.sum(np.equal(np.argmax(test2mask, axis=-1), pred_mask2))/(256*256)
print("pix_acc2:",pix_acc2)


