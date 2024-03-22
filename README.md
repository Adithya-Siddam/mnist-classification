# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset
Digit categorization of scanned handwriting images, together with answer verification.
There are a number of handwritten digits in the MNIST dataset. The assignment is to place a handwritten digit picture into one of ten classes that correspond to integer values from 0 to 9, inclusively. The dataset consists of 60,000 handwritten digits that are each 28 by 28 pixels in size. In this case, we construct a convolutional neural network model that can categorise to the relevant numerical value.
## Neural Network Model

![image](https://github.com/Adithya-Siddam/mnist-classification/assets/93427248/ccd82fe6-a9b3-4984-9b82-2a2495f786e6)


## DESIGN STEPS

### STEP 1: 

Import the required packages

### STEP 2: 

Load the dataset

### STEP 3: 

Scale the dataset

### STEP 4: 

Use the one-hot encoder

### STEP 5: 

Create the model

### STEP 6: 

Compile the model

### STEP 7: 

Fit the model

### STEP 8: 

Make prediction with test data and with an external data


## PROGRAM :

```
Developed  By : S Adithya Chowdary.
Reference Number : 212221230100.
```

## Importing the required packages
~~~

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape
X_test.shape
single_image= X_train[52000]
single_image.shape
plt.imshow(single_image,cmap='gray')
y_train.shape
X_train.min()
X_train.max()

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
X_train_scaled.min()
X_train_scaled.max()
y_train[0]
y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)
type(y_train_onehot)
y_train_onehot.shape
single_image = X_train[7800]
plt.imshow(single_image,cmap='gray')
y_train_onehot[7800]
X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)
Ai_img = keras.Sequential()
Ai_img.add(layers.Input(shape=(28,28,1))) 
Ai_img.add(layers.Conv2D(filters=64,kernel_size=(3,3),activation="relu")) 
Ai_img.add(layers.MaxPool2D(pool_size=(2,2))) 
Ai_img.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation="relu")) 
Ai_img.add(layers.MaxPool2D(pool_size=(2,2))) 
Ai_img.add(layers.Flatten()) 
Ai_img.add(layers.Dense(128,activation="relu"))
Ai_img.add(layers.Dense(64)) 
Ai_img.add(layers.Dense(32)) 
Ai_img.add(layers.Dense(10,activation="softmax"))
Ai_img.summary()
Ai_img.compile(loss='categorical_crossentropy',optimizer='adam',metrics='accuracy')
Ai_img.fit(X_train_scaled ,y_train_onehot, epochs=5,
          batch_size=64, 
          validation_data=(X_test_scaled,y_test_onehot))
metrics = pd.DataFrame(Ai_img.history.history)
metrics.head()
metrics[['accuracy','val_accuracy']].plot()
metrics[['loss','val_loss']].plot()
x_test_predictions = np.argmax(Ai_img.predict(X_test_scaled), axis=1)
print(confusion_matrix(y_test,x_test_predictions))
print(classification_report(y_test,x_test_predictions))
img = image.load_img('4.jpeg')
type(img)
img = image.load_img('4.jpeg')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0
x_single_prediction = np.argmax(
    Ai_img.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)
print(x_single_prediction)
print("Adithya Chowdary")
print("212221230100")
plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')
img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0
x_single_prediction = np.argmax(
    Ai_img.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),
     axis=1)
print("Adithya Chowdary")
print("212221230100")
print("Prediction output:",x_single_prediction)

~~~

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![image](https://github.com/Adithya-Siddam/mnist-classification/assets/93427248/dbfd4def-51c9-467b-b6c4-c7da2189ed33)
![image](https://github.com/Adithya-Siddam/mnist-classification/assets/93427248/c910a611-a644-4190-a4d2-81f24d9092a5)


### Classification Report

![image](https://github.com/Adithya-Siddam/mnist-classification/assets/93427248/04f944aa-9e94-4f58-8de7-9e814cdc264d)


### Confusion Matrix
![image](https://github.com/Adithya-Siddam/mnist-classification/assets/93427248/eb2c6158-81ae-45c6-9498-6d87e3892924)



### New Sample Data Prediction
![image](https://github.com/Adithya-Siddam/mnist-classification/assets/93427248/0793ee07-af85-4c09-9d78-dc90da736a42)


![image](https://github.com/Adithya-Siddam/mnist-classification/assets/93427248/034c090e-c9b3-46ce-897c-5b1ec922451a)


## RESULT
Therefore a model has been successfully created for digit classification using mnist dataset.
