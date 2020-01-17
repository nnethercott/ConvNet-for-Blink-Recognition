#this is a test

import keras
from sklearn.model_selection import train_test_split
from keras import layers
from keras import models

inp = layers.Input(shape=(500,1))
C = layers.Conv1D(filters=32, kernel_size=5, strides=1)(inp)

C11 = layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu')(C)
C12 = layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(C11)
S11 = layers.Add()([C12,C])
A1 = layers.Activation(activation='relu')(S11)
M11 = layers.MaxPooling1D(pool_size=5, strides=2)(A1)
N1 = layers.BatchNormalization()(M11)

C21 = layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu')(N1)
C22 = layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(C21)
S21 = laye

rs.Add()([C22,M11])
A2 = layers.Activation(activation='relu')(S21)
M21 = layers.MaxPooling1D(pool_size=5, strides=2)(A2)

C31 = layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu')(M21)
C32 = layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(C31)
S31 = layers.Add()([C32,M21])
A3 = layers.Activation(activation='relu')(S31)
M31 = layers.MaxPooling1D(pool_size=5, strides=2)(A3)

C41 = layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu')(M31)
C42 = layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(C41)
S41 = layers.Add()([C42,M31])
A4 = layers.Activation(activation='relu')(S41)
M41 = layers.MaxPooling1D(pool_size=5, strides=2)(A4)

C51 = layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu')(M41)
C52 = layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(C51)
S51 = layers.Add()([C52,M41])
A5 = layers.Activation(activation='relu')(S51)
M51 = layers.MaxPooling1D(pool_size=5, strides=2)(A5)

F1 = layers.Flatten()(M51)

D1 = layers.Dense(32, activation = 'relu')(F1)
D2 = layers.Dense(32)(D1)
D3 = layers.Dense(2)(D2)
A6 = layers.Softmax()(D3)

model1 = models.Model(inputs=inp, outputs=A6)


#training
model1.compile(loss='categorical_crossentropy', optimizer = 'Adam', metrics=['accuracy'])

x, y = data_prep(data_stim, data_labels, 250, 500, 100, 1)
x = x.reshape(x.shape[0], x.shape[1], 1)
y = keras.utils.to_categorical(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7)

model1.fit(x_train, y_train, epochs=18, verbose=2, validation_data = (x_test, y_test))
