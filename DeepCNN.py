import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, pooling, Input
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras.optimizers import Adam, Adagrad, RMSprop, Adadelta

np.random.seed(777)  # for reproducibility

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28,28,1)
X_test = X_test.reshape(X_test.shape[0], 28,28,1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

cnn = Sequential()
cnn.add(ZeroPadding2D((2, 2), input_shape=(28, 28, 1)))
cnn.add(Conv2D(64, (5, 5), kernel_initializer='he_normal'))
cnn.add(Activation('relu'))

cnn.add(ZeroPadding2D((2, 2)))
cnn.add(Conv2D(128, (5, 5), kernel_initializer='he_normal'))
cnn.add(Activation('relu'))
cnn.add(MaxPooling2D(strides=(2, 2)))

cnn.add(ZeroPadding2D((2, 2)))
cnn.add(Conv2D(256, (5, 5), kernel_initializer='he_normal'))
cnn.add(Activation('relu'))
cnn.add(ZeroPadding2D((1, 1)))
cnn.add(Conv2D(256, (3, 3), kernel_initializer='he_normal'))
cnn.add(Activation('relu'))
cnn.add(MaxPooling2D(strides=(2, 2)))
cnn.add(Dropout(0.2))

cnn.add(ZeroPadding2D((1, 1)))
cnn.add(Conv2D(512, (3, 3), kernel_initializer='he_normal'))
cnn.add(Activation('relu'))
cnn.add(Dropout(0.2))
cnn.add(ZeroPadding2D((1, 1)))
cnn.add(Conv2D(512, (3, 3), kernel_initializer='he_normal'))
cnn.add(Activation('relu'))
cnn.add(MaxPooling2D(strides=(2, 2)))

cnn.add(ZeroPadding2D((1, 1)))
cnn.add(Conv2D(1024, (3, 3), kernel_initializer='he_normal'))
cnn.add(Activation('relu'))
cnn.add(Dropout(0.2))
cnn.add(ZeroPadding2D((1, 1)))
cnn.add(Conv2D(1024, (3, 3), kernel_initializer='he_normal'))
cnn.add(Activation('relu'))
cnn.add(MaxPooling2D(strides=(2, 2)))

cnn.add(Flatten())
cnn.add(Dropout(0.5))
cnn.add(Dense(2048, activation="relu", kernel_initializer='he_normal'))
cnn.add(Dense(128, activation="relu", kernel_initializer='he_normal'))
cnn.add(Dense(10, activation="softmax"))

cnn.summary()

opt = Adagrad(lr=0.001, epsilon=1e-8, decay=0.)
cnn.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

cnn.fit(X_train, Y_train, batch_size=64, shuffle=True, epochs=50, validation_split=0.1)

score = cnn.evaluate(X_test, Y_test)
print(cnn.metrics_names)
print(score)

f = open("./saved/MNIST_DeepCNN_model.json", 'w')
f.write(cnn.to_json())
f.close()
cnn.save_weights('./saved/MNIST_DeepCNN_weight.h5')