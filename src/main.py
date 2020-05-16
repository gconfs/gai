import numpy as np
import os
import cv2
from tqdm import tqdm
import random
import pickle
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Flatten

IMG_SIZE = 50
DATA="/home/enulp/ressources/train_set"
CATEGORIES=["gconfs", "gconf"]


def create_training_data():
    training_data = []
    for category in CATEGORIES:
        path = os.path.join(DATA, category)
        class_num = CATEGORIES.index(category)

        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path, img), \
                        cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
    training_data
    random.shuffle(training_data)
    return training_data

def save_training_data(training_data):
    X = []
    Y = []

    for features, label in training_data:
        X.append(features)
        Y.append(label)

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    Y = np.array(Y)

    NAME = "GAI"

    pickle_out = open("X.pickle", "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open("Y.pickle", "wb")
    pickle.dump(Y, pickle_out)
    pickle_out.close()


def train_and_save_model():

    pickle_in = open("X.pickle", "rb")
    X = pickle.load(pickle_in)

    pickle_in = open("Y.pickle", "rb")
    Y = pickle.load(pickle_in)

    X = X / 255.0

    model = Sequential()

    model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(64))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    model.fit(X, Y, batch_size = 100, epochs = 50, validation_split = 0.3)
    model.save('GAI.model')

def prepare(filepath):
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

def is_gconfs(model, filepath):
    prediction = model.predict([prepare(filepath)])
    return CATEGORIES[int(prediction[0][0])] == 'gconfs'

def main():
    model = tensorflow.keras.models.load_model("GAI.model")
    print(is_gconfs(model, 'gconf.png'))

#data_set = create_training_data()
#save_training_data(data_set)
#train_and_save_model()

main()
