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
import glob
import subprocess
import sys

IMG_SIZE = 50
DATA="../ressources/train_set"
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
    random.shuffle(training_data)
    save_training_data(training_data)

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

def download_stream(url):
    subprocess.call(["youtube-dl", url, '-o', 'stream.webm'])

def video_to_images(path):
    subprocess.call(["ffmpeg", "-i", path, "-vf", "fps=0.1", "stream%04d.png"])

def find_breaks():
    breaks = []
    break_time = True
    model = tensorflow.keras.models.load_model("GAI.model")
    all_images = glob.glob("stream*.png")
    all_images.sort()
    counter = 0

    while counter < len(all_images) and is_gconfs(model, all_images[counter]):
        counter += 1

    while counter < len(all_images):
        if (not is_gconfs(model, all_images[counter])):
            start = (counter - 1) * 10
            counter += 1
            while (counter < len(all_images) and not is_gconfs(model, all_images[counter])):
                counter += 1
            end = counter * 10
            breaks.append((start, end))
        counter += 1

    return breaks


def cut_stream(name, start, end):
    start_h = start // (60 * 60)
    start_m = (start - start_h * 60 * 60) // 60
    start_s = start - start_h * 60 * 60 - start_m * 60
    start_time = str(start_h) + ":" + str(start_m) + ":" + str(start_s)

    end_h = end // (60 * 60)
    end_m = (end - end_h * 60 * 60) // 60
    end_s = end - end_h * 60 * 60 - end_m * 60
    end_time = str(end_h) + ":" + str(end_m) + ":" + str(end_s)

    video_name = name + str(start) + '-' + str(end) + '.mkv'
    subprocess.call(["ffmpeg", "-i", name, "-ss", start_time, "-to", end_time, video_name])

def split_video(name):
    video_to_images(name)
    breaks = find_breaks()
    for (start, end) in breaks:
        cut_stream(name, start, end)

def main():
    i = sys.argv[1]
    if (i == '-h' or i == '-help'):
        print(''' Welcome to GAI, Gconfs Artifical Intelligence:
    -h:
    --help: print all options
    -s [arg]
    --split [arg]: split the video named [arg]
    --ds: [arg]
    --download-split [arg]: download video with url [arg] and split it
    -t:
    --train: train the neural network
    -d:
    --data: create the database to train the data''')
    elif (i == '-s' or i == '--split'):
        split_video(sys.argv[2])
    elif (i == '-t' or i == '--train'):
        train_and_save_model()
    elif (i == '-d' or i == '--data'):
        create_training_data()
    elif (i == '--ds' or i == '--download-split'):
        download_stream(sys.argv[2])
        split_video('stream.webm')

main()
