import numpy as np
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2DTranspose, Conv2D, MaxPooling2D, BatchNormalization, Activation, Dropout, UpSampling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers


def create_model(input_dim, pool_size):

    model = Sequential()
    model.add(BatchNormalization(input_shape=input_dim))

    model.add(Conv2D(16, (3, 3), activation = 'relu'))
    model.add(Conv2D(32, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Conv2D(32, (3, 3), activation = 'relu'))
    model.add(Conv2D(64, (3, 3), activation = 'relu'))
    model.add(Conv2D(64, (3, 3), activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Conv2D(128, (3, 3), activation = 'relu'))
    model.add(Conv2D(128, (3, 3), activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(UpSampling2D(size=pool_size))
    model.add(Conv2DTranspose(128,(3, 3), activation = 'relu'))
    model.add(Conv2DTranspose(128,(3, 3), activation = 'relu'))

    model.add(UpSampling2D(size=pool_size))
    model.add(Conv2DTranspose(64,(3, 3), activation = 'relu'))
    model.add(Conv2DTranspose(64,(3, 3), activation = 'relu'))
    model.add(Conv2DTranspose(32,(3, 3), activation = 'relu'))
    model.add(Dropout(0.2))

    model.add(UpSampling2D(size=pool_size))
    model.add(Conv2DTranspose(32,(3, 3), activation = 'relu'))
    model.add(Conv2DTranspose(1,(3, 3), activation = 'relu'))

    return model


def main():

    train_images = pickle.load(open("/content/drive/MyDrive/Colab Notebooks/Copy of train.p", "rb" ))
    labels = pickle.load(open("/content/drive/MyDrive/Colab Notebooks/Copy of labels.p", "rb" ))

    train_images = np.array(train_images)
    labels = np.array(labels)
    labels = labels / 255

    train_images, labels = shuffle(train_images, labels)
    X_train, X_val, y_train, y_val = train_test_split(train_images, labels, test_size=0.2)

    batch_size,pool_size,epochs = 128,(2,2),10
    input_dim = X_train.shape[1:]

    model = create_model(input_dim, pool_size)

    datagen = ImageDataGenerator(channel_shift_range=0.2)
    datagen.fit(X_train)

    model.compile(optimizer='Adam', loss='mean_squared_error')
    model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size), steps_per_epoch=len(X_train)/batch_size, epochs=epochs, verbose=1, validation_data=(X_val, y_val))

    model.trainable = False
    model.compile(optimizer='Adam', loss='mean_squared_error')

    model.save('lan_dec_model.h5')

    model.summary()

if __name__ == '__main__':
    main()

# Un-comment while running first time
!pip3 install imageio==2.4.1
!pip install scipy==1.2.3

import numpy as np
import cv2
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from tensorflow import keras
from keras.models import load_model


class Lane():
    def __init__(self):
        self.last_fit = []
        self.avg_fit = []


def road_lines(image):

    small_img = cv2.resize(image, (160,80))
    small_img = np.array(small_img)
    small_img = small_img[None, :, :, :]

    prediction = model.predict(small_img)[0] * 255

    lane.last_fit.append(prediction)
    if len(lane.last_fit) > 5:
        lane.last_fit = lane.last_fit[1:]

    lane.avg_fit = np.mean(np.array([i for i in lane.last_fit]), axis=0)

    zeros = np.zeros_like(lane.avg_fit).astype(np.uint8)
    lane_drawn = np.dstack((zeros, lane.avg_fit, zeros))
    lane_image = cv2.resize(lane_drawn, (image.shape[1], image.shape[0]))
    #lane_image = cv2.resize(lane_drawn,(854,480))
    #lane_image = lane_image.astype(np.uint8)

    print("Image shape:", image.shape)
    print("Lane image shape:", lane_image.shape)
    if lane_image.shape[2] != image.shape[2]:
        lane_image = lane_image[:, :, :image.shape[2]]
    image = image.astype(np.uint8)
    lane_image = lane_image.astype(np.uint8)



    result = cv2.addWeighted(image, 1, lane_image, 1, 0)

    return result


if __name__ == '__main__':
    model = load_model('/content/model.h5')
    lane = Lane()

    output = 'lane_dec_output.mp4'
    input_video = VideoFileClip("/content/safa.mp4")

    clip = input_video.fl_image(road_lines)
    clip.write_videofile(output, audio=False)