import os
import csv
import cv2
import json
import numpy as np
import pickle
import sklearn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import (
    Flatten,
    Dense,
    MaxPooling2D,
    Dropout,
    Lambda,
    AveragePooling2D
)
from keras.layers.convolutional import Conv2D, Cropping2D

from ConfigParser import SafeConfigParser
conf = SafeConfigParser()
conf.read('model.ini')

def read_in_csv(path, old_data=False):
    """
    Data read-in helper

    Parameters
    ----------
    path : string
        Path to a csv file
    old_data : boolean
        Indicate if to read-in udacity supplied data

    Returns
    -------
    list of paths to data
    """
    samples = []
    with open(os.path.join(path,'driving_log.csv')) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            if old_data:
                line[0] = os.path.join(path,line[0].strip())
                line[1] = os.path.join(path,line[1].strip())
                line[2] = os.path.join(path,line[2].strip())
            samples.append(line)
    return samples

def add_noise(img, thresh=30):
    """
    Randomly add noise to an image

    Parameters
    ----------
    img: numpy ndarray
        input image
    thresh: int
        threshold to be added

    Returns
    -------
    img: numpy ndarray
        output image with noise
    """
    return (img + np.random.randint(-thresh,thresh,img.shape)).astype(img.dtype)

def normalize_image(img):
    """
    Normalizes an image between 0 and 1

    Parameters
    ----------
    img : ndarray
        Input image

    Returns
    -------
    img: numpy ndarray
        output image normalized

    """
    img = img.astype(np.float32)
    img -= img.min()
    img /= img.max()
    return img

def perturb_brightness(img, thresh=0.5):
    """
    Randomly modify image brightness

    Parameters
    ----------
    img : ndarray
        input image
    thresh : float
        relative brightness modifier

    Returns
    -------
    img : ndarray
        Modified image
    """
    img = 255 * (normalize_image(img) + thresh * (np.random.rand() - 0.5))
    return (img).astype(np.uint8)

def generator(samples, conf):
    """
    Batch Image generator

    Parameters
    ----------
    conf : ConfigParser instance

    Yields
    -------
    image_batch : list
        An array of images read-into memory with a size of `batch_size`

    Raises
    ------
        N/A

    """
    batch_size = conf.getint('Train', 'batch_size')
    corr_factor = conf.getfloat('Preprocess', 'correction_factor')
    flip_thresh = conf.getfloat('Preprocess', 'flip_threshold')
    lnthresh = conf.getfloat('Preprocess', 'line_image_probality')
    perturb_prob = conf.getfloat('Preprocess', 'perturbation_probability')


    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    center_image = cv2.imread(batch_sample[i])
                    # Do some preprocessing here
                    center_angle = float(batch_sample[3])
                    if i == 1:
                        center_angle += corr_factor
                    elif i == 2:
                        center_angle -= corr_factor

                    if np.abs(center_angle) > flip_thresh:
                        # Images with steering angles greater then the
                        # flip-threshold get flipped
                        images.append(np.fliplr(center_image))
                        angles.append(-center_angle)
                    elif np.random.rand() <= lnthresh:
                        # Append only a fraction of straight-line images
                        # (> line-threshold)
                        images.append(center_image)
                        angles.append(center_angle)

                    if len(angles) > batch_size:
                        break

            images, angles = (np.array(images), np.array(angles))

            if len(angles) > batch_size:
                images = images[:batch_size]
                angles = angles[:batch_size]

            # Perturb the image quality
            for indx, image in enumerate(images):
                if np.random.rand() < perturb_prob:
                    images[indx] = add_noise(image)
                if np.random.rand() < perturb_prob:
                    images[indx] = perturb_brightness(image)

            yield np.array(images), np.array(angles)

DEFAULT_DATA_PATH_LIST = [
    'tr2_960x720',
    'tr_960x720',
    'tr_960x720_right'
]

def get_samples(data_paths=DEFAULT_DATA_PATH_LIST, test_size=0.2, plot=False):
    samples = []
    if type(data_paths) is not list and type(data_paths) is str:
        data_paths = list(data_paths)
    for path in data_paths:
        samples.extend(read_in_csv(path))
    samples = sklearn.utils.shuffle(samples)
    train, valid = train_test_split(samples, test_size=test_size)
    return train, valid

def show_samples(samples):

    plt.rcParams['figure.figsize'] = 14,12
    gen = generator(samples=samples, batch_size=100)
    [X,y] = gen.next()

    left_image_indx = np.where(y == y.min())[0]
    right_image_indx = np.where(y == y.max())[0]
    straight_image_indx = np.where(np.abs(y)==np.min(np.abs(y)))[0]

    indx = [left_image_indx[0],straight_image_indx[0],right_image_indx[0]]

    plt.subplot2grid((1,3),(0,0))
    plt.imshow(X[indx[0]])
    plt.title("Left Curve:{}".format(y[indx[0]]))
    plt.subplot2grid((1,3),(0,1))
    plt.imshow(X[indx[1]])
    plt.title("Straight:{}".format(y[indx[1]]))
    plt.subplot2grid((1,3),(0,2))
    plt.imshow(X[indx[2]])
    plt.title("Right Curve:{}".format(y[indx[2]]))
    plt.show()

def build_cnn_model(crop=(60,25)):
    UPPER_CROPPING_BOUND = crop[0]
    LOWER_CROPPING_BOUND = crop[1]
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(
        Cropping2D(
            cropping=((UPPER_CROPPING_BOUND,LOWER_CROPPING_BOUND),
            (0,0))
        )
    )
    model.add(Conv2D(12, strides=(1,1), kernel_size=(5,5), activation='relu'))
    model.add(AveragePooling2D(pool_size=(2,2)))
    model.add(Conv2D(36, strides=(1,1), kernel_size=(5,5)))
    model.add(MaxPooling2D())
    model.add(Conv2D(72, strides=(1,1), kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(36, strides=(1,1), kernel_size=(1,1), activation='relu'))
    model.add(Conv2D(12, strides=(1,1), kernel_size=(1,1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(Dropout(0.5))
    model.add(Dense(24))
    model.add(Dense(1))
    print(model.summary())
    return model

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="""
    Trains a CNN for behavioral cloning for the Udacity CARND course (1-3)
    """)


    parser.add_argument(
        '--output',
        type=str,
        default='model',
        dest='model_name',
        help="""Name the model for saving
        """
    )
    args = parser.parse_args()
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    MODEL_NAME = args.model_name

    train_samples, validation_samples = get_samples()

    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=BATCH_SIZE)
    validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

    model = build_cnn_model()
    model.compile(loss='mse', optimizer='adam')
    hobj = model.fit_generator(
        train_generator,
        steps_per_epoch = len(train_samples)//BATCH_SIZE,
        validation_data = validation_generator,
        validation_steps = len(validation_samples)//BATCH_SIZE,
        epochs=12,
        max_q_size=2,
        pickle_safe=False,
        verbose=1
    )


    plt.plot(hobj.history['loss'])
    plt.plot(hobj.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

    model.save('{name}.h5'.format(name=MODEL_NAME))
    with open('{name}.json'.format(name=MODEL_NAME), 'w') as fid:
        json.dump(model.to_json(), fid, indent=4, sort_keys=True)
