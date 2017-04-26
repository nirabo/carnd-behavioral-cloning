import os
import sys
import csv
import cv2
import json
import numpy as np
import pickle
import sklearn
import argparse
import shutil
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

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

            yield images, angles

def get_samples(path_list, test_size=0.2,  plot=False):

    samples = []
    if type(path_list) is not list and type(path_list) is str:
        path_list = list(path_list)
    for path in path_list:
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

def build_model(conf):

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

    UPPER_CROPPING_BOUND = conf.getint('Model', 'upper_cropping_bound')
    LOWER_CROPPING_BOUND = conf.getint('Model', 'lower_cropping_bound')
    DROPOUT = conf.getfloat('Model', 'dropout')

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
    model.add(Dropout(DROPOUT))
    model.add(Dense(128))
    model.add(Dropout(DROPOUT))
    model.add(Dense(24))
    model.add(Dense(1))
    print(model.summary())
    return model

def save_model(model, run_name):
    data_dir = os.path.join('runs', run_name)
    try:
        os.mkdir(data_dir)
    except OSError as e:
        print(e)
        ans  = raw_input('Run with same name exists. Do you want to override it (N/y): ')
        if not 'y' in ans.lower():
            return False

    model.save(os.path.join(data_dir, 'model.h5'))
    with open(os.path.join(data_dir, 'model.json'), 'w') as fid:
        json.dump(model.to_json(), fid)
    shutil.copyfile('model.ini', os.path.join(data_dir, 'model.ini'))

    plt.plot(hobj.history['loss'])
    plt.plot(hobj.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig(os.path.join(data_dir, 'model.png'))
    plt.show()

def load_model(path):
    from keras.models import model_from_json
    with open(os.path.join(path, 'model.json'), 'r') as fid:
        model = model_from_json(json.load(fid))
    model.load_weights(os.path.join(path, 'model.h5'))
    return model

def parser():
    parser = argparse.ArgumentParser(description="""
    Trains a CNN for behavioral cloning for the Udacity CARND course
    """)

    parser.add_argument(
        '-o',
        type=str,
        default=None,
        dest='output',
        help="""Specify output model name (Required!)
        """
    )

    parser.add_argument(
        '-i',
        type=str,
        default=None,
        dest='input',
        help="""Input a pre-trained model (path to a past run directory)
        """
    )

    parser.add_argument(
        '-s',
        default=False,
        dest="plot_epochs",
        help="Plot model performance"
    )
    args = parser.parse_args()
    if args.output is None:
        print("Output model name required")
        sys.exit(1)
    return args

if __name__ == "__main__":

    args = parser()
    BATCH_SIZE = conf.getint('Train', 'batch_size')
    EPOCHS = conf.getint('Train', 'epochs')
    RUNS = conf.options('Run')
    SPLIT_RATIO = conf.getfloat('Train', 'split')
    MAX_Q_SIZE = conf.getint('Train', 'max_queue_size')
    OVERLAP_FACTOR = conf.getfloat('Train', 'overlap_factor')

    if args.input is not None:
        model = load_model(args.input)
    else:
        model = build_model(conf)
    model.compile(loss='mse', optimizer='adam')

    for run_indx, run in enumerate(RUNS):
        path_list = [os.path.join('data', path) for path in conf.get('Run', run).split()]
        train_samples, validation_samples = get_samples(path_list, test_size=SPLIT_RATIO)
        # compile and train the model using the generator function
        train_gen = generator(train_samples,conf)
        valid_gen = generator(validation_samples,conf)

        hobj = model.fit_generator(
            train_gen,
            steps_per_epoch = OVERLAP_FACTOR * len(train_samples)//BATCH_SIZE,
            validation_data = valid_gen,
            validation_steps = OVERLAP_FACTOR * len(validation_samples)//BATCH_SIZE,
            epochs=EPOCHS,
            max_q_size=MAX_Q_SIZE,
            pickle_safe=False,
            verbose=1
        )

    save_model(model, args.output)
