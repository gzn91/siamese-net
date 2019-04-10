import numpy as np
import cv2
import os
import pandas as pd
import pickle
#from tensorflow.keras.applications import ResNet50, VGG16
#from tensorflow.keras.preprocessing import image
#from tensorflow.keras.applications.resnet50 import preprocess_input
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def write2file(path):
    np.random.seed(1)
    valid_ext = ['.jpg', '.png', '.jpeg']
    imgarr = []
    labelarr = []
    namearr = []
    labeldict = {'Koket': 0, 'Dryck': 1, 'Krydda': 2, 'Gront': 2, 'Leverantorsvara': 3, 'Ej': 4}
    for d in sorted(os.listdir(path)):
        d = os.path.join(path, d)
        imgs = os.listdir(d)
        print(d)
        cat = d.split(' ')[-1]
        cat = str(cat)
        for img in imgs:
            label = labeldict[cat]
            name, ext = os.path.splitext(img)
            if ext not in valid_ext:
                continue
            print(name)
            if 'ej' in name:
                label = labeldict['Ej']
            if 'inte' in name:
                label = labeldict['Ej']
            print(label)
            _img = os.path.join(d, img)
            _img = cv2.imread(_img)
            _img = cv2.resize(_img, dsize=(105, 105), interpolation=cv2.INTER_AREA)
            imgarr.append(_img)
            labelarr.append(label)
            namearr.append(name)
    imgarr = np.array(imgarr)
    labelarr = np.array(labelarr)
    namearr = np.array(namearr)
    inds = np.arange(labelarr.size)
    np.random.shuffle(inds)
    imgarr = imgarr[inds]
    labelarr = labelarr[inds]
    namearr = namearr[inds]

    with open('data.pkl', 'wb') as f:
        pickle.dump(imgarr, f)

    with open('labels.pkl', 'wb') as f:
        pickle.dump(labelarr, f)

    with open('names.pkl', 'wb') as f:
        pickle.dump(namearr, f)


def load_data():

    with open('data.pkl', 'rb') as data:
        imgarr = pickle.load(data)

    with open('labels.pkl', 'rb') as labels:
        labelarr = pickle.load(labels)

    with open('names.pkl', 'rb') as names:
        namearr = pickle.load(names)

    def _load_data(ratio=.9):

        x_test = imgarr[labelarr == 4]/255.
        y_test = labelarr[labelarr == 4]
        test_names = namearr[labelarr == 4]
        x, y, names = imgarr[labelarr != 4]/255., labelarr[labelarr != 4], namearr[labelarr != 4]
        for i, name in enumerate(test_names):
            if 'dryck' in name:
                y_test[i] = 1
            elif 'kryddor' in name:
                y_test[i] = 2
            else:
                y_test[i] = 0
            print(y_test[i], name)
        #mean, std = np.mean(x, axis=(0,1,2)), np.std(x, axis=(0,1,2))

        #def normalize(x):
        #    return (x-mean)/std

        N = y.size
        x_train, y_train = x[:int(ratio*N)], y[:int(ratio*N)]
        x_test, y_test = np.concatenate([x_test, x[int(ratio*N):]], axis=0), np.concatenate([y_test, y[int(ratio*N):]], axis=0)
        train_names, test_names = namearr[:int(ratio*N)], np.concatenate([test_names, names[int(ratio*N):]], axis=0)

        return (x_train, y_train), (x_test, y_test), (train_names, test_names)

    return _load_data


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def extract_features():

    res = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

    def _extract_features(img):

        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        features = res.predict(img)
        return features

    return _extract_features


def create_feature_csv(dst='train'):
    df = pd.DataFrame()
    path = './{}/'.format(dst)
    clf = extract_features()
    for img in os.listdir(path):
        _img = image.load_img(path+img, target_size=(224, 224))
        features = clf(_img)
        df[img] = features.ravel()
    df.T.to_csv('{}_features.csv'.format(dst))


def mbgenerator(x, y, mb_size):
    inds = np.arange(x.shape[0])
    np.random.shuffle(inds)
    while True:
        ind = np.random.choice(inds, mb_size, replace=False)
        img1_mb = x[ind]
        label1_mb = y[ind]
        ind = np.random.choice(inds, mb_size, replace=False)
        img2_mb = x[ind]
        label2_mb = y[ind]
        label_mb = np.float32(label1_mb == label2_mb)

        yield img1_mb, img2_mb, label_mb[:, None]


def log_gaussian(x, mean, stddev=1.0):
    return (-0.5 * np.log(2 * np.pi) - tf.log(stddev) - tf.square(x - mean) /
            (2 * tf.square(stddev)))


def mse(diff):
    return tf.sqrt(2 * tf.nn.l2_loss(diff))

if __name__=='__main__':
    # create_feature_csv()
    write2file('./imgs/')
