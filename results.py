from matplotlib import patches

from my_CNN_model import load_current_model, test_model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import cv2
import numpy as np

import matplotlib.pyplot as plt


def bounding_box(x_array, y_array):
    min_x = x_array.min()
    min_y = y_array.min()
    max_x = x_array.max()
    max_y = y_array.max()

    return np.array([(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)])


def load_data(test=False, size=3000, test_size=630):
    bb_original = []

    if test:
        # size = test_size
        size = 105

    for i in range(0, size):  # take the landmarks

        txt_path = 'data/train/o_landmarks/train_' + str(i) + '.txt'
        if test:
            txt_path = 'data/test/o_landmarks/test_' + str(i) + '.txt'

        with open(txt_path, 'r') as f:
            lines_list = f.readlines()

            for j in range(3, 58):  # in landmark text files, landmarks start at 3rd line end in 57th
                string = lines_list[j]
                str1, str2 = string.split(' ')
                x_ = float(str1)
                y_ = float(str2)
                if j == 3:  # if first landmark point
                    temp_x = np.array(x_)
                    temp_y = np.array(y_)
                    continue

                # if not first landmark point

                temp_x = np.hstack((temp_x, x_))
                temp_y = np.hstack((temp_y, y_))

        if i == 0:  # if first image's landmarks
            X = np.hstack((temp_x, temp_y))
            X = X[None, :]
            bb = bounding_box(temp_x, temp_y)
            bb_original.append(bb)
            continue

        # if not first image's landmarks
        temp = np.hstack((temp_x, temp_y))
        temp = temp[None, :]
        X = np.vstack((X, temp))
        bb = bounding_box(temp_x, temp_y)
        bb_original.append(bb)

    for i in range(0, size):  # take the images

        img_path = 'data/train/images/train_' + str(i) + '.png'
        if (test):
            img_path = 'data/test/images/test_' + str(i) + '.png'

        img = image.load_img(img_path)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        if (i == 0):  # if first image
            Y = x
            continue
        Y = np.vstack((Y, x))  # if not first image

    return X, Y, bb_original


model = load_current_model('my_model')

X, Y, bb_original = load_data(test=True, test_size=630)     # please make sure your single image consists of only ear
sum_acc = 0

for i in range(0, len(X)):

    original_image = 'data/test/o_images/test_' + str(i) + '.png'
    test_image = 'data/test/images/test_' + str(i) + '.png'
    img_original = cv2.imread(original_image)
    img_test = cv2.imread(test_image)
    img_path = 'data/test/images/test_' + str(i) + '.png'

    temp_y = Y[i]
    temp_y = temp_y[None, :]  # adjust the dimensions for the model
    prediction = model.predict(temp_y)
    for p in range(len(prediction[0])):     # adjust the landmark points for 224x224 image (they were normalized in range 0 to 1)
        prediction[0][p] = int(prediction[0][p] * 224)

    pred_x = prediction[0][0:55]
    pred_y = prediction[0][55:]
    bb = bounding_box(pred_x, pred_y)

    cropped = img_original[int(bb_original[i][0][1]):int(bb_original[i][2][1]), int(bb_original[i][0][0]):int(bb_original[i][1][0])]
    h, w = 244, 244
    dim = (w, h)
    cropped = cv2.resize(cropped, dim, interpolation=cv2.INTER_AREA)

    # Blue color in BGR
    color = np.array((0, 0, 0))

    # Line thickness of 2 px
    thickness = 2

    start_point = bb[0]
    end_ponint = bb[3]

    for point1, point2 in zip(bb, bb[1:]):
        cv2.line(cropped, tuple(point1), tuple(point2), (0,0,255),2)
    cv2.line(cropped, tuple(bb[3]), tuple(bb[0]), (0, 0, 255), 2)
    plt.imshow(cropped)
    plt.show()

    original_area = w * h
    w_pred = bb[2][0] - bb[0][0]
    h_pred = bb[2][1] - bb[0][1]
    prediction_area = w_pred * h_pred

    accuracy = prediction_area / original_area
    sum_acc += accuracy


print("Accuracy: ", sum_acc/105)
