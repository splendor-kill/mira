import os
from PIL import Image
import numpy as np
from scipy.misc import imresize, imsave
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from scipy.ndimage import measurements


def compute_feature(im):
    norm_im = imresize(im, (30, 30))
    norm_im = norm_im[3:-3, 3:-3]
    return norm_im.flatten()


def load_ocr_data(path):
    imlist = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    labels = [int(imfile.split('/')[-1][0]) for imfile in imlist]
    features = []
    for imname in imlist:
        im = np.array(Image.open(imname).convert('L'))
        features.append(compute_feature(im))
    return np.array(features), labels


def find_sudoku_edges(im, axis=0):
    """ Finds the cell edges for an aligned sudoku image. """

    trim = 1 * (im < 128)
    s = trim.sum(axis=axis)
    # find center of strongest lines
    s_labels, s_nbr = measurements.label(s > (0.5 * max(s)))
    m = measurements.center_of_mass(s, s_labels, range(1, s_nbr + 1))
    x = [int(x[0]) for x in m]
    # if only the strong lines are detected add lines in between
    if len(x) == 4:
        dx = np.diff(x)
        x = [x[0], x[0] + dx[0] / 3, x[0] + 2 * dx[0] / 3,
             x[1], x[1] + dx[1] / 3, x[1] + 2 * dx[1] / 3,
             x[2], x[2] + dx[2] / 3, x[2] + 2 * dx[2] / 3, x[3]]
    if len(x) == 10:
        return np.array(x, dtype=np.int)
    else:
        raise RuntimeError('Edges not detected.')


def ocr(clf, im):
    im = np.array(im.convert('L'))
    # find the cell edges
    x = find_sudoku_edges(im, axis=0)
    y = find_sudoku_edges(im, axis=1)
    # crop cells and classify
    crops = []
    for col in range(9):
        for row in range(9):
            l, r = y[col], y[col + 1]
            t, b = x[row], x[row + 1]
            crop = im[l:r, t:b]
            global axarr
            if axarr is not None:
                axarr[col, row].axes.get_xaxis().set_visible(False)
                axarr[col, row].axes.get_yaxis().set_visible(False)
                # imsave('/home/splendor/Pictures/cell_s%d%d.jpg' % (col, row), imresize(crop, (32, 32)))
                # imsave('/home/splendor/Pictures/cell_m%d%d.jpg' % (col, row), imresize(crop, (64, 64)))
                # imsave('/home/splendor/Pictures/cell_l%d%d.jpg' % (col, row), imresize(crop, (96, 96)))
                axarr[col, row].imshow(imresize(crop, (30, 30)), cmap=plt.get_cmap('gray'))
            crops.append(compute_feature(crop))
    res = clf.predict(crops)
    res_im = np.array(res).reshape(9, 9)
    return res_im


def train(features, labels):
    clf = svm.SVC(kernel='linear', decision_function_shape='ovr')
    clf.fit(features, labels)
    return clf


if __name__ == '__main__':
    dat_dir = '/home/splendor/learn/PCV/data/sudoku_images/'
    features, labels = load_ocr_data(os.path.join(dat_dir, 'ocr_data/training/'))
    test_features, test_labels = load_ocr_data(os.path.join(dat_dir, 'ocr_data/testing/'))
    clf = train(features, labels)

    preds = clf.predict(test_features)
    print('Accuracy:', accuracy_score(test_labels, preds))
    # print('Confusion Matrix:\n', confusion_matrix(test_labels, preds))
    # print('Classification Report:\n', classification_report(test_labels, preds))

    # imname = 'sudokus/sudoku18.JPG'
    # vername = 'sudokus/sudoku18.sud'
    # verify = np.loadtxt(os.path.join(dat_dir, vername), dtype=np.int).reshape(9, 9)
    # res_im = ocr(clf, Image.open(os.path.join(dat_dir, imname)))
    # print('Result:')
    # print(res_im)
    # print('Ground Truth:')
    # print(verify)
    # print('correct numbers:', np.sum(verify == res_im))

    import gui_utils
    import time
    import matplotlib.pyplot as plt

    time.sleep(5)
    x, y, w, h = gui_utils.get_xwin_dims('Sudoku')
    # print(x, y, w, h)
    im = gui_utils.capture_window(x, y, w, h, 45)
    plt.imshow(im, cmap=plt.get_cmap('gray'))

    fig, axarr = plt.subplots(9, 9)
    g_idx = 0
    tm = ocr(clf, im)
    print(tm)
    plt.show()
