import cv2 as cv
import numpy as np
import sqlite3
import feature_extraction as fe
import database_numpy_modifications as dnm

# Input Parameters: img - an absolute or relative Image Path (if relative, then within this directory please)
# classes - list of classes that fit best to the input image (Calculated in classify_image_classes)
# Output:
# List of tuple of the 9 nearest image neighbours, tuples consist of histogram correlation and URL

# Functions required to read arrays in sqlite3
sqlite3.register_adapter(np.ndarray, dnm.adapt_array)

# Converts TEXT to np.array when selecting
sqlite3.register_converter("array", dnm.convert_array)


# Calculates Histogram from an Image file, hist_type is either 0,1,2 (0 = blue, 1 = green, 2 = red)


def calc_hist(img, hist_type):
    pic = cv.imread(img)
    bgr_planes = cv.split(pic)

    hist_size = 256

    hist_range = (0, 256)
    # the upper boundary is exclusive
    accumulate = False
    return cv.calcHist(bgr_planes, [hist_type], None, [hist_size], hist_range, accumulate=accumulate)


def feature_comparison(img, class_list, method):
    class1 = class_list[0]
    class2 = class_list[1]
    class3 = class_list[2]

    model = fe.prep_model()
    vector = fe.extract_vis_features(img, model)

    conn = sqlite3.connect('ImSeDatabaseFINAL.db', detect_types=sqlite3.PARSE_DECLTYPES)
    c = conn.cursor()

    c.execute("""SELECT * FROM imgFeatures WHERE classifiedAs == ? OR classifiedAs == ?\
        OR classifiedAs == ?""", (class1, class2, class3))
    result = []
    for row in c.fetchall():
        result.append((np.linalg.norm(vector - row[7]), row[3], row[4], row[5], row[2]))

    result.sort(key=lambda x: x[0])

    histogram_order = []
    b_hist = calc_hist(img, 0)
    g_hist = calc_hist(img, 1)
    r_hist = calc_hist(img, 2)
    j = 0
    for i in result:
        hist_corr = cv.compareHist(b_hist, i[1], cv.HISTCMP_CORREL) \
                 + cv.compareHist(g_hist, i[2], cv.HISTCMP_CORREL) \
                 + cv.compareHist(r_hist, i[3], cv.HISTCMP_CORREL)
        histogram_order.append((hist_corr, i[4], i[0]))
        j += 1
        if j == 9:
            break

    if method == "hist":
        histogram_order.sort(key=lambda x: x[0])
        return list(reversed(histogram_order))
    else:
        histogram_order.sort(key=lambda x: x[2])
        return list(histogram_order)


# print(feature_comparison('Sonnenblume.png', ['seashore', 'magazine', 'web site'], "idunno"))