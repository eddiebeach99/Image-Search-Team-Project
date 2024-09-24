import numpy as np
import cv2 as cv
from joblib import load
from sklearn.neighbors import NearestNeighbors
import descriptor_lists as dl
import time

### VARS ###
reference_images_descriptor_list = dl.get_descriptor_list()

### KEYPOINTS AND DESCRIPTORS ###
# find the keypoints and descriptors with ORB
def get_orb_features(img_path):
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    kp, des = orb.detectAndCompute(img,None)
    return kp, des


### Feature Histogram ###
def get_feature_hist(img_path):
    kp, des = get_orb_features(img_path)
    if des is not None:
        feature_hist= build_histogram(des, kmeans)
    return feature_hist

def build_histogram(des, cluster_alg):
    histogram = np.zeros(len(cluster_alg.cluster_centers_))
    cluster_result =  cluster_alg.predict(des)
    for i in cluster_result:
        histogram[i] += 1.0
    return histogram

### Neighbor ####
def get_neighbors(feature_histogram_of_inputimage, feature_hist_list_off_all_pictures):
    # Number of Neighbors
    neighbor = NearestNeighbors(n_neighbors=9)
    # sqlite
    neighbor.fit(feature_hist_list_off_all_pictures)    #fit(x,y) -> Fit the k-nearest neighbors classifier from the training dataset.
    # output
    dist, result = neighbor.kneighbors([feature_histogram_of_inputimage])  #kneighbors([X, n_neighbors, return_distance]) -> Finds the K-neighbors of a point.
    return dist, result

### Code für GUI ###
def main(input_img_path, feature_hist_list_off_all_pictures):
    # Initialize ORB object
    orb = cv.ORB_create()

    # Initialize Clustering
    kmeans = load("cluster_model_400_5")
    feat_hist = get_feature_hist(input_img_path)

    # Calculate nearest neighbors
    dist, result=get_neighbors(feat_hist, feature_hist_list_off_all_pictures)
    x=2

    return dist, result


# ### EXAMPLE CODE ###
# # Initialize ORB object
# start_time1 = time.time()
orb = cv.ORB_create()
#
# # Initialize Clustering
# start_time3 = time.time()
kmeans= load("cluster_model_400_5")
# print("Clustering Dauer --- %s seconds ---" % (time.time() - start_time3))
#
# start_time2 = time.time()
# examp_hist = get_feature_hist('Sonnenblume.png')
# print(examp_hist)
# print("Histogram Dauer --- %s seconds ---" % (time.time() - start_time2))
# print("Code Dauer --- %s seconds ---" % (time.time() - start_time1))
# print("done")



