import cv2 as cv
from sklearn.cluster import KMeans
import descriptor_lists as dl
from joblib import dump

### VARS ###
reference_images_descriptor_list = dl.get_descriptor_list()

#Code for Clustering and saving the model
orb = cv.ORB_create()
kmeans= KMeans(n_clusters =800)
model =kmeans.fit(reference_images_descriptor_list)
dump(model, "cluster_model_400_5")
print("clustering finally finished")