import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import orb_features as of
import os, os.path
from PIL import Image

imgs = []
path = r"test_img"
valid_images = [".jpg", ".png"]
for img in os.listdir(path):
    imgs.append(Image.open(os.path.join(path,img)))

imgs_feature_hist = []

path = r"test_img"
valid_images = [".jpg", ".png"]
orb = cv.ORB_create()
for img in os.listdir(path):
    ext = os.path.splitext(img)[1]
    if ext.lower() not in valid_images:
        continue
    imgs_feature_hist.append(of.get_feature_hist(os.path.join(path,img)))


print(of.main("Sonnenblume.png", imgs_feature_hist))

x = 2