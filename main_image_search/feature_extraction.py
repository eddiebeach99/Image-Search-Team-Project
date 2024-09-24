import cv2
import numpy as np

def prep_model():

    # read the ImageNet class names
    with open('classification_classes_ILSVRC2012.txt', 'r') as f:
        image_net_names = f.read().split('\n')

    # final class names (just the first word of the many ImageNet names for one image)
    class_names = [name.split(',')[0] for name in image_net_names]

    # load the neural network model
    model = cv2.dnn.readNet(model='DenseNet_121.caffemodel',
                            config='DenseNet_121.prototxt',
                            framework='Caffe')
    return model

def extract_vis_features(img_path, model):
    # load the image from disk
    image = cv2.imread(img_path)

    # create blob from image
    blob = cv2.dnn.blobFromImage(image=image, scalefactor=0.01, size=(224, 224),
                                 mean=[104, 117, 123])

    # set the input blob for the neural network
    model.setInput(blob)

    # forward pass image blog through the model until second last layer
    outputs = model.forward("pool5")

    final_outputs = outputs[0]

    # make all the outputs 1D
    final_outputs = final_outputs.reshape(1024, 1)
    final_outputs = np.array(final_outputs)
    return final_outputs

# EXAMPLE of execution:
# model = prep_model()
# featuresImg = extract_vis_features('../../input/image_1.jpg', model)
