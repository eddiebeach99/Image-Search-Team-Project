import cv2

# Input Parameters :
# imgPath - an absolute or relative Image Path (if relative, then within this directory please)
# n - specify to find the top n highest confidence classes

# Output :
# out_names - an array with the names of the highest confidence n classes


def classify_n_classes(img_path, n):

    # load the image from disk
    image = cv2.imread(img_path)

    # read the ImageNet class names
    with open('classification_classes_ILSVRC2012.txt', 'r') as f:
        image_net_names = f.read().split('\n')
    # final class names (just the first word of the many ImageNet names for one image)
    class_names = [name.split(',')[0] for name in image_net_names]

    # load the neural network model
    model = cv2.dnn.readNet(model='DenseNet_121.caffemodel',
                            config='DenseNet_121.prototxt',
                            framework='Caffe')

    # create blob from image
    blob = cv2.dnn.blobFromImage(image=image, scalefactor=0.01, size=(224, 224),
                                 mean=[104, 117, 123])

    # set the input blob for the neural network
    model.setInput(blob)
    # forward pass image blog through the model
    outputs = model.forward()

    final_outputs = outputs[0]
    # make all the outputs 1D
    # final_outputs = final_outputs.reshape(1024, 1)
    final_outputs = final_outputs.reshape(1000, 1)
    # flatten to make finding top n classes easier within the array
    final_outputs = final_outputs.flatten()
    # get the class labels
    label_ids = (-final_outputs).argsort()[:n]
    # map the max confidence to the class label names
    out_names = []
    for classes in label_ids:

        out_names.append('' + class_names[classes])

    # Will return Strings later, this is temporary
    return out_names
