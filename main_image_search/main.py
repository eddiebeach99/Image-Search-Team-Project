import classify_image_classes as cic
import image_search as ims
import time

# Input Parameters:
# img_path - an absolute or relative Image Path (if relative, then within this directory please)
# Output:
# List of tuple of the 9 nearest image neighbours, tuples consist of histogram correlation and URL


def main(img_path, method):
    start_time = time.time()
    classes = cic.classify_n_classes(img_path, 3)

    if method == "orb":
        result = ims.feature_comparison(img_path, classes, "hist")
    elif method == "feature":
        result = ims.feature_comparison(img_path, classes, method)
    else:
        result = ims.feature_comparison(img_path, classes, "hist")

    print("Images found in --- %s seconds ---" % (time.time() - start_time))
    return result


# Example image search
# main(r"C:\Users\edwar\Documents\main\test1.jpg")
