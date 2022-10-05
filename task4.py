#Programmer: Henry Kutlik
# Object Thresholding CITED USING: https://aihints.com/how-to-draw-bounding-box-in-opencv-python/
# Dominant color CITED USING: https://code.likeagirl.io/finding-dominant-colour-on-an-image-b4e075f98097
# rgba(78,103,100,255)
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def find_histogram(clt):
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist
def plot_colors2(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar

cap = cv.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, img = cap.read()

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_green = np.array([78,104,99])
    upper_green = np.array([0,255,0])
    mask = cv.inRange(hsv, lower_green, upper_green)
    # Threshold the HSV image to get only blue colors

    blur = cv.GaussianBlur(img,(5,5),0)
    gray_image = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
    thresh = cv.threshold(gray_image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    contours = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnt = contours[0] if len(contours) == 2 else contours[1]
    x,y,w,h = cv.boundingRect(cnt[0]) #Creates a box only for the first object
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    box = img[y:y+h,x:x+w] #Gets an image of the central box

    cv.imshow('Bounding Box', img)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

img = cv.cvtColor(box, cv.COLOR_BGR2RGB)

img = img.reshape((img.shape[0] * img.shape[1],3)) #represent as row*column,channel number
clt = KMeans(n_clusters=3) #cluster number
clt.fit(img)

hist = find_histogram(clt)
bar = plot_colors2(hist, clt.cluster_centers_)

plt.axis("off")
plt.imshow(bar)
plt.show()
# When everything done, release the capture
cap.release()