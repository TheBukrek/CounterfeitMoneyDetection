import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'/Users/ardacakiroglu/opt/anaconda3/pkgs/tesseract-5.0.1-h6be3199_0/bin/tesseract'

#read IMG-0227.jpg
img = cv2.imread('IMG-0227.jpg')
img2 = cv2.imread('IMG-0227-2.jpg')

def backgroundSubtraction(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #extract biggest object in the image
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    #find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #find the biggest contour
    cnt = max(contours, key=cv2.contourArea)
    #draw the biggest contour
    cv2.drawContours(img, [cnt], 0, (0, 255, 0), 3)
    #extract outside of contour
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [cnt], 0, 255, -1)
    #extract outside of contour
    out = cv2.bitwise_and(img, img, mask=mask)
    return out

def SIFTMatching(img1, img2):
    #convert to grayscale
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    #compute SIFT
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    #match SIFT
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    #ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.25* n.distance:
            good.append([m])
    #draw matches
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
    return img3


#show image
#cv2.imshow('out', cv2.resize(backgroundSubtraction(img),(0,0),fx=0.2,fy=0.2))
cv2.imshow('out2', SIFTMatching(backgroundSubtraction(img), img2))
cv2.waitKey(0)


