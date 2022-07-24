import cv2
from cv2 import bitwise_not
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'/Users/ardacakiroglu/opt/anaconda3/envs/bil/bin/tesseract'

#read IMG-0227.jpg
img = cv2.imread('IMG-0247.jpg')
img2 = cv2.imread('IMG-0250.jpg')

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
    left = 5000
    right = 0
    bottom = 5000
    top = 0
    #find the lefttop , rightbottom, leftbottom, righttop
    for i in range(0, len(cnt)):
        if cnt[i][0][0] < left:
            left = cnt[i][0][0]
        if cnt[i][0][0] > right:
            right = cnt[i][0][0]
        if cnt[i][0][1] < bottom:
            bottom = cnt[i][0][1]
        if cnt[i][0][1] > top:
            top = cnt[i][0][1]
    cornersArr = np.float32([[[left, bottom]] , [[right, bottom]] , [[left, top]] , [[right, top]]])

    return out,cornersArr

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

def wrapFlat(img, cornersArr):
    #wrap the image
    width = 1280
    height = 576
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix  = cv2.getPerspectiveTransform(cornersArr, pts2)
    img = cv2.warpPerspective(img, matrix, (width, height))
    return img

def getText(img):
    #get text from image
    text = pytesseract.image_to_string(img, lang='tur')
    return text

def align_Images(img, template, maxFeatures=2000, keepPercent=0.5,debug = False):
    imageGray = cv2. cvtColor(img, cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(maxFeatures)
    (kp1, des1) = orb.detectAndCompute(imageGray, None)
    (kp2, des2) = orb.detectAndCompute(templateGray, None)
    method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matcher = cv2.DescriptorMatcher_create(method)
    matches = matcher.match(des1, des2, None)
    
    matches = sorted(matches, key=lambda x:x.distance)

    keep  = int(len(matches) * keepPercent)
    matches = matches[:keep]
    if debug:
        img3 = cv2.drawMatches(imageGray, kp1, templateGray, kp2, matches, None, flags=2)
        cv2.namedWindow('img3', cv2.WINDOW_NORMAL)
        cv2.imshow("img3", img3)
        cv2.waitKey(0)
    
    points1 = np.zeros((len(matches), 2), dtype="float")
    points2 = np.zeros((len(matches), 2), dtype="float")

    for (i, match) in enumerate(matches):
        points1[i] = kp1[match.queryIdx].pt
        points2[i] = kp2[match.trainIdx].pt

    (h, mask) = cv2.findHomography(points1, points2, method = cv2.RANSAC)
    height, width = template.shape[:2]
    aligned = cv2.warpPerspective(img, h, (width, height))
    return aligned

def cropImage(img, x, y, w, h):
    #crop image
    img = img[y:y+h, x:x+w]
    return img

def binarize(img, threshold):
    #binarize image
    ret, thresh = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return thresh

#https://pyimagesearch.com/2020/08/31/image-alignment-and-registration-with-opencv/
#y2000 - 2100
#x1100 - 1700

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
a = align_Images(img2, img)
b = cropImage(a, 1100, 2000, 1700-1100, 2100-2000)
#convert to grayscale
b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
c = binarize(b, 50)
c = cv2.bitwise_not(c)
d = cv2.dilate(c, np.ones((5,5), np.uint8))
print(getText(d))
cv2.imshow('image',c)
cv2.waitKey(0)


