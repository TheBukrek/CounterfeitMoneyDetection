import cv2
from cv2 import threshold
import numpy as np
from paddleocr import PaddleOCR, draw_ocr
import os
import re

os.environ['KMP_DUPLICATE_LIB_OK']='True'
ocr_model = PaddleOCR(lang='en')


#read IMG-0227.jpg
img = cv2.imread('./5lira/5on.jpg')
img3 = cv2.imread('./test/IMG-0243.jpg')

SerialNumber = [["G010204073","20"],["C407591522","50"],["D331377364","50"],["C512326409","50"],["E066829516","5"],["D162200281","10"],["D137366422","200"],["C063576040","200"],["B290643115","200"],["D974027460","100"],["E117263172","200"],["F057129366","100"],["E022393618","5"],["D875492359","100"]]
AverageColors = [[202.56503923, 211.121212, 218.24664536],[196.40794344,204.51941937,211.44919905],   #5 on arka
[197.09770052,206.99769764,221.83599514],[190.53522185,199.3998723,222.00563065],  #10
[190.23110401,206.23049607,219.89297753],[179.4851876,198.05615184,217.17329562], #50
[200.46706143,207.18465437,208.16439179],[186.17739601,188.89303057,186.71363359], #100
[198.70505261,205.32944652,213.69402226],[174.66094381,188.52701888,205.65357185]] #200

regex = "[A-Z][0-9]{9}"
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

def SIFTMatching(img1, img2,threshold=0.25, debug = False):
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
        if m.distance < threshold* n.distance:
            good.append([m])
    #draw matches
    if debug:
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
        cv2.namedWindow('img3', cv2.WINDOW_NORMAL)
        cv2.imshow("img3", img3)
        cv2.waitKey(0)
    return  good

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
    img = img[y:y+h, x:x+w]
    return img

def binarize(img, threshold):
    ret, thresh = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return thresh

def switch(i):
    if i == 0:
        return "5"
    elif i == 1:
        return "10"
    elif i == 2:
        return "20"
    elif i == 3:
        return "50"
    elif i == 4:
        return "100"
    elif i == 5:
        return "200"
    else:
        return -1

def getValue(img):
    banknot5 = cv2.imread('./SayisalDegerler/5.jpg')
    banknot10 = cv2.imread('./SayisalDegerler/10.jpg')
    banknot20 = cv2.imread('./SayisalDegerler/20.jpg')
    banknot50 = cv2.imread('./SayisalDegerler/50.jpg')
    banknot100 = cv2.imread('./SayisalDegerler/100.jpg')
    banknot200 = cv2.imread('./SayisalDegerler/200.jpg')
    a = [banknot5, banknot10, banknot20, banknot50, banknot100, banknot200]
    for i in range(len(a)):
        if (len(SIFTMatching(img, a[i],threshold= 0.6)) > 10):
            return i
    return -1

def compareText(img):
    beslira = []
    onlira = []
    yirmilira = []
    ellilira = []
    yuzlira = []
    ikiyuzlira = []


def compareFeatures(img, banknot, onArkaVar):
    boolArr = [False, False, False, False]
    feature1 = cv2.imread('./'+banknot+'lira/')
    feature2 = cv2.imread('./'+banknot+'lira/')
    feature3 = cv2.imread('./'+banknot+'lira/')
    feature4=0
    if(onArkaVar == 1):
        feature4 = cv2.imread('./'+banknot+'lira/')
    else:    
        feature4 = cv2.imread('./'+banknot+'lira/')
    
    if (len(SIFTMatching(img, feature1,threshold= 0.6)) > 10):
        boolArr[0] = True
    if (len(SIFTMatching(img, feature2,threshold= 0.6)) > 10):
        boolArr[1] = True
    if (len(SIFTMatching(img, feature3,threshold= 0.6)) > 10):
        boolArr[2] = True
    if (len(SIFTMatching(img, feature4,threshold= 0.6)) > 10):
        boolArr[3] = True

        
    return boolArr[0]&boolArr[1]&boolArr[2]&boolArr[3]

def getSerialNumber(img):
    a =  ocr_model.ocr(img)
    for res in a:
        x = re.search(regex,res[1][0]) 
        if(x):
            return x.group()
        
def getAverage(img):
    average_color_row = np.average(img, axis=0)
    average_color = np.average(average_color_row, axis=0)
    return average_color
#https://pyimagesearch.com/2020/08/31/image-alignment-and-registration-with-opencv/

def onArka(img):
    if True:
        return 0
    return 1

def isItReal(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    valueAsIndex = getValue(img)
    valueAsString = switch(valueAsIndex)
    
    onArkaVar = onArka(img)
    averagePixelValues = getAverage(img)
    if ((averagePixelValues[0] > (AverageColors[valueAsIndex+onArkaVar][0]-10) or averagePixelValues[0] < (AverageColors[valueAsIndex+onArkaVar][0]+10)) and (averagePixelValues[1] > (AverageColors[valueAsIndex+onArkaVar][1]-10) or averagePixelValues[1] < (AverageColors[valueAsIndex+onArkaVar][1]+10)) and (averagePixelValues[2] > (AverageColors[valueAsIndex+onArkaVar][2]-10) or averagePixelValues[2] < (AverageColors[valueAsIndex+onArkaVar][2]+10))): #color check
        if(onArka):
            leftSerialNumber = getSerialNumber(img)
            rightSerialNumber = getSerialNumber(img)
            if(leftSerialNumber==rightSerialNumber):#first one should be left bottom second one should be right top
                for i in range(len(SerialNumber)):
                    if ((SerialNumber[i][0] == leftSerialNumber) and (SerialNumber[i][1] == valueAsString)):
                        break
                    else:
                        return False
            else:
                return False
        return compareFeatures(img, valueAsString, onArkaVar)
    else:
        return False

    
    


# print(getTextTesseract(img))
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# a = align_Images(img3, img, keepPercent=0.25)
print( getSerialNumber(img))

# cv2.waitKey(0)
# cv2.imshow('image',a)
# cv2.waitKey(0)


