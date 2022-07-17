import cv2
import numpy as np
import pytesseract

#read IMG-0227.jpg
img = cv2.imread('IMG-0227.jpg')
#convert to gray scale
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
#extract text from image
text = pytesseract.image_to_string(out)
print(text)
#show image
cv2.imshow('out', out)
cv2.waitKey(0)


