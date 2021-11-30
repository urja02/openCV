import numpy as np
import argparse
import cv2
import imutils
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--image",required=True,
help="Path to the image to be scanned")
args=vars(ap.parse_args())
image=cv2.imread(args["image"])

width=480
#print("height is ",int(img.shape[0]))
#print("width is",int(img.shape[1]))
ratio = width/image.shape[1]
height=int(image.shape[0]*ratio)
dim=(width,height)
print("dim",dim)
resized=cv2.resize(image,dim,interpolation=cv2.INTER_AREA)



def rescale(image):
    width=int(image.shape[1]*0.1)
    height=int(image.shape[0]*0.1)
    dimensions=(width,height)
    return cv2.resize(image,dimensions,interpolation=cv2.INTER_AREA)

# resized_image=rescale(image)
# resized=cv2.resize(image,(500,500),interpolation=cv2.INTER_AREA)
gray=cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
blur=cv2.GaussianBlur(gray,(5,5),1)

v = np.median(gray)
lower = int(max(0, (1.0-0.33)*v))
#upper = int(min(255, (1.0 +0.33)*v))
#lower = 0
upper = 0
edged=cv2.Canny(blur,lower,upper)
cv2.imshow("canny`",edged)
cv2.waitKey(0)
kernel=np.ones((5,5))
imgdial=cv2.dilate(edged,kernel,iterations=2)
imgerode=cv2.erode(imgdial,kernel,iterations=1)
orig=imgerode.copy()

cv2.imshow("canny",imgerode)
cv2.waitKey(0)
print("step1 edge detection")

# cnts,hierarchy=cv2.findContours(orig,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
# b=cv2.drawContours(orig,biggest,-1,(70,255,255),3)
# cv2.imshow("contour",b)
# cv2.waitKey(0)
# plt.imshow(b, cmap='gray')
# plt.show()


biggest = np.array([])
maxArea=0
big=[]
cnts,hierarchy=cv2.findContours(orig,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
for c in cnts:
    area = cv2.contourArea(c)
    if area>500:
        cv2.drawContours(resized,cnts,-1,(0,255,0),3)

        cv2.imshow("finalImg",resized)
        cv2.waitKey(0)

        peri = cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c,0.02*peri,True)
        print(len(approx))
        if area>maxArea and len(approx) == 4:
             biggest = approx
             big.append(biggest)
             maxArea = area
        cv2.drawContours(resized,biggest,-1,(0,255,0),3)

print(biggest)
big_sort=sorted(big)
#print(big_sort[0])
cv2.imshow("contour",resized)
cv2.waitKey(0)
plt.imshow(resized, cmap='gray')
plt.show()

def reorder(myPoints):
    myPoints=myPoints.reshape((4,2))
    myPointsNew = np.zeros((4,1,2),np.int32)  #returned from this function

    add = myPoints.sum(axis=1)
    print("add",add)
    a=np.argmin(add)
    print(a)
    myPointsNew[0] = myPoints[np.argmin(add)] #get the index of smallest in the list
    myPointsNew[3] = myPoints[np.argmax(add)]
    print("new points",myPointsNew)

    diff = np.diff(myPoints,axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    print("new points",myPointsNew)

    return myPointsNew
    #the smallest points will be origin and the largest points will be the width and height

def getWarp(img,biggest):
    biggest = reorder(biggest)
    print(biggest.shape)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (width,height))

    #imgCropped = imgOutput[20:imgOutput.shape[0]-]
    return imgOutput

img_warped = getWarp(resized,biggest)
cv2.imshow("warped",img_warped)
cv2.waitKey(0)

# cnts=imutils.grab_contours(cnts)
# cnts=sorted(cnts,key=cv2.contourArea,reverse=True)[:5]
# for c in cnts:
#     peri=cv2.arcLength(c,True)
#     approx=cv2.approxPolyDP(c,0.02*peri,True)
#     print(approx)
#
#     if len(approx)==4:
#         screenCnt= approx
#         break
# print("Step 2:find contours of paper")
# cv2.drawContours(resized,[screenCnt],-1,(0,255,0),2)
#
# def rotate_image(image,angle):
#     image_centre=tuple(np.array(image.shape[1::-1]))

# warped=four_point_transform(orig,screenCnt.reshape(4,2)*ratio)
warped=cv2.cvtColor(img_warped,cv2.COLOR_BGR2GRAY)

threshold=cv2.adaptiveThreshold(warped,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,3)
print("step3:apply perspective transform")
original=cv2.resize(image,(650,500),interpolation=cv2.INTER_AREA)
scanned=cv2.resize(threshold,(height,width),interpolation=cv2.INTER_AREA)
cv2.imshow("original",original)
cv2.imshow("scanned",scanned)
cv2.waitKey(0)
