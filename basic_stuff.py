import cv2 as cv
import numpy as np

img=cv.imread("/home/parsa/Codes/opencv projects/python/Resources/Photos/cat.jpg")
cv.imshow("real",img)
# basic filters
# gray=cv.cvtColor(img,cv.COLOR_RGB2BGRA)
# cv.imshow("gray",gray)

# gblur=cv.GaussianBlur(img,(7,3),0)
# cv.imshow("gblur",gblur)

# mblur=cv.medianBlur(img,3)
# cv.imshow("mblur",mblur)

# bblur=cv.bilateralFilter(img,20,40,25)
# cv.imshow("bblur",bblur)


#edge detection

# canny=cv.Canny(gblur,threshold1=120,threshold2=140)
# cv.imshow("canny",canny)

img2=cv.imread("/home/parsa/Codes/opencv projects/python/Resources/Photos/cats.jpg")
# gray2=cv.cvtColor(img2,cv.COLOR_RGB2GRAY)
# lap = cv.Laplacian(gray,ddepth=cv.CV_32F)
# lap = np.uint8(np.absolute(lap))

# cv.imshow("lap",lap)

# sobelx=cv.Sobel(gray2,ddepth=cv.CV_32F,dx=1,dy=0,ksize=3)
# sobely=cv.Sobel(gray2,ddepth=cv.CV_32F,dx=0,dy=1,ksize=3)
# abs_sobelx=np.uint8(np.absolute(sobelx))
# abs_sobely=np.uint8(np.absolute(sobely))

# sobel=cv.bitwise_or(abs_sobelx,abs_sobely)
# cv.imshow("sobel",sobel)

# sobel=np.sqrt(sobelx**2 + sobely**2)
# sobel=cv.convertScaleAbs(sobel)
# cv.imshow("sobel2",sobel)

# #resizing
# resized=cv.resize(img2,dsize=(400,400),interpolation=cv.INTER_CUBIC)
# cv.imshow("resized",resized)

# rotated=cv.rotate(img2,rotateCode=cv.ROTATE_180)
# cv.imshow("rotated",rotated)

blank=np.zeros(img2.shape[:2],dtype=np.uint8)
circle=cv.circle(blank.copy(),center=(blank.shape[1]//2,blank.shape[0]//2),
                 radius=100,color=255,thickness=-1)
cv.imshow("circle",circle)

rectangle=cv.rectangle(blank.copy(),pt1=(blank.shape[1]//2-50,blank.shape[0]//2-50),
                       pt2=(blank.shape[1]//2+100,blank.shape[0]//2+100),color=255,thickness=-1)
cv.imshow("rectangle",rectangle)

or_shape=cv.bitwise_or(circle,rectangle)

cv.imshow("bit_or",or_shape)

masked_img=cv.bitwise_and(img2,img2,mask=or_shape)
cv.imshow("masked",masked_img)
cv.waitKey(0)