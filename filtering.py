import cv2
import numpy as np
# grayscale to be suitable for filtering
image = cv2.imread('man.jpg')
image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# median filter with kernel size=5
image_filt=cv2.medianBlur(image_grey, 7)

# display pictures
cv2.imshow('original', image_grey)
cv2.imshow('filtered', image_filt)

#apply laplacian filter to find edges
laplacian = cv2.Laplacian(image_filt, cv2.CV_8U, ksize=5)
cv2.imshow('laplacian', laplacian)

_, laplacian = cv2.threshold(laplacian, 80, 255, cv2.THRESH_BINARY)

laplace_inv = cv2.bitwise_not(laplacian)
cv2.imshow('thresholded laplacian', laplace_inv)

### Bilateral part
downSampled_img = cv2.pyrDown(image)
#applying small nilateral filters on downsampled img]
img_bilateral=downSampled_img
for i in range(7):
    img_bilateral = cv2.bilateralFilter(img_bilateral, d=9, sigmaColor=9, sigmaSpace=7)


upsampled_image = cv2.pyrUp(img_bilateral)
cv2.imshow('bilateral', upsampled_image)

laplace_inv = cv2.resize(laplace_inv, (upsampled_image.shape[1], upsampled_image.shape[0]))
cartoonified_image = cv2.bitwise_and(upsampled_image, upsampled_image, mask=laplace_inv)
cv2.imshow('final', cartoonified_image)

cv2.waitKey(0)
cv2.destroyAllWindows()