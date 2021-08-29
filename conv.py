import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
#from io import BytesIO

def conv(image,im_filter):
    height = image.shape[0]
    width  = image.shape[1]

    im=np.zeros((height - len(im_filter)+1,width- len(im_filter)+1))

    for row in range(len(im)):
        for col in range(len(im[0])):
            for i in range(len(im_filter)):
                for j in range(len(im_filter[0])):
                    im[row,col]+=image[row+i,col+j]*im_filter[i][j]

    im[im>255]=255
    im[im<0]=0

    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(image,cmap=cm.Greys_r)
    plt.subplot(1,2,2)
    plt.imshow(im,cmap=cm.Greys_r)
    plt.show()


img = np.array(Image.open('gng.jpg'))
image_grayscale = np.mean(img, axis=2, dtype=np.uint)
sobel_y = [[-1, 0, 1],
 [-2, 0, 2],
 [-1, 0, 1]]
conv(image_grayscale, sobel_y)