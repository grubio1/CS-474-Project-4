import matplotlib.pyplot as plt
from FFT import *
from PIL import Image
import numpy as np
import math

def Exp1():
  DFT()

# Does the FFT of a 2D image
def DFT():
   
    image = Image.open("./data_input/boy_noisy.pgm")
    px = list(image.getdata())
    width, height = image.size
    new = Image.new("L", (width, height))
    px = [px[i * width:(i + 1) * width] for i in range(height)]

    # # N = height x width
    N = (new.size[0] + new.size[1]) // 4
    px = normalizeImage(px, height, width, N)
            
    # Iterate over all the rows and store it into test2D
    px = ApplyFFTRow(px, width, N, -1)
            
    # Iterate over all the columns and store it into px
    px = ApplyFFTCol(px, height, N, -1)

    minVal = 1000000000000000000000000000
    maxVal = -101010100000000000000000000

    px = gaussian_band(px, 72, 6, width, height, False)

    # new = LogScaleValues(new, maxVal, minVal, px, height, width)
    # new = LinearScaleValues(new, maxVal, minVal, px, height, width)

    # Reverse 
    # Iterate over all the columns and store it into px
    px = ApplyFFTCol(px, height, N, 1)

    # Iterate over all the rows and store it into test2D
    px = ApplyFFTRow(px, width, N, 1)

    # px = changeAmplitudeSpatial(px, height, width, N, 1)
    new = LinearScaleValues(new, maxVal, minVal, px, height, width)

    new.save("./data_output/New_noisy_boy.png")
    # new1.save("./data_output/GaussianRadius.png")



def gaussian_band(array, C0, W, width, height, PassReject=True):
    # center = 128 # math.sqrt((width//2)**2 + (height//2)**2)
    # newArray = array

    for i in range(width):
      for j in range(height):
        distanceToCenter = math.sqrt((i - (width//2))**2 + (j - (height//2))**2)
        # print(distanceToCenter, i, j, ((i - (width//2))**2), ((j - (height//2))**2))
        # print(distanceToCenter)
        
        # Gaussian Band Reject Filtering
        H = 1 - math.e**(-( (distanceToCenter - C0)**2 / (W)**2 )**2)
       
        # Band Pass = H_bp(u,v) = 1 - H_br
        if PassReject is True:
          H = 1 - H

        array[i][j] = array[i][j] * H
    return array

Exp1()