from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt
from FFT import *

def Experiment2():
  # generateImg(512, 512)
  DFT2D()


def applyFiltering(array1, array2, h, w):
    new = array1
    for rows in range(h):
        for cols in range(w):
            new[rows][cols] = (array1[rows][cols]) * (array2[rows][cols])
            # if(cols % 2 and rows % 2):
            #   new[rows][cols] = 0.0
    return new

def DFT2D():
    image = Image.open("./data_input/newLenna3.png")
    pixels = list(image.getdata())
    w, h = image.size
    newImage = Image.new("L", (w, h))
    pixels = [pixels[i * w:(i + 1) * w] for i in range(h)]
    
    image1 = Image.open("./data_input/SobelPadded5123.png")
    pixels1 = list(image1.getdata())
    w1, h1 = image1.size
    newImage1 = Image.new("L", (w1, h1))
    pixels1 = [pixels1[i * w1:(i + 1) * w1] for i in range(h1)]

    # # N = h x w
    N = (newImage.size[0] + newImage.size[1]) // 4

    print(N, w, h)
    sign = -1

    pixels1[1][1] = 1.0
    pixels1[1][2] = 0.0
    pixels1[1][3] = -1.0

    pixels1[2][1] = 2.0
    pixels1[2][2] = 0.0
    pixels1[2][3] = -2.0

    pixels1[3][1] = 1.0
    pixels1[3][2] = 0.0
    pixels1[3][3] = -1.0

    pixels = normalizeImage(pixels, h, w, N)
    pixels1 = normalizeImage(pixels1, h, w, N)
    
    # Iterate over all the rows and store it into test2D
    pixels = ApplyFFTRow(pixels, w, N, sign)
    pixels1 = ApplyFFTRow(pixels1, w, N, sign)
    
    # Iterate over all the columns and store it into pixels
    pixels = ApplyFFTCol(pixels, h, N, sign)
    pixels1 = ApplyFFTCol(pixels1, h, N, sign)

    pixels = applyFiltering(pixels, pixels1, h, w)

    minVal = 1000000000000000000000000000.0
    maxVal = -101010100000000000000000000.0

    # newImage = LinearScaleValues(newImage, maxVal, minVal, pixels, h, w)
    # newImage1 = LinearScaleValues(newImage1, maxVal, minVal, pixels1, h, w)

    # newImage = LogScaleValues(newImage, maxVal, minVal, pixels, h, w)
    # newImage1 = LogScaleValues(newImage1, maxVal, minVal, pixels1, h, w)

    ################################## Do the reverse ##################################   
    # Iterate over all the columns and store it into pixels
    pixels = ApplyFFTCol(pixels, h, N, 1)

    # Iterate over all the rows and store it into test2D
    pixels = ApplyFFTRow(pixels, w, N, 1)


    # # pixels = changeAmplitudeSpatial(pixels, h, w, N, 1)
    newImage = LinearScaleValues(newImage, maxVal, minVal, pixels, h, w)

    newImage.save("./data_output/lenna_Sobel3.png")
    # newImage1.save("./data_output/SobelFFT.png")

Experiment2()