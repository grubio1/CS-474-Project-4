from PIL import Image
import numpy as np
import cmath
import math
import matplotlib.pyplot as plt
from FFT import *

def Experiment3():

  # Load the image
  image = Image.open("./data_input/E3/motion_blurred_lenna.1_Muller100.png")
  pixels = list(image.getdata())
  width, height = image.size
  newImage = Image.new("L", (width, height))
  pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]


  # # N = height x width
  N = (newImage.size[0] + newImage.size[1]) // 4
  print(N, width, height)
  isign = -1

  newPixels = pixels

  pixels = normalizeImage(pixels, height, width, N)
          
  # Iterate over all the rows and store it into test2D
  pixels = ApplyFFTRow(pixels, width, N, isign)
          
  # Iterate over all the columns and store it into pixels
  pixels = ApplyFFTCol(pixels, height, N, isign)

  minVal = 100000000000000000000000000000000000000000
  maxVal = -101010100000000000000000000000000000000000

  # pixels = blurImage(pixels, height, width, 1, 0.02, 0.02)
  
  pixels = unblurImage_Weiner_AND_Inverse(pixels, height, width, 1, 0.1, 0.1, 0, 0.025)

  # newImage = LinearScaleValues(newImage, maxVal, minVal, np.real(pixels), height, width)
  # newImage = LogScaleValues(newImage, maxVal, minVal, np.real(pixels), height, width)

  ################################## Do the reverse ##################################   
  # Iterate over all the columns and store it into pixels
  pixels = ApplyFFTCol(pixels, height, N, 1)

  # Iterate over all the rows and store it into test2D
  pixels = ApplyFFTRow(pixels, width, N, 1)

  # # pixels = changeAmplitudeSpatial(pixels, height, width, N, 1)
  newImage = LinearScaleValues(newImage, maxVal, minVal, np.real(pixels), height, width)

  newImage.save("./data_output/Experiment 3/Weiner.1_Muller100_K0.025.png")


def unblurImage_Weiner_AND_Inverse(array, height, width, T, a, b, IorW, K):

  for i in range(width):
    for j in range(height):

      u = (i - (width//2)) # Center u 
      v = (j - (height//2)) # Center v
      blur = cmath.pi*((u*a)+(v*b) + 0.00001) # Calculate blur using a = b linear uniform motion
    
      H = ( T / (blur)) * cmath.sin(blur) * cmath.e**(-(cmath.sqrt(-1)) * blur)

      # Hr = np.real(H)

      # Butterworth Lowpass Filter
      radius = 256
      magnitude = 10
      distanceToCenter = math.sqrt((i - (width//2))**2 + (j - (height//2))**2)
      B = 1 / (1 + (distanceToCenter/radius)**(2*magnitude) )
      
      #Inverse Unblur with Butterworth lowpass filter radius
      if(IorW):
        array[i][j] = array[i][j]/H * B
      #Weiner
      else:
        array[i][j] = ( ( (1/H) * ((abs(H*np.conj(H)))/(abs(H*np.conj(H)) + K)))  * array[i][j])

  return array

#Blur and then add Noise
def blurImage(array, height, width, T, a, b):
  for i in range(width):
    for j in range(height):

      u = (i - (width//2)) # Center u 
      v = (j - (height//2)) # Center v
      blur = cmath.pi*((u*a)+(v*b) + 0.00001) # Calculate blur using a = b linear uniform motion
      
      H = ( T / (blur)) * cmath.sin(blur) * cmath.e**(-(cmath.sqrt(-1)) * blur)

      Hr = np.real(H)
      G = box_muller(0.0, 100)

      array[i][j] = (array[i][j] * H ) + G

  return array

# Box muller implemenation for adding Guassian noise
def box_muller(m, s):
  y2 = x1 = x2 = 0.0
  use_last = 0

  if use_last:
    y1 = y2
    use_last = 0
  else:
    while True:
      x1 = 2.0 * np.random.random_sample() - 1.0
      x2 = 2.0 * np.random.random_sample() - 1.0
      w = x1 * x1 + x2 * x2

      if (w <= 1.0):
        break
    w = math.sqrt( (-2.0 * math.log( w )) / w )
    y1 = x1 * w
    y2 = x2 * w
    use_last = 1

  return (m + y1 * s)

Experiment3()