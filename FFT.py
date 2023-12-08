from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt

def doubleImageSize():
    image = Image.open("./data_input/lenna.pgm")
    pixels1 = list(image.getdata())
    w1, h1 = image.size
    pixels1 = [pixels1[i * w1:(i + 1) * w1] for i in range(h1)]
    h1 = h1 * 2
    w1 = w1 * 2
    

    genPixels = np.zeros((h1, w1), dtype=int)
    # Store pixels into image to check
    newImage1 = Image.new("L", (w1, h1))

    for i in range(w1//2):
        for j in range(h1//2):
            val = pixels1[j][i]
            genPixels[j][i] = val
            newImage1.putpixel((i, j), val)

    N1 = (newImage1.size[0] + newImage1.size[1]) // 4
    newImage1.save("./data_input/newLenna.png")


# Iterate over all the rows and store it into pixels
def ApplyFFTRow(pixels, w, N, sign):
    for i in range(w):
        if(sign == -1): # Forward
            test = np.array(pixels)[i, 0:w] # set test to the pixel row of i
            test = np.insert(test, 0, 0.0)  # Add 0 in the front since we don't use data[0]
            test = fft(test, N, sign) # fft on the current row
            test = np.delete(test, 0) # Delete data[0]
            test = changeAmplitudeFrequency(test, N) # Move the array over by N
        elif(sign == 1): # Inverse
            test = np.array(pixels)[i, 0:w] # set test to the pixel row of i
            test = changeAmplitudeFrequency(test, N) # Move the array over by N
            test = np.insert(test, 0, 0.0)  # Add 0 to front since we don't use data[0]
            test = fft(test/N, N, 1) # Inverse fft on the current Row
            test = np.delete(test, 0) # Delete data[0]
        else:
            print("Error isign can only be 1 or -1, exiting.")
            break

        # Now place test into the current row of pixels
        for j in range(w):  
            pixels[i][j] = test[j]

    return pixels

# Iterate over all the columns and store it into pixels
def ApplyFFTCol(pixels, h, N, isign):
    for i in range(h):
        if(isign == -1): # Forward
            test = np.array(pixels)[0:h, i]
            test = np.insert(test, 0, 0.0)
            test = fft(test, N, isign) 
            test = np.delete(test, 0)
            test = changeAmplitudeFrequency(test, N)
        elif(isign == 1): # Inverse
            test = np.array(pixels)[0:h, i]
            test = changeAmplitudeFrequency(test, N)
            test = np.insert(test, 0, 0.0)
            test = fft(test/N, N, isign) 
            test = np.delete(test, 0)
        else:
            print("Error isign can only be 1 or -1, exiting.")
            break

        for j in range(h):
            pixels[j][i] = test[j]
    return pixels


# Fast Fourier Transform for Discrete 1-D Array
def fft(data, nn, isign):
    n = mmax = m = j = istep = i = 0.0
    wtemp = wr = wpr = wpi = wi = theta = 0.0
    tempr = tempi = float(0)

    n = nn << 1
    j = 1
    for i in range(1, n, 2):
        if(j > i):
            data = SWAP(data, j, i)
            data = SWAP(data, j+1, i+1)
        m = n >> 1
        while (m >= 2 and j > m):
            j -= m
            m = m >> 1
        j += m
    mmax = 2
    while (n > mmax):
        istep = mmax << 1
        theta = isign * (6.28318530717959/mmax)
        wtemp = math.sin(0.5 * theta)
        wpr = -2.0 * wtemp * wtemp
        wpi = math.sin(theta)
        wr = 1.0
        wi = 0.0
        for m in range(1, mmax, 2): #start at 1 because 0 isn't used
            # m += 2
            for i in range(m, n + 1, istep):
                j = i + mmax
                tempr = (wr * data[j]) - (wi * data[j+1])
                tempi = wr * data[j+1] + wi * data[j]
                data[j] = data[i] - tempr 
                data[j+1] = data[i+1] - tempi
                data[i] += tempr
                data[i+1] += tempi
            wtemp = wr
            wr = wr * wpr - wi * wpi + wr
            wi = wi * wpr + wtemp * wpi + wi
                
        

        mmax = istep
    return data

# Scale the image Linearly through min max values 
def LinearScaleValues(img, maxVal, minVal, arrayImage, h, w):
    
    # find Min Max
    for rows in range(h):
        for cols in range(w):
            # find Min Max Values
            if (arrayImage[rows][cols] > maxVal):
                maxVal = arrayImage[rows][cols]
            if (arrayImage[rows][cols] < minVal):
                minVal = arrayImage[rows][cols]

    print(minVal, maxVal)
    scalar = 255 / maxVal
    scalar = round(scalar, 10)
    for rows in range(h):
        for cols in range(w):
            val = arrayImage[rows][cols]
            val = int(val * scalar)
            img.putpixel((cols, rows), val)
            arrayImage[rows][cols] = val
    
    return img

#Log Scale the images and then bring the values back in range from 0 - 255
def LogScaleValues(img, max, min, arrayImage, h, w):

    # Gets the Magnitude of the image
    arrayImage = getMagnitude(arrayImage, h, w)
    for rows in range(h):
        for cols in range(w):
            val = arrayImage[rows][cols]
            val = float(math.log(1 + (abs(val))))
            image.putpixel((cols, rows), int(val))
            arrayImage[rows][cols] = val
    image = LinearScaleValues(image, max, min, arrayImage, h, w)        

    return image

# Swap array[data1] with array[data2]
def SWAP(arr, data1, data2):
    temp1 = arr[data1]
    arr[data1] = arr[data2]
    arr[data2] = temp1
    return arr

# Translate to center of the frequency in the spatial domain f(x)
def normalizeImage(arr, h, w, N):
    for i in range(0, h, 1):
        for j in range(0, w, 1):
            v = float(arr[i][j] / N)
            arr[i][j] = float(v)
    return arr

# Translate to the center of the frequency in the Frequency domain F(u)
def changeAmplitudeFrequency(arr, n):
    arr = np.roll(arr, n)
    return arr

# get the Magnitude of the Fourier transform For Fun
def getMagnitude(arr, h, w):
    for rows in range(h):
        for columns in range(w):
            v = arr[rows][columns]
            v = abs(v)
            arr[rows][columns] = v
    return arr


def generateImg(h, w):
    # Generate a 512 x 512, place a 32x32 white square at the center
    genImg32x32 = Image.new("L", size=(h, w))
    genPixels = np.zeros((h, w), dtype=int)
    
    # Store pixels into image to check
    for i in range(h):
        for j in range(w):
            val = int(genPixels[i][j])
            genImg32x32.putpixel((i, j), val)
    genImg32x32.save("./data_input/SobelPadded512.png")