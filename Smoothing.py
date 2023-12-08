from PIL import Image
import numpy as np
import math

gaussianMask7 = np.array([[1, 1, 2, 2,  2, 1, 1],
                          [1, 2, 2, 4,  2, 2, 1],
                          [2, 2, 4, 8,  4, 2, 2],                
                          [2, 4, 8, 16, 8, 4, 2],
                          [2, 2, 4, 8,  4, 2, 2],
                          [1, 2, 2, 4,  2, 2, 1],
                          [1, 1, 2, 2,  2, 1, 1]], np.float32)

gaussianMask15 = np.array([[2, 2, 3,  4,  5,  5,  6,  6,  6,  5,  5,  4,  3,  2, 2],
                           [2, 3, 4,  5,  7,  7,  8,  8,  8,  7,  7,  5,  4,  3, 2],
                           [3, 4, 6,  7,  9,  10, 10, 11, 10, 10, 9,  7,  6,  4, 3],                
                           [4, 5, 7,  9,  10, 12, 13, 13, 13, 12, 10, 9,  7,  5, 4],
                           [5, 7, 9,  11, 13, 14, 15, 16, 15, 14, 13, 11, 9,  7, 5],
                           [5, 7, 10, 12, 14, 16, 17, 18, 17, 16, 14, 12, 10, 7, 5],
                           [6, 8, 10, 13, 15, 17, 19, 19, 19, 17, 15, 13, 10, 8, 6],
                           [6, 8, 11, 13, 16, 18, 19, 20, 19, 18, 16, 13, 11, 8, 6],
                           [6, 8, 10, 13, 15, 17, 19, 19, 19, 17, 15, 13, 10, 8, 6],
                           [5, 7, 10, 12, 14, 16, 17, 18, 17, 16, 14, 12, 10, 7, 5],
                           [5, 7, 9,  11, 13, 14, 15, 16, 15, 14, 13, 11, 9,  7, 5],
                           [4, 5, 7,  9,  10, 12, 13, 13, 13, 12, 10, 9,  7,  5, 4],
                           [3, 4, 6,  7,  9,  10, 10, 11, 10, 10, 9,  7,  6,  4, 3],
                           [2, 3, 4,  5,  7,  7,  8,  8,  8,  7,  7,  5,  4,  3, 2],
                           [2, 2, 3,  4,  5,  5,  6,  6,  6,  5,  5,  4,  3,  2, 2]], np.float32)

averagingMask7 = np.array([[1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1],                
                           [1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1]], np.float32)
                
averagingMask15 = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],                
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], np.float32)

def smoothing(mask_size, image):
    averagedMask = Average(gaussianMask15)
    mapSmoothing(averagedMask, image)
    
def Average(mask):
    # Find the sum of the elements in the mask
    v = 0
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            v += mask[i][j]
        
    # Find the sum of the elements in the mask
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            avg = 1/v
            mask[i][j] = avg

    return mask

def mapSmoothing(mask, image):
    # Initialize new Image to store the Average pixels
    newImage = Image.new("L", (image.size[0], image.size[1]))

    # new array to store the summation values
    arrayImage = np.array(newImage, np.int32)

    max = 0
    min = 4546465165
    # Always 256 for either anyways
    for rows in range(image.size[1]):
        for columns in range(image.size[0]):
            
            value = int(_mask(mask, image, columns, rows))
            arrayImage[rows][columns] = value
    
            if (arrayImage[rows][columns] > max):
                max = arrayImage[rows][columns]
            if (arrayImage[rows][columns] < min):
                min = arrayImage[rows][columns]

    newImage = ScaleValues(newImage, max, min, arrayImage)

    newImage.save("./data_output/boy_noisy_15x15G.pgm")

def _mask(mask, image, imageCols, imageRows):
    sum = 0

    for maskRows in range(-(mask.shape[1] // 2), mask.shape[1] // 2):
        for maskCols in range(-(mask.shape[0] // 2), mask.shape[0] // 2):

            newCol = maskCols + imageCols
            newRow = maskRows + imageRows
            checkBottom = image.size[1] - (newRow)
            checkRight = image.size[0] - (newCol)
            
            colMask = maskCols + (mask.shape[0] // 2)
            rowMask = maskRows + (mask.shape[1] // 2)

            # Ensure all bounds that are negative
            if(newCol >= 0 and newRow >= 0 and checkBottom >= 0 and checkRight >= 0):
                try:
                    F = image.getpixel((newCol, newRow))
                    W = mask[rowMask][colMask]
                    sum = sum + (F * W)
                except:
                    sum += 0
            else:
                sum += 0

    return sum

def ScaleValues(img, maxVal, minVal, arrayImage):
    scalar = 255 / maxVal
    scalar = round(scalar, 2)

    for rows in range(arrayImage.shape[1]):
        for columns in range(arrayImage.shape[0]):
            val = arrayImage[rows][columns]
            # print(val)
            val = int(val * scalar)
            # print(val)
            img.putpixel((columns, rows), val)
    
    return img

maskImage = Image.open("./data_input/boy_noisy.pgm")
smoothing(maskImage.size, maskImage)