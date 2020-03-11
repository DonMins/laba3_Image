import  cv2
import numpy as np

def addNoise(im,noise_percentage):
    vals = len(im.flatten())
    p = img.shape
    width = p[1]
    height = p[0]
    listAllCoord = []
    for i in range(height):
        for j in range(width):
            listAllCoord.append([i,j])
    out = np.copy(im)
    num_salt = int(np.ceil(noise_percentage * vals /100))

    for i in range(num_salt):
        # coord = listAllCoord.pop(np.random.randint(0, len(listAllCoord) - 1))
        coord = [np.random.randint(0, i - 1) for i in im.shape]

        if img[coord[0],coord[1]] == 255:
            out[coord[0], coord[1]] = 0
        else :
            out[coord[0],coord[1]] = 255
    return out


def noise_percentage(img,img2):
    p = img.shape
    width = p[1]
    height = p[0]
    countWhite1 = 0
    countBlack1 = 0
    countWhite2 = 0
    countBlack2 = 0

    for i in range(height):
        for j in range(width):
            if (img[i, j] == 255):
                countWhite1 += 1
            else:
                countBlack1+=1
            if (img2[i, j] == 255):
                countWhite2 += 1
            else:
                countBlack2+=1


    print("Отношение : " ,
          100*((abs(countWhite1 - countWhite2) + abs(countBlack1 - countBlack2))/ (width * height))
)



def threshold_processing(img, threshold):
    p = img.shape
    width = p[1]
    height = p[0]

    for i in range(height):
        for j in range(width):
            if (img[i, j] > threshold):
                img[i, j] = 255
            else:
                img[i, j] = 0

    return img


def morphology(img, type, sizeMask, typeMask):
    p = img.shape
    width = p[1]
    height = p[0]
# нужно убрать говнокод и сделать обработку границ , при изменение размера маски, в конце он будет выходить за границы картинки
    def erosion(size, type, img, I, J, threshold):
        if type == "cross":
            center = int (np.ceil(size / 2) - 1)
            for i in range(1, center + 1):
                bool = img[I, J + i] == threshold and img[I, J - i] == threshold and img[I + i, J] == threshold and img[
                    I - i, J] == threshold
                if (bool == False):
                    return False
            return True
        if type == "square":
            center = int(np.ceil(size / 2) - 1)
            for i in range(1, center + 1):
                for j in range(1, center + 1):
                    bool = img[I, J + j] == threshold and img[I, J - j] == threshold and img[I + i, J] == threshold and img[
                    I - i, J] == threshold and img[I - i, J - j] == threshold and img[I - i, J + j] == threshold and\
                       img[I + i, J - j] == threshold and img[I + i, J + j] == threshold
                    if (bool == False):
                        return False
            return True

    def dilation(size, type, img, I, J, threshold):
        if type == "cross":
            center = int(np.ceil(size / 2) - 1)
            for i in range(1, center + 1):
                bool = img[I, J + i] == threshold or img[I, J - i] == threshold or img[I + i, J] == threshold or \
                       img[ I - i, J] == threshold
                if (bool == False):
                    return False
            return True
        if type == "square":
            center = int(np.ceil(size / 2) - 1)
            for i in range(1, center + 1):
                for j in range(1, center + 1):
                    bool = img[I, J + j] == threshold or img[I, J - j] == threshold or img[
                        I + i, J] == threshold or img[
                               I - i, J] == threshold or img[I - i, J - j] == threshold or img[
                               I - i, J + j] == threshold or \
                           img[I + i, J - j] == threshold or img[I + i, J + j] == threshold
                    if (bool == False):
                        return False
            return True

    center = int(np.ceil(sizeMask / 2) - 1)
    img2 = np.zeros(( height + center*2, width + center*2,))
    img2[center:height+center,center:width+center] = img
    img3 = np.copy(img2)
# нужна обработка границ, смысла в img3 никакого нет
    for i in range(0, height):
        for j in range(0, width):
            if type == "erosion":
                if (erosion(sizeMask, typeMask, img2, i, j, 255)):
                    img3[i, j] = 255
                else:
                    img3[i, j] = 0

            if type == "dilation":
                if (dilation(sizeMask, typeMask, img2, i, j, 255)):
                    img3[i, j] = 255
                else:
                    img3[i, j] = 0

    return img3[center:height + center,  center:width + center]

if __name__ == '__main__':

    img = cv2.imread("rab2.jpg", 0)
    cv2.imshow('Input image', img)

    img = threshold_processing(img,195)
    cv2.imshow('Input image', img)

    out1 = addNoise(img,1)
    cv2.imshow("noise1%.jpg", out1)
    #
    # out2 = addNoise(img,2 )
    # cv2.imshow("noise2%.jpg", out2)
    #
    # out5 = addNoise(img,5 )
    # cv2.imshow("noise5%.jpg", out5)
    #
    # out10 = addNoise(img,10 )
    # cv2.imshow("noise10%.jpg", out10)
    #
    # out20 = addNoise(img, 20)
    # cv2.imshow("noise20%.jpg", out20)
    #
    out50 = addNoise(img,50 )
    cv2.imshow("noise50%.jpg", out50)

    av  = morphology(out50, "erosion", 3, "cross")
    #
    cv2.imshow("dilation1%.jpg", morphology(av, "erosion", 5, "cross"))

    autopsy = morphology(out50, "dilation", 3, "square")
    cv2.imshow("closing%.jpg", morphology(autopsy, "erosion", 3, "square"))

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(out50 ,cv2.MORPH_CLOSE, kernel)
    dilation = cv2.dilate(img, kernel, iterations=1)

    cv2.imshow("closingDefault%.jpg", closing)


    noise_percentage(img,out50)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #
    #--------------------Контура---------------------------

    # img = cv2.imread("rec.jpg",0)
    # img = threshold_processing(img,195)
    # cv2.imshow('Input image', img)
    #
    # cv2.imshow("dilation1.jpg", morphology(img, "dilation1", 7, "square"))
    # out1 = addNoise(img, 50)
    # cv2.imshow("noise50%.jpg", out1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

