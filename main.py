import  cv2
import numpy as np

def addNoise(im,noise_percentage):
    vals = len(im.flatten())
    out = np.copy(im)
    num_salt = np.ceil(noise_percentage * vals /100)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in im.shape]
    out[coords] = 255
    return out

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


def morphology(img):
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

        def erosion(size, type, img, I, J, threshold):
            if type == "cross":
                center = int(np.ceil(size / 2) - 1)
                for i in range(1, center + 1):
                    bool = img[I, J + i] == threshold or img[I, J - i] == threshold or img[I + i, J] == threshold or \
                           img[
                               I - i, J] == threshold
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



    img2 = np.zeros((width + 1, height + 1))
    img2[1:width + 1, 1:height + 1] = img
    img3 = np.copy(img2)
# нужна обработка границ, смысла в img3 никакого нет
    for i in range(0, height-1):
        for j in range(0, width-1):
            if (erosion(7, 'square', img2, i, j, 255)):
                img3[i, j] = 255
            else:
                img3[i, j] = 0
    return img3

if __name__ == '__main__':

    img = cv2.imread("rab2.jpg", 0)
    cv2.imshow('Input image', img)

    img = threshold_processing(img,195)

    out1 = addNoise(img,1)
    cv2.imshow("noise1%.jpg", out1)

    out2 = addNoise(img,2 )
    cv2.imshow("noise2%.jpg", out2)

    out5 = addNoise(img,5 )
    cv2.imshow("noise5%.jpg", out5)

    out10 = addNoise(img,10 )
    cv2.imshow("noise10%.jpg", out10)

    out20 = addNoise(img, 20)
    cv2.imshow("noise20%.jpg", out20)

    out50 = addNoise(img,50 )
    cv2.imshow("noise50%.jpg", out50)


    cv2.imshow("er1%.jpg", morphology(img))
    cv2.waitKey(0)
    cv2.destroyAllWindows()