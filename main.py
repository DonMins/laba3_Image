import operator
import  cv2
import numpy as np

class cv3:
    def __init__(self):
        pass

    @classmethod
    def erodeIsTrue(self,type, img, I, J , center):
        threshold = 255
        if type == 'cross':
            for i in range(1, center + 1):
                bool = img[I, J + i] == threshold and img[I, J - i] == threshold and img[I + i, J] == threshold and \
                       img[I - i, J] == threshold
                if (bool == False):
                    return False
            return True

        if type == "square":
            if center == 1:
                bool = img[I, J + 1] == threshold and  img[I, J - 1] == threshold and img[
                    I + 1, J] == threshold and img[
                           I - 1, J] == threshold or img[I - 1, J - 1] == threshold and img[
                           I - 1, J + 1] == threshold and \
                       img[I + 1, J - 1] == threshold and img[I + 1, J + 1] == threshold
                if (bool == False):
                    return False
                return True

            if center == 2:
                bool = \
                    img[I, J + 1] == threshold \
                    and img[I, J - 1] == threshold \
                    and img[I + 1, J] == threshold \
                    and img[I - 1, J] == threshold \
                    and img[I - 1, J - 1] == threshold \
                    and img[I - 1, J + 1] == threshold \
                    and img[I + 1, J - 1] == threshold \
                    and img[I + 1, J + 1] == threshold \
                    and img[I, J + 2] == threshold \
                    and img[I, J - 2] == threshold \
                    and img[I + 2, J] == threshold \
                    and img[I - 2, J] == threshold \
                    and img[I - 1, J - 2] == threshold \
                    and img[I - 1, J + 2] == threshold \
                    and img[I + 2, J - 2] == threshold \
                    and img[I + 2, J + 2] == threshold \
                    and img[I + 2, J - 1] == threshold \
                    and img[I + 2, J + 1] == threshold \
                    and img[I + 1, J + 2] == threshold \
                    and img[I + 1, J - 2] == threshold \
                    and img[I - 2, J + 2] == threshold \
                    and img[I - 2, J - 2] == threshold \
                    and img[I - 2, J + 1] == threshold \
                    and img[I - 2, J - 1] == threshold

                if (bool == False):
                    return False
                return True

    @classmethod
    def dilateIsTrue(self,type, img, I, J , center):
        threshold = 255
        if type == 'cross':
            for i in range(1, center + 1):
                bool = img[I, J + i] == threshold or img[I, J - i] == threshold or img[I + i, J] == threshold or \
                       img[I - i, J] == threshold
                if (bool == False):
                    return False
            return True

        if type == "square":
            if center == 1:
                bool = img[I, J + 1] == threshold or img[I, J - 1] == threshold or img[
                    I + 1, J] == threshold or img[
                           I - 1, J] == threshold or img[I - 1, J - 1] == threshold or img[
                           I - 1, J + 1] == threshold or\
                       img[I + 1, J - 1] == threshold or img[I + 1, J + 1] == threshold

                if (bool == False):
                    return False
                return True

            if center == 2:
                bool = \
                    img[I, J + 1] == threshold \
                    or img[I, J - 1] == threshold \
                    or img[I + 1, J] == threshold \
                    or img[I - 1, J] == threshold \
                    or img[I - 1, J - 1] == threshold \
                    or img[I - 1, J + 1] == threshold \
                    or img[I + 1, J - 1] == threshold \
                    or img[I + 1, J + 1] == threshold \
                    or img[I, J + 2] == threshold \
                    or img[I, J - 2] == threshold \
                    or img[I + 2, J] == threshold \
                    or img[I - 2, J] == threshold \
                    or img[I - 1, J - 2] == threshold \
                    or img[I - 1, J + 2] == threshold \
                    or img[I + 2, J - 2] == threshold \
                    or img[I + 2, J + 2] == threshold \
                    or img[I + 2, J - 1] == threshold \
                    or img[I + 2, J + 1] == threshold \
                    or img[I + 1, J + 2] == threshold \
                    or img[I + 1, J - 2] == threshold \
                    or img[I - 2, J + 2] == threshold \
                    or img[I - 2, J - 2] == threshold \
                    or img[I - 2, J + 1] == threshold \
                    or img[I - 2, J - 1] == threshold

                if (bool == False):
                    return False
                return True

    @staticmethod
    def dilate(img, mask):
        p = img.shape
        width = p[1]
        height = p[0]
        n = len(mask) - 1

        center = int(np.ceil(len(mask) / 2) - 1)
        img2 = np.zeros((height + center * 2, width + center * 2))
        img2[center:height + center, center:width + center] = img
        img3 = np.copy(img2)


        if (mask[0][0] == 0 and mask[0][n] == 0 and mask[n][0] == 0 and  mask[n][n] == 0):
            type = 'cross'
        else:
            type = 'square'

        for i in range(n + 1):
            for j in range(n + 1):
                if mask[i][j] == 1:
                    mask[i][j] = 255


        for i in range(center, height + center):
            for j in range(center, width + center):
                if (cv3.dilateIsTrue(type, img2, i, j, center)):
                    img3[i, j] = 255
                else:
                    img3[i, j] = 0

        return  img3[center:height + center, center:width + center]

    @staticmethod
    def erode(img, mask):
        p = img.shape
        width = p[1]
        height = p[0]
        n = len(mask) - 1

        center = int(np.ceil(len(mask) / 2) - 1)
        img2 = np.zeros((height + center * 2, width + center * 2))
        img2[center:height + center, center:width + center] = img
        img3 = np.copy(img2)

        if (mask[0][0] == 0 and mask[0][n] == 0 and mask[n][0] == 0 and mask[n][n] == 0):
            type = 'cross'
        else:
            type = 'square'

        for i in range(n + 1):
            for j in range(n + 1):
                if mask[i][j] == 1:
                    mask[i][j] = 255

        for i in range(center, height + center):
            for j in range(center, width + center):
                if (cv3.dilateIsTrue(type, img2, i, j, center)):
                    img3[i, j] = 255
                else:
                    img3[i, j] = 0

        return img3

    @staticmethod
    def open(img, mask):
        img2 = cv3.erode(img, mask)
        return cv3.dilate(img2, mask)

    @staticmethod
    def close(img, mask):
        img2 = cv3.dilate(img, mask)
        return cv3.erode(img2, mask)


    @staticmethod
    def addNoise(im, noise_percentage):
        vals = len(im.flatten())
        p = img.shape
        width = p[1]
        height = p[0]
        listAllCoord = []
        for i in range(height):
            for j in range(width):
                listAllCoord.append([i, j])
        out = np.copy(im)
        num_salt = int(np.ceil(noise_percentage * vals / 100))

        for i in range(num_salt):
            coord = [np.random.randint(0, i - 1) for i in im.shape]

            if img[coord[0], coord[1]] == 255:
                out[coord[0], coord[1]] = 0
            else:
                out[coord[0], coord[1]] = 255

        return out

    @staticmethod
    def noise_percentage(img, img2):
        p = img.shape
        width = p[1]
        height = p[0]
        img3 = img
        countNoise = 0
        for i in range(height):
            for j in range(width):
                if (operator.xor(img2[i, j], img[i, j])) == 255:
                    countNoise += 1

        print("Отношение : ", countNoise * 100 / (width * height))
        return countNoise * 100 / (width * height)

    @staticmethod
    def contour(img, mask):
        dilation = cv3.dilate(img, mask).astype(np.uint8)
        p1 = img.shape[0]
        p2 = img.shape[1]
        img2 = img
        for i in range(p1):
            for j in range(p2):
                img2[i, j] = operator.xor(dilation[i, j], img[i, j])
        return img2

    @staticmethod
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

if __name__ == '__main__':

    img = cv2.imread("rab2.jpg", 0)

    img = cv3.threshold_processing(img, 195)
    out1 = cv3.addNoise(img, 99)
    cv2.imshow(' out1', out1)

    kernel = np.ones((5, 5), np.uint8)

    im = cv3.contour(img,kernel)
    cv2.imshow('im', im)
    cv2.waitKey(0)


