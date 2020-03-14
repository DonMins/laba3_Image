import operator
import  cv2
import numpy as np

class cv3:
    def __init__(self):
        pass

    @classmethod
    def erodeIsTrue(cls,img, mask, cenH, cenW):
        p = img.shape
        width = p[1]
        height = p[0]
        flag = True
        for i in range(height):
            for j in range(width):
                    if  img[i, j] != mask[i, j]:
                        return False
        return True

    @classmethod
    def dilateIsTrue(cls,img, mask, cenH, cenW):
        p = img.shape
        width = p[1]
        height = p[0]
        for i in range(height):
            for j in range(width):
                if img[i, j] == mask[i, j]:
                    return True
        return False

    @classmethod
    def preparation(cls,img, mask):
        p = img.shape
        pM = mask.shape
        width = p[1]
        height = p[0]

        widthM = pM[1]
        heightM = pM[0]

        centerHeightM = int(np.ceil(heightM / 2) - 1)
        centerwidthM = int(np.ceil(widthM / 2) - 1)
        img2 = np.zeros((height + centerHeightM * 2, width + centerwidthM * 2), np.uint8)
        img2[centerHeightM:height + centerHeightM, centerwidthM:width + centerwidthM] = img

        for i in range(heightM):
            for j in range(widthM):
                if mask[i, j] == 1:
                    mask[i, j] = 255
        return width,height,centerHeightM,centerwidthM,img2, mask,



    @staticmethod
    def dilate(img, mask):
        width,height,centerHeightM,centerwidthM,img2, mask = cv3.preparation(img,mask)
        img3 = np.copy(img2)

        for i in range(centerHeightM, height + centerHeightM):
            for j in range(centerwidthM, width + centerwidthM):
                chunk = img2[i - centerHeightM: i + centerHeightM + 1, j - centerwidthM: j + centerwidthM + 1]
                if (cv3.dilateIsTrue(chunk, mask, centerHeightM,centerwidthM)):
                    img3[i, j] = 255
                else:
                    img3[i, j] = 0
        return img3[centerHeightM:height + centerHeightM, centerwidthM:width + centerwidthM]


    @staticmethod
    def erode(img, mask):
        width, height, centerHeightM, centerwidthM, img2, mask = cv3.preparation(img, mask)
        img3 = np.copy(img2)

        for i in range(centerHeightM, height + centerHeightM):
            for j in range(centerwidthM, width + centerwidthM):
                chunk = img2[i - centerHeightM: i + centerHeightM + 1, j - centerwidthM: j + centerwidthM + 1]
                if (cv3.erodeIsTrue(chunk, mask, centerHeightM, centerwidthM)):
                    img3[i, j] = 255
                else:
                    img3[i, j] = 0
        return img3[centerHeightM:height + centerHeightM, centerwidthM:width + centerwidthM]

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
        return dilation ^ img

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
    out1 = cv3.addNoise(img, 1)
    cv2.imshow(' out1', out1)

    kernel = np.ones((3,3),np.uint8)
    my = cv3.erode(out1,kernel)

    cv2.imshow('my', my)

    st = cv2.erode(out1, kernel)
    cv2.imshow('st', st)

    cv2.imshow('xor', st^my)
    cv2.waitKey(0)

