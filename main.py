import  cv2
import numpy as np

def addNoise(im,noise_percentage):
    vals = len(im.flatten())
    out = np.copy(im)
    num_salt = np.ceil(noise_percentage * vals /100)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in im.shape]
    out[coords] = 255
    return out

def erosion(img):

    p = img.shape
    width = p[1]
    height = p[0]
    mask = [[0,255,0],
            [255,255,255],
            [0,255,0]]

    img2 = np.zeros((width+1,height+1 ))
    img2[1:width+1,1:height+1] = img
    img3 =  np.copy(img2)


    for i in range(0,height):
        for j in range(0, width):

            if (img2[i, j + 1] == 255 and img2[i+1, j]==255 and img2[i-1, j]==255 and img2[i, j - 1] ==255):
                img3[i,j] = 255
            else:
                img3[i,j] = 0
    return img3

if __name__ == '__main__':

    img = cv2.imread("rab2Gr.jpg",0)
    cv2.imshow('Input image', img)

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


    cv2.imshow("er1%.jpg", erosion(out1))
    cv2.waitKey(0)
    cv2.destroyAllWindows()