import  cv2
import numpy as np

def addNoise(im,noise_percentage):

    vals = len(im.flatten())

    amount = 0.1
    out = np.copy(im)
    num_salt = np.ceil(noise_percentage * vals /100)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in im.shape]
    out[coords] = 155
    return out

if __name__ == '__main__':

    img = cv2.imread("rab2Gr.jpg")
    cv2.imshow('Input image', img)


    out = addNoise(img,10)
    cv2.imshow("noise.jpg", out)

    cv2.waitKey(0)
    cv2.destroyAllWindows()