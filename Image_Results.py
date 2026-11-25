import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def Image_Results():
    I = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    for n in range(2):
        Images = np.load('Image.npy', allow_pickle=True)
        GT = np.load('GroundTruth.npy', allow_pickle=True)
        SynImage = np.load('Synthetic_Images.npy', allow_pickle=True)
        Seg = np.load('Segmented.npy', allow_pickle=True)
        PROPOSED = np.load('Annotation_Images.npy', allow_pickle=True)
        for i in range(len(I[n])):
            plt.subplot(2, 3, 1)
            plt.title('Original')
            plt.imshow(Images[I[n][i]])
            plt.subplot(2, 3, 2)
            plt.title('GroundTruth')
            plt.imshow(GT[I[n][i]])
            plt.subplot(2, 3, 3)
            plt.title('Synthetic Image')
            plt.imshow(SynImage[I[n][i]])

            plt.subplot(2, 3, 4)
            plt.title('Segmented')
            plt.imshow(Seg[I[n][i]])
            plt.subplot(2, 3, 5)
            plt.title('Proposed')
            plt.imshow(PROPOSED[I[n][i]])
            plt.tight_layout()
            plt.show()


def Sample_Images():
    for n in range(2):
        Orig = np.load('Image_' + str(n + 1) + '.npy', allow_pickle=True)
        ind = [20, 40, 60, 80, 88, 90]
        fig, ax = plt.subplots(2, 3)
        plt.suptitle("Sample Images from Dataset " + str(n + 1))
        plt.subplot(2, 3, 1)
        plt.title('Image-1')
        plt.imshow(Orig[ind[0]])
        plt.subplot(2, 3, 2)
        plt.title('Image-2')
        plt.imshow(Orig[ind[1]])
        plt.subplot(2, 3, 3)
        plt.title('Image-3')
        plt.imshow(Orig[ind[2]])
        plt.subplot(2, 3, 4)
        plt.title('Image-4')
        plt.imshow(Orig[ind[3]])
        plt.subplot(2, 3, 5)
        plt.title('Image-5')
        plt.imshow(Orig[ind[4]])
        plt.subplot(2, 3, 6)
        plt.title('Image-6')
        plt.imshow(Orig[ind[5]])
        plt.show()


if __name__ == '__main__':
    Image_Results()
    Sample_Images()
