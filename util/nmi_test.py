from sklearn import metrics as mr
import imageio
import numpy as np
def main():
    img1 = imageio.imread('../data/market1501/query/0030_c1s1_002551_03.jpg')
    img2 = imageio.imread('../data/market1501/query/0030_c1s1_002576_01.jpg')

    img2 = np.resize(img2, (img1.shape[0], img1.shape[1], img1.shape[2]))

    img1 = np.reshape(img1, -1)
    img2 = np.reshape(img2, -1)
    print(img2.shape)
    print(img1.shape)
    mutual_infor = mr.mutual_info_score(img1, img2)

    print(mutual_infor)

if __name__ == '__main__':
    main()