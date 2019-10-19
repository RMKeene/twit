import numpy as np
import twit
import twitc
from PIL import Image
import matplotlib.pyplot as plt

def scale_image():
    """
    Test code and example to scale an image using Twit.
    """

    im = Image.open("dog1.jpg")
    plt.imshow(im, cmap='Greys_r')
    plt.show()
    pass
    np_im = np.array(im)
 
    # 225 x 225 x 3
    print("Tensor shape: " + str(np_im.shape))

    dest = np.zeros((100, 200, 3))
    twit.tensor_transfer(np_im, dest, preclear=False, weight_range=(0.0, 1.0), weight_axis=1)
    new_im = Image.fromarray(dest.astype(np.uint8))

    plt.imshow(new_im, cmap='Greys_r')
    plt.show()
    pass

def scale_image_C_code():
    """
    Test code and example to scale an image using Twit.
    """

    im = Image.open(r"K:\twit\dog1.jpg")
    plt.imshow(im, cmap='Greys_r')
    plt.show()
    pass
    np_im = np.array(im, dtype=np.float64)
 
    # 225 x 225 x 3
    print("Tensor shape: " + str(np_im.shape))

    dest = np.zeros((100, 200, 3), dtype=np.float64)

    # twitc.make_and_apply_twit(3, np.array([0, np_im.shape[0] - 1, 0, np_im.shape[1] - 1, 0, np_im.shape[2] - 1, 0, 99, 0, 199, 0, 2], dtype=np.int64),  np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64), np_im, dest, 0)
    twt = twitc.compute_twit_multi_dimension(3, np.array([0, np_im.shape[0] - 1, 0, 99, 0, np_im.shape[1] - 1, 0, 199, 0, np_im.shape[2] - 1, 0, 2], dtype=np.int64),  np.array([0.0, 1.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64))
    # t = twitc.unpack_twit_multi_axis(twt)
    N = 1
    pc = 1
    print("Start apply twit loop %d iterations with preclear %d" % (N, pc))
    for i in range(N):
        twitc.apply_twit(twt, np_im, dest, 1)
    print("End apply twit loop")
    new_im = Image.fromarray(dest.astype(np.uint8))

    plt.imshow(new_im, cmap='Greys_r')
    plt.show()
    pass

 
if __name__ == '__main__':
    print("Scale Image with twit sample program.")
    scale_image_C_code()
