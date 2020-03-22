import builtins

import numpy as np
import twit
import twitc
from PIL import Image
import matplotlib.pyplot as plt
import time


def scale_image():
    """
    Test code and example to scale an image using Twit.
    """
    print("SCALE IMAGE - Pure Python")
    fp = builtins.open(r"K:\twit\dog1.jpg", "rb")
    im = Image.open(fp)
    plt.imshow(im, cmap='Greys_r')
    plt.show()
    pass
    np_im = np.array(im, dtype=np.float64)

    # 225 x 225 x 3
    print("Tensor shape: " + str(np_im.shape))
    start = time.time()

    dest = np.zeros((100, 200, 3), dtype=np.float64)

    twt = twit.compute_twit_multi_dimension(3, np.array(
        [0, np_im.shape[0] - 1, 0, 99, 0, np_im.shape[1] - 1, 0, 199, 0, np_im.shape[2] - 1, 0, 2], dtype=np.int64),
                                            np.array([0.0, 1.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64))
    # t = twitc.unpack_twit_multi_axis(twt)
    N = 3
    pc = 1
    print("Start apply twit loop %d iterations with preclear %d" % (N, pc))
    for i in range(N):
        twit.apply_twit(twt, np_im, dest, 1)
    end = time.time()
    print("Time elapsed: %d sec." % (end - start))
    print("End apply twit loop")
    new_im = Image.fromarray(dest.astype(np.uint8))

    plt.imshow(new_im, cmap='Greys_r')
    plt.show()
    pass


def scale_image_c_code():
    """
    Test code and example to scale an image using Twit.
    """
    print("SCALE IMAGE - C Code")
    im = Image.open(r"K:\twit\dog1.jpg")
    plt.imshow(im, cmap='Greys_r')
    plt.show()
    pass
    np_im = np.array(im, dtype=np.float64)

    # 225 x 225 x 3
    print("Tensor shape: " + str(np_im.shape))
    start = time.time()
    dest = np.zeros((100, 200, 3), dtype=np.float64)

    twitc.set_thread_count(8);

    twt = twitc.compute_twit_multi_dimension(3, np.array(
        [0, np_im.shape[0] - 1, 0, 99, 0, np_im.shape[1] - 1, 0, 199, 0, np_im.shape[2] - 1, 0, 2], dtype=np.int64),
                                             np.array([0.0, 1.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64))
    # t = twitc.unpack_twit_multi_axis(twt)
    N = 10000
    pc = 1
    print("Start apply twit loop %d iterations with preclear %d" % (N, pc))
    for i in range(N):
        twitc.apply_twit(twt, np_im, dest, pc)
    end = time.time()
    print("Time elapsed: %d sec.  %f milliseconds per twit" % (end - start, float(end - start) * 1000.0 / float(N)))
    print("End apply twit loop")
    new_im = Image.fromarray(dest.astype(np.uint8))

    plt.imshow(new_im, cmap='Greys_r')
    plt.show()
    pass


if __name__ == '__main__':
    print("Scale Image with twit sample program.")

    scale_image()

    scale_image_c_code()
