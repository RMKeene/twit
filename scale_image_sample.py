import numpy as np
import twit
from PIL import Image
import matplotlib.pyplot as plt

def scale_image():
    """
    Test code and example to scale an image using Twit.
    """
    im = Image.open("dog1.jpg")
    np_im = np.array(im, dtype=np.float64)
    # 225 x 225 x 3
    print("Tensor shape: " + str(np_im.shape))

    dest = np.zeros((100, 200, 3))
    twit.tensor_transfer(np_im, dest, preclear=False)
    new_im = Image.fromarray(dest, mode='RGB')

    plt.imshow(im, cmap='Greys_r')
    plt.show()
    pass
    plt.imshow(new_im, cmap='Greys_r')
    plt.show()
    pass

 
if __name__ == '__main__':
    print("Scale Image with twit sample program.")
    scale_image()
