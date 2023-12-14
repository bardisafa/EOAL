from torchvision.transforms import *

class ToGray(object):
    """
    Convert image from RGB to gray level.
    """
    def __call__(self, img):
        return img.convert('L')