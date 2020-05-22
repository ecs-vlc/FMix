import numpy as np
from PIL import Image

SIZE = 256


def process(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')
        return np.array(img.resize((SIZE, SIZE), Image.BILINEAR))


def read_bytes(path):
    f = open(path, 'rb')
    return f.read()
