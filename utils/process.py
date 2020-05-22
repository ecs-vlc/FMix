from PIL import Image

SIZE = 256


def process(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')
        return img.resize((SIZE, SIZE), Image.BILINEAR)
