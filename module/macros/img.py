from PIL import Image
import os

class Img():
    def __init__(self):
        pass

    def resized_dataset(self, path, nb, resolution, categories=[1, 1, 1]):
        for folder in os.walk("D:\PA_Dataset"):
            pass