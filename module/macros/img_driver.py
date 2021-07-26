from PIL import Image
import numpy as np

import os

def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(
        ((img_width - crop_width) // 2,
         (img_height - crop_height) // 2,
         (img_width + crop_width) // 2,
         (img_height + crop_height) // 2))

def crop_max_square(pil_img):
    return crop_center(pil_img, min(pil_img.size), min(pil_img.size))

class Img_Driver():
    def __init__(self):
        pass

    def remove_bar(self, path="D:\PA_Dataset\\raw\\input"):
        new_path = "D:\PA_Dataset\\raw\\removed\\"
        for f in os.scandir(path):
            img_path = path + "\\" + f.name
            img = Image.open(img_path)
            box = (0, 0, img.width, img.height-20)
            cropped = img.crop(box)
            cropped.save(new_path + "removed" + f.name)

    def flip(self, path="D:\PA_Dataset\\raw\\input"):
        path = "D:\PA_Dataset\\modern"
        new_path = "D:\PA_Dataset\\raw\\flipped\\"
        for f in os.scandir(path):
            img_path = path + "\\" + f.name
            img = Image.open(img_path)
            enhanced = img.transpose(Image.FLIP_LEFT_RIGHT)
            enhanced.save(new_path + "flipped" + f.name)

    def resized_dataset(self, px, max=10, rgb=False, path="D:\PA_Dataset", categories=["asian", "modern", "palladian"]):
        for f in os.scandir(path):
            if f.name not in categories:
                continue

            i_count = 0
            for i in os.scandir(path + "\\" + f.name):
                i_count += 1
                if i_count > max:
                    break

                color = "rgb" if rgb else "grey"
                raw_path = path + "\\" + f.name + "\\" + i.name
                new_path = path + "\\" + "output" + "\\" + str(px) + "_" + color + "\\" + f.name + "_" + str(i_count) + ".png"
                print(raw_path)
                print(new_path)
                print()
                img = None
                if rgb:
                    img = Image.open(raw_path)
                else:
                    img = Image.open(raw_path).convert('L')

                im_new = crop_max_square(img)
                size = px, px
                im_new.thumbnail(size)
                im_new.save(new_path)

                # box = (px, px, px, px)
                # print(box)
                # cropped = raw_img.crop(box)
                # cropped.save(new_path)

            # print(f.name)

    def pixels_rgb(self, px, rgb=False, divider=255, test=False):
        t = "output_test" if test else "output"
        color = "rgb" if rgb else "grey"
        delete_path = "D:\PA_Dataset\\delete.txt"
        path = "D:\PA_Dataset" + "\\" + t + "\\" + str(px) + "_" + color
        # lis = os.listdir(path)
        # number_files = len(lis)

        images_pix = []
        for f in os.scandir(path):
            img = Image.open(path + "\\" + f.name)
            data = img.getdata()
            if rgb:
                single_img = []
                for d in data:
                    try:
                        single_img += [float(x/divider) for x in d]
                    except:
                        continue
                if len(single_img) == 0:
                    with open(delete_path, "a") as file:
                        file.write(f.name + "\n")
                    os.remove(path + "\\" + f.name)
                images_pix.append(single_img)
            else:
                images_pix.append([float(x/divider) for x in img.getdata()])

        return images_pix

    def define_y_keyword(self, keyword, px, rgb=False, test=False):
        t = "output_test" if test else "output"
        color = "rgb" if rgb else "grey"
        path = "D:\PA_Dataset" + "\\" + t + "\\" + str(px) + "_" + color
        entries = {"asian": [1, 0, 0],
                   "modern": [0, 1, 0],
                   "palladian": [0, 0, 1]}

        img_y = []
        for f in os.scandir(path):
            if f.name.startswith(keyword):
                img_y.append(entries[keyword])

        return img_y


    def define_y(self, px, rgb=False, test=False):
        t = "output_test" if test else "output"
        color = "rgb" if rgb else "grey"
        path = "D:\PA_Dataset" + "\\" + t + "\\" + str(px) + "_" + color

        count = 0
        img_y = []
        for f in os.scandir(path):
            if f.name.startswith('asian'):
                img_y.append([1, 0, 0])
                # print(f"{f.name} {img_y[count]}")
            elif f.name.startswith('modern'):
                img_y.append([0, 1, 0])
                # print(f"{f.name} {img_y[count]}")
            elif f.name.startswith('palladian'):
                img_y.append([0, 0, 1])
                # print(f"{f.name} {img_y[count]}")
            count += 1

        return img_y

    def define_y_linear(self, keyword, px, rgb=False, test=False):
        t = "output_test" if test else "output"
        color = "rgb" if rgb else "grey"
        path = "D:\PA_Dataset" + "\\" + t + "\\" + str(px) + "_" + color

        count = 0
        img_y = []
        for f in os.scandir(path):
            if f.name.startswith(keyword):
                img_y.append(1.0)
            else:
                img_y.append(-1.0)
            count += 1

        return img_y






