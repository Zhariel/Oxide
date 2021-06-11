from selenium import webdriver
import urllib.request
import requests
import matplotlib.pyplot as plt
from PIL import Image
import PIL
from io import BytesIO
import time
import os
import sys

key1 = "965d8098e98446d580ecb998d19b9670"
key2 = "384d03cf4bc348f9b555205f96c4ac79"
endpoint = "https://api.bing.microsoft.com/v7.0/images/search/"

def run():
    keywords = "maison colombages"
    name = "alsacienne"
    path = 'D:\\PA_Dataset\\alsacienne\\'
    ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36 OPR/52.0.2871.99"
    searchbar_path = '//*[@id="sb_form_q"]'
    mini_path = '//*[@id="mmComponent_images_2_list_1"]/li[1]/div/div/a/div/img'
    collected_urls = []
    length = 150
    offset = 0
    count = 0
    count_saved = 1

    urllib_header = {"User-Agent": ua}
    headers = {"Ocp-Apim-Subscription-Key": key1, "User-Agent" : ua}

    for _ in range(10):
        params = {"q": keywords, "license": "public", "imageType": "photo", "count" : length, "offset" : offset, "nextOffset" : offset + length}
        response = requests.get(endpoint, headers=headers, params=params)
        response.raise_for_status()
        search_results = response.json()
        # thumbnail_urls = [img["thumbnailUrl"] for img in search_results["value"][:16]]

        content = [img["contentUrl"] for img in search_results["value"]]

        print(len(search_results["value"]))
        for s in content:
            count += 1
            print(count)
            print(s)
            if s not in collected_urls:
                try:
                    pass
                    req = urllib.request.Request(s, headers=urllib_header)
                    img = Image.open(urllib.request.urlopen(req))
                    img.save(path + name + str(count_saved) + ".png")
                    count_saved += 1
                    collected_urls.append(s)
                except Exception as e:
                    print(e)
        offset += length
        time.sleep(2)
    #
    file = open(path + "urls.txt", "a")
    file.write('\n'.join(collected_urls))
    file.close()


if __name__ == '__main__':
    run()

