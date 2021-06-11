import base64
from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pyautogui as pg
import bs4
from PIL import Image
import time
import os
import sys

keyword = "building"
chromeOptions = Options()
# chromeOptions.add_argument("--headless")
PATH = "C:\Program Files (x86)\chromedriver.exe"
driver = webdriver.Chrome(executable_path=PATH, options=chromeOptions)

def img_url(nb):
    return f'// *[ @ id = "islrg"] / div[1] / div[{nb}] / a[1] / div[1] / img'

def save_macro(count):
    pg.moveTo(1500, 500)
    pg.click(button='right')
    pg.moveTo(1520, 715)
    pg.click(button='left')
    pg.moveTo(100, 400)
    time.sleep(0.1)
    pg.click(button='left')
    time.sleep(0.1)
    pg.moveTo(400, 935)
    pg.click(button='left')
    time.sleep(1)
    pg.write("modern"+str(count)+".png")
    pg.press('enter')

def run():
    consent_path = '/html/body/div[2]/div[1]/div[3]/span/div/div/div[3]/button[2]/div'
    searchbar_path = '/html/body/div[2]/div[3]/div[2]/form/div[1]/div[1]/div[1]/div/div[2]/input'
    body_path = '/html/body/div[2]/c-wiz/div[4]/div[1]/div/div/div/div/div[1]/span/div[1]/div[1]'
    img_after = '//*[@id="Sva75c"]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div/div[2]/a/img'
    show_more = '//*[@id="islmp"]/div/div/div/div/div[3]/div[2]/input'

    driver.get("https://www.google.fr/imghp?hl=en&authuser=0&ogbl")
    driver.maximize_window()
    consent = driver.find_element_by_xpath(consent_path)
    consent.click()
    WebDriverWait(driver, 1).until(EC.presence_of_element_located((By.XPATH, searchbar_path)))

    box = driver.find_element_by_xpath(searchbar_path)
    box.send_keys(keyword)
    box.send_keys(Keys.ENTER)

    #######################
    # for i in range(100):
    #     driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
    #     try:
    #         show_button = driver.find_element_by_xpath(show_more)
    #         show_button.click()
    #     except:
    #         pass
    #     time.sleep(0.1)
    #
    # soup = bs4.BeautifulSoup(driver.page_source, 'html.parser')
    # img_tags = soup.find_all("img", class_='rg_i')
    # print(len(img_tags))
    #
    # count = 0
    # for elt in img_tags:
    #     try:
    #         src = elt['src']
    #         # print(src)
    #         if src[0] == 'h':
    #                 urllib.request.urlretrieve(src, 'img\\'+str(count)+".png")
    #         else:
    #             with open('img\\'+str(count)+".png", "wb") as fh:
    #                 # print(src[23:])
    #                 fh.write(base64.b64decode(src[22:]))
    #     except Exception as e:
    #         print(e)
    #     count += 1
    #######################

    #######################
    # pg.press("left")
    # pg.press("enter")
    count = 0
    for _ in range(20):
        try:
            save_folder = 'C:\\Users\\Revive\\PycharmProjects\\Rust_Native_ML\\img'
            count += 1
            print(count)

            pg.press("right")

            img_mini = driver.find_element_by_xpath(img_url(count))
            img_mini.click()

            img_macro = driver.find_element_by_xpath(img_after)

            # pg.click(button="left")



            # for a in img_macro.get_property('attributes'):
            #     print(a)

            # if src[0] == 'h':
            #     urllib.request.urlretrieve(src, 'img\\'+'modern'+str(count)+".png")
            # else:
            #     with open('img\\'+str(count)+".png", "wb") as fh:
            #         print(src[23:])
            #         fh.write(base64.b64decode(src[22:]))

        except Exception as e:
            print(e)

        try:
            show_button = driver.find_element_by_xpath(show_more)
            show_button.click()
        except:
            pass

        # time.sleep(1)
    #######################

    # driver.quit()


if __name__ == '__main__':
    run()

