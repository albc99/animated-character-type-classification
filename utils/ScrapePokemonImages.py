from selenium import webdriver
import urllib.request
import time

options = webdriver.ChromeOptions()
options.add_argument('--ignore-certificate-errors')

# replace with path to your chromedriver
driver = webdriver.Chrome(executable_path=r'C:\Users\Adam Kasumovic\Desktop\chromedriver.exe', options=options)
driver.get('https://pokemondb.net/pokedex/national')

SCROLL_PAUSE_TIME = 0.5

# Get scroll height
last_height = driver.execute_script("return document.body.scrollHeight")

current_height = 400
while True:
    # Scroll down to bottom
    driver.execute_script(f"window.scrollTo(0, {current_height});")

    # Wait to load page
    time.sleep(SCROLL_PAUSE_TIME)

    # Calculate new scroll height and compare with last scroll height
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height <= current_height:
        break

    current_height += 400

card_pics = driver.find_elements_by_xpath("//img[@class='img-fixed img-sprite']")
card_pics_links = [i.get_attribute('src') for i in card_pics]
print(card_pics_links)
print(len(card_pics_links))

driver.close()

# calling urlretrieve function to get resource
for link in card_pics_links:
    opener = urllib.request.URLopener()
    opener.addheader('User-Agent', 'whatever')
    filename, headers = opener.retrieve(link, '../data/images/'+ link[link.rfind('/')+1:link.rfind('.')] + '.jpg')
