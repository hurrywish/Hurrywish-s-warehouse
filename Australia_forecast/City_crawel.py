from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from lxml import etree
import pickle
import time

city_list = ['Nuriootpa', 'Perth', 'Bendigo', 'Albury', 'CoffsHarbour',
             'WaggaWagga', 'Williamtown', 'Ballarat', 'Sale',
             'Tuggeranong', 'Townsville', 'Launceston', 'Darwin', 'Albany',
             'Cobar', 'NorfolkIsland', 'Sydney', 'SydneyAirport',
             'MelbourneAirport', 'Witchcliffe', 'MountGinini', 'Adelaide',
             'Portland', 'SalmonGums', 'Hobart', 'MountGambier', 'Woomera',
             'Wollongong', 'Mildura', 'AliceSprings', 'PerthAirport',
             'Melbourne', 'Brisbane', 'Watsonia', 'Cairns', 'Penrith',
             'PearceRAAF', 'NorahHead', 'Nhil', 'Richmond', 'Newcastle',
             'Moree', 'Uluru', 'Walpole', 'GoldCoast', 'Canberra', 'Katherine',
             'BadgerysCreek']
major_city_list = ['Sydney', 'Melbourne', 'Perth', 'Brisbane', 'Darwin', 'Hobart', 'Adelaide', 'Canberra']

options = webdriver.ChromeOptions()
prefs = {'profile.managed_default_content_settings.images': 2}
options.add_experimental_option('prefs', prefs)
driver = webdriver.Chrome('/Users/hurrywish/Downloads/chromedriver')
driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
    "source": """
    Object.defineProperty(navigator, 'webdriver', {
      get: () => undefined
    })
  """
})

geo_dict = dict()
for city in city_list:
    driver.get('https://www.geodatos.net/en/search?o=en%2Fcoordinates&q=&s=')
    search_column = driver.find_element(By.XPATH, '//input[@x-ref="q"]')

    actions = ActionChains(driver)
    search_column.click()
    search_column.send_keys(city)
    actions.send_keys(Keys.ENTER).perform()

    middle_link = driver.find_element(By.XPATH, '//a[contains(@class,"blue")][1]')
    middle_link.click()
    page_source = driver.page_source
    html = etree.HTML(page_source)
    try:
        location = html.xpath('//div[contains(@class,"rounded p-4")][2]//p//text()')[0].strip()
        country = html.xpath('//span[contains(@class,"hidden")][1]//a[last()]//text()')
    except:
        location=None
        country=None



    print(country, city, location)
    geo_dict[city] = location
geo_dict['Dartmoor'] = '37.55° S  141.16° E'
with open('geo_dict.pkl', 'wb') as fp:
    pickle.dump(geo_dict, fp, pickle.HIGHEST_PROTOCOL)

# major_city_dict = dict()
# for city in city_list:
#     driver.get('https://www.geodatos.net/en/search?o=en%2Fcoordinates&q=&s=')
#     search_column = driver.find_element(By.XPATH, '//input[@x-ref="q"]')
#
#     actions = ActionChains(driver)
#     search_column.click()
#     search_column.send_keys(city)
#     actions.send_keys(Keys.ENTER).perform()
#
#     middle_link = driver.find_element(By.XPATH, '//a[contains(@class,"blue")][1]')
#     middle_link.click()
#     page_source = driver.page_source
#     html = etree.HTML(page_source)
#     location = html.xpath('//div[contains(@class,"rounded p-4")][2]//p//text()')[0].strip()
#     print(city, location)
#     major_city_dict[city] = location
#
# with open('major_city_dict.pkl', 'wb') as fp:
#     pickle.dump(major_city_dict, fp, pickle.HIGHEST_PROTOCOL)
