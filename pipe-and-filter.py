from datetime import datetime

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.select import Select
import pandas as pd


def filter_1(url):
    browser = webdriver.Chrome()

    browser.get(url)

    browser.implicitly_wait(10)

    all_codes = []
    codes = browser.find_elements(By.ID, 'Code')

    for code in codes:
        all_codes.append(code.text.split('\n'))
    all_codes = all_codes[0]
    all_codes = [code for code in all_codes if code.isalpha()]

    browser.quit()

    return all_codes


def filter_2(code, url):
    try:
        csv_file = pd.read_csv(f"{code}.csv")
        csv_file['Date'] = pd.to_datetime(csv_file['Date'], dayfirst=True)
        last_date = csv_file['Date'].max().strftime('%d/%m/%Y')
    except FileNotFoundError:
        browser = webdriver.Chrome()
        browser.get(url)

        browser.implicitly_wait(10)
        list = []
        options = Select(browser.find_element(By.ID, 'Code'))
        options.select_by_visible_text(code)
        find = browser.find_element(By.CSS_SELECTOR, '.container-end > input')
        find.click()
        body = browser.find_element(By.TAG_NAME, 'tbody')
        rows = body.find_elements(By.TAG_NAME, 'tr')
        dates = []
        cena_na_posledna_transakcija = []
        mak = []
        min = []
        prosecna_cena = []
        prom = []
        kolicina = []
        promet = []
        vkupen_promet = []

        for row in rows:
            date = row.find_element(By.CSS_SELECTOR, 'td:nth-child(1)').text
            if not date:
                continue
            else:
                dates.append(datetime.strptime(date, '%m/%d/%Y').strftime('%d/%m/%Y'))
            cena_na_posledna_transakcija.append(float(row.find_element(By.CSS_SELECTOR, 'td:nth-child(2)').text.replace(',', '')))
            mak.append(float(row.find_element(By.CSS_SELECTOR, 'td:nth-child(3)').text.replace(',', '')))
            min.append(float(row.find_element(By.CSS_SELECTOR, 'td:nth-child(4)').text.replace(',', '')))
            prosecna_cena.append(float(row.find_element(By.CSS_SELECTOR, 'td:nth-child(5)').text.replace(',', '')))
            prom.append(float(row.find_element(By.CSS_SELECTOR, 'td:nth-child(6)').text.replace(',', '')))
            kolicina.append(int(row.find_element(By.CSS_SELECTOR, 'td:nth-child(7)').text.replace(',', '')))
            promet.append(int(row.find_element(By.CSS_SELECTOR, 'td:nth-child(8)').text.replace(',', '')))
            vkupen_promet.append(int(row.find_element(By.CSS_SELECTOR, 'td:nth-child(9)').text.replace(',', '')))
        for i in range(len(dates)):
            list.append({'Date': dates[i], 'Price of last transaction': cena_na_posledna_transakcija[i],
                         'Max': mak[i], 'Min': min[i], 'Average price': prosecna_cena[i],
                         '%chg.': prom[i], 'Volume': kolicina[i],
                         'Turnover in BEST in denars': promet[i],
                         'Total turnover in denars': vkupen_promet[i]})
        df = pd.DataFrame(list)
        csv_file = df.to_csv(f'{code}.csv', index=False, float_format='%.2f')

    csv_file['Date'] = pd.to_datetime(csv_file['Date'], dayfirst=True)
    last_date = csv_file['Date'].max().strftime('%d/%m/%Y')


    return code, last_date


def filter_3(code, lastDate):
    print(code,lastDate)


if __name__ == '__main__':
    codes = filter_1('https://www.mse.mk/en/stats/symbolhistory/kmb')
    code,lastday = filter_2('KMB', 'https://www.mse.mk/en/stats/symbolhistory/kmb')
    filter_3(code,lastday)

