import concurrent.futures
import time
from datetime import datetime, timedelta

from flask import Flask, jsonify
from selenium.common import TimeoutException
from selenium.webdriver.support import expected_conditions as EC

from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd
from bs4 import BeautifulSoup
from selenium.webdriver.support.wait import WebDriverWait

app = Flask(__name__)

class Pipe:

    def filter_1(self):
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        browser = webdriver.Chrome(options=options)
        browser.get('https://www.mse.mk/en/stats/current-schedule')
        source = browser.page_source
        soup = BeautifulSoup(source, 'html.parser')
        all_codes = []
        codes = soup.select('tr td:nth-child(1) a')

        for code in codes:
            code = code.text.split('\n')[0]
            if code.isalpha():
                all_codes.append(code)
        browser.quit()

        return all_codes

    def filter_2(self, code):
        try:
            csv_file = pd.read_csv(f"../data/{code}.csv")
            csv_file['Date'] = pd.to_datetime(csv_file['Date'], dayfirst=True)
            last_date = csv_file['Date'].max().strftime('%d.%m.%Y')
            self.filter_3(code, last_date)
        except FileNotFoundError:
            options = webdriver.ChromeOptions()
            options.add_argument("--headless")
            browser = webdriver.Chrome(options=options)
            browser.get(f"https://www.mse.mk/en/stats/symbolhistory/{code}")

            to_date = datetime.today()
            from_date = (to_date - timedelta(days=364)).date()
            to_date = to_date.date()
            all_data = []

            for i in range(10):
                enter_dates(browser, from_date, to_date)
                data = extract_data(browser)
                if data is not None:
                    all_data.append(data)
                to_date = from_date
                from_date = to_date - timedelta(days=364)

            from_date = to_date - timedelta(days=10)

            enter_dates(browser, from_date, to_date)

            data = extract_data(browser)
            if data is not None:
                all_data.append(data)
            df = pd.concat(all_data, ignore_index=True)
            df.drop_duplicates(subset='Date')
            df.to_csv(f'../data/{code}.csv', index=False)

            browser.quit()

    def filter_3(self, code, last_date):
        if datetime.today().date().strftime('%d.%m.%Y') != last_date:
            options = webdriver.ChromeOptions()
            options.add_argument("--headless")
            browser = webdriver.Chrome(options=options)
            browser.get(f"https://www.mse.mk/en/stats/symbolhistory/{code}")
            from_input = browser.find_element(By.CSS_SELECTOR,
                                              '#report-filter-container > ul > li:nth-child(1) .div-input > input')
            from_input.click()
            from_input.clear()
            from_input.send_keys(datetime.strptime(str(last_date), "%d.%m.%Y").strftime("%m/%d/%Y"))
            find = browser.find_element(By.CSS_SELECTOR, '.container-end > input')
            find.click()

            data = extract_data(browser)
            if data is not None:
                existing_df = pd.read_csv(f'../data/{code}.csv')
                existing_df = pd.concat([data, existing_df], ignore_index=True)
                existing_df = existing_df.drop_duplicates(subset='Date')
                existing_df.to_csv(f'../data/{code}.csv', index=False)

            browser.quit()


def enter_dates(browser, from_date, to_date):
    from_input = browser.find_element(By.CSS_SELECTOR,
                                      '#report-filter-container > ul > li:nth-child(1) .div-input > input')
    to_input = browser.find_element(By.CSS_SELECTOR,
                                    '#report-filter-container > ul > li:nth-child(2) .div-input > input')
    from_input.click()
    from_input.clear()
    from_input.send_keys(datetime.strptime(str(from_date), "%Y-%m-%d").strftime("%m/%d/%Y"))

    to_input.click()
    to_input.clear()
    to_input.send_keys(datetime.strptime(str(to_date), "%Y-%m-%d").strftime("%m/%d/%Y"))
    find = browser.find_element(By.CSS_SELECTOR, '.container-end > input')
    find.click()


def extract_data(browser):
    while True:
        list = []

        try:
            WebDriverWait(browser, 1).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "table tbody"))
            )
        except TimeoutException:
            return None

        source = browser.page_source
        soup = BeautifulSoup(source, 'html.parser')
        body = soup.select_one('tbody')
        rows = body.select('tr')

        for row in rows:
            promet = row.select_one('td:nth-child(8)').text
            if promet is None or promet == 0 or promet == "0":
                continue
            promet = format_numbers(promet)
            date = row.select_one('td:nth-child(1)').text
            date = datetime.strptime(date, '%m/%d/%Y').strftime('%d.%m.%Y')
            cena_na_posledna_transakcija = format_numbers(row.select_one('td:nth-child(2)').text)
            mak = format_numbers(row.select_one('td:nth-child(3)').text)
            min = format_numbers(row.select_one('td:nth-child(4)').text)
            prosecna_cena = format_numbers(row.select_one('td:nth-child(5)').text)
            prom = format_numbers(row.select_one('td:nth-child(6)').text)
            kolicina = format_numbers(row.select_one('td:nth-child(7)').text)
            vkupen_promet = format_numbers(row.select_one('td:nth-child(9)').text)
            list.append({'Date': date, 'Price of last transaction': cena_na_posledna_transakcija,
                         'Max': mak, 'Min': min, 'Average price': prosecna_cena,
                         '%chg.': prom, 'Volume': kolicina,
                         'Turnover in BEST in denars': promet,
                         'Total turnover in denars': vkupen_promet})

        return pd.DataFrame(list)


def format_numbers(number):
    number = number.replace(",", "#")
    number = number.replace(".", ",")
    return number.replace("#", ".")


@app.route('/api/run-scraping', methods=['GET'])
def run_scraping():
    start_time = time.time()
    pipe = Pipe()
    codes = pipe.filter_1()

    codes = [codes[i:i + 20] for i in range(0, len(codes), 20)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for code_chunk in codes:
            for code in code_chunk:
                executor.submit(pipe.filter_2, code)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Time taken: {(execution_time / 60):.2f} min")

    return jsonify({"message": "Scraping completed successfully."})


if __name__ == '__main__':
    app.run(debug=True, port=5005, host='0.0.0.0')
