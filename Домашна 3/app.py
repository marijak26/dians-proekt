import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import ta

app = Flask(__name__)

DATA_FOLDER = 'data'

def clean_data(file_path):
    df = pd.read_csv(file_path)

    numeric_columns = ['Price of last transaction', 'Max', 'Min', 'Average price', '%chg.']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col].replace({',': '', '%': ''}, regex=True), errors='coerce')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna()
    return df

def calculate_indicators(df):
    df = df.sort_values('Date')

    df['SMA_10'] = df['Price of last transaction'].rolling(window=10).mean()
    df['SMA_50'] = df['Price of last transaction'].rolling(window=50).mean()
    df['EMA_10'] = df['Price of last transaction'].ewm(span=10, adjust=False).mean()
    df['EMA_50'] = df['Price of last transaction'].ewm(span=50, adjust=False).mean()

    df['RSI'] = ta.momentum.RSIIndicator(close=df['Price of last transaction'], window=14).rsi()
    df['Stochastic'] = ta.momentum.StochasticOscillator(
        high=df['Max'], low=df['Min'], close=df['Price of last transaction'], window=14
    ).stoch()
    df['MACD'] = ta.trend.MACD(close=df['Price of last transaction']).macd()
    df['Williams %R'] = ta.momentum.WilliamsRIndicator(
        high=df['Max'], low=df['Min'], close=df['Price of last transaction'], lbp=14
    ).williams_r()
    df['CCI'] = ta.trend.CCIIndicator(high=df['Max'], low=df['Min'], close=df['Price of last transaction'], window=20).cci()

    df['Signal'] = np.where(df['RSI'] < 30, 'Buy',
                            np.where(df['RSI'] > 70, 'Sell', 'Hold'))
    return df

def resample_data(df, timeframe):
    df = df.set_index('Date').resample(timeframe).agg({
        'Price of last transaction': 'last',
        'Max': 'max',
        'Min': 'min',
        'Average price': 'mean',
        '%chg.': 'mean',
        'Volume': 'sum'
    }).dropna().reset_index()
    return df

@app.route('/company/all')
def list_files():
    try:
        files = [f[:-4] for f in os.listdir(DATA_FOLDER) if f.endswith('.csv')]
    except FileNotFoundError:
        files = []
    return render_template('companies.html', files=files)

@app.route('/company/<filename>')
def display_file(filename):
    try:
        file_path = os.path.join(DATA_FOLDER, filename + ".csv")
        df = clean_data(file_path)

        daily_data = calculate_indicators(df.copy())
        weekly_data = calculate_indicators(resample_data(df.copy(), 'W'))
        monthly_data = calculate_indicators(resample_data(df.copy(), 'M'))

        return render_template(
            'file_contents.html',
            filename=filename,
            daily_data=daily_data.to_html(classes='table table-bordered', index=False),
            weekly_data=weekly_data.to_html(classes='table table-bordered', index=False),
            monthly_data=monthly_data.to_html(classes='table table-bordered', index=False)
        )
    except FileNotFoundError:
        return f"File {filename} not found.", 404

if __name__ == '__main__':
    app.run(debug=True)
