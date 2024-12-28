import os
import pandas as pd
import numpy as np
from flask import Flask, render_template
import ta
import plotly.graph_objects as go
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from translate import Translator
import pdfplumber
import pytesseract

app = Flask(__name__)

DATA_FOLDER = 'data'
REPORTS_FOLDER = 'financial_reports'

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()


def clean_data(file_path):
    try:
        df = pd.read_csv(file_path)

        if df.empty:
            raise pd.errors.EmptyDataError("The CSV file is empty, no data to load.")
    except pd.errors.EmptyDataError as e:
        print(f"Error: {e}")
        return pd.DataFrame()

    numeric_columns = ['Price of last transaction', 'Max', 'Min', 'Average price', '%chg.']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col].replace({',': '', '%': ''}, regex=True), errors='coerce')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna()

    df = df.sort_values(by='Date', ascending=False)

    return df


def calculate_indicators(df):
    try:
        if df.empty:
            raise ValueError("DataFrame is empty. No indicators to calculate.")

        df = df.sort_values(by='Date', ascending=False)

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
        df['CCI'] = ta.trend.CCIIndicator(high=df['Max'], low=df['Min'], close=df['Price of last transaction'],
                                          window=20).cci()

        df['Signal'] = np.where(df['RSI'] < 30, 'Buy',
                                np.where(df['RSI'] > 70, 'Sell', 'Hold'))
        return df
    except ValueError as e:
        print(f"Error: {e}")
        return pd.DataFrame()


def resample_data(df, timeframe):
    try:
        if df.empty:
            raise ValueError("DataFrame is empty. No data to resample.")

        df = df.set_index('Date').resample(timeframe).agg({
            'Price of last transaction': 'last',
            'Max': 'max',
            'Min': 'min',
            'Average price': 'mean',
            '%chg.': 'mean',
            'Volume': 'sum'
        }).dropna().reset_index()

        df = df.sort_values(by='Date', ascending=False)

        return df
    except ValueError as e:
        print(f"Error: {e}")
        return pd.DataFrame()


def plot_custom_chart(df, company_name):
    try:
        if df.empty:
            raise ValueError(f"No data available to plot for {company_name}.")

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['Price of last transaction'],
            mode='lines', name='Price', line=dict(color='blue')
        ))

        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['SMA_10'],
            mode='lines', name='SMA 10', line=dict(color='orange')
        ))
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['EMA_10'],
            mode='lines', name='EMA 10', line=dict(color='green')
        ))

        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['RSI'],
            mode='lines', name='RSI', line=dict(color='red', dash='dot')
        ))

        fig.update_layout(
            title=f"{company_name} Stock Data with Indicators",
            xaxis=dict(title='Date'),
            yaxis=dict(title='Price'),
            legend=dict(orientation="h", x=0, y=-0.2),
            template="plotly_white"
        )

        return fig.to_html(full_html=False)

    except ValueError as e:
        print(f"Error: {e}")
        return ""


def translate_text(text):
    translator = Translator(to_lang='en', from_lang='mk')
    chunks = chunk_text(text)

    translated_chunks = []
    for chunk in chunks:
        translated_chunk = translator.translate(chunk)
        translated_chunks.append(translated_chunk)

    translated_text = " ".join(translated_chunks)
    return translated_text


def chunk_text(text, max_length=500):
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]


def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text


def extract_text_from_image_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            image = page.to_image()
            text += pytesseract.image_to_string(image.original, lang='mkd')
    return text


def analyze_sentiment(pdf_path):
    try:
        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            text = extract_text_from_image_pdf(pdf_path)

        translated_text = translate_text(text)

        sentiment = sia.polarity_scores(translated_text)
        compound_score = sentiment['compound']
        sentiment_classification = classify_sentiment(compound_score)

        return sentiment_classification

    except FileNotFoundError:
        return "No financial reports from this year found."


def classify_sentiment(compound_score):
    if compound_score > 0.5:
        return "Buy"
    elif 0.2 <= compound_score <= 0.5:
        return "Buy/Hold"
    elif 0.0 <= compound_score < 0.2:
        return "Hold"
    elif -0.2 <= compound_score < 0.0:
        return "Sell"
    else:
        return "Sell/Avoid"


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

        if df.empty:
            return f"No valid data found for {filename}.", 404

        daily_data = calculate_indicators(df.copy())
        weekly_data = calculate_indicators(resample_data(df.copy(), 'W'))
        monthly_data = calculate_indicators(resample_data(df.copy(), 'M'))

        chart_html = plot_custom_chart(daily_data, filename)
        classification = analyze_sentiment(f'{REPORTS_FOLDER}/{filename}_report.pdf')

        return render_template(
            'file_contents.html',
            filename=filename,
            daily_data=daily_data.to_html(classes='table table-bordered', index=False),
            weekly_data=weekly_data.to_html(classes='table table-bordered', index=False),
            monthly_data=monthly_data.to_html(classes='table table-bordered', index=False),
            chart_html=chart_html,
            classification=classification
        )
    except FileNotFoundError:
        return f"File {filename} not found.", 404


if __name__ == '__main__':
    app.run(debug=True)
