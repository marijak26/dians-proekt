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
from sklearn.preprocessing import MinMaxScaler
from keras.api.models import Sequential
from keras.api.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

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


def prepare_data_for_lstm(df, feature_col='Price of last transaction', time_steps=30):
    try:
        if feature_col not in df.columns:
            raise ValueError(f"Feature column '{feature_col}' not found in DataFrame.")

        data = df[feature_col].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        if len(scaled_data) <= time_steps:
            raise ValueError(f"Not enough data to create sequences with {time_steps} time steps.")

        X, y = [], []
        for i in range(time_steps, len(scaled_data)):
            X.append(scaled_data[i - time_steps:i, 0])
            y.append(scaled_data[i, 0])

        X = np.array(X)
        y = np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        return X, y, scaler
    except Exception as e:
        print(f"Error in prepare_data_for_lstm: {e}")
        return None, None, None



def train_lstm_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    try:
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            LSTM(units=50, return_sequences=False),
            Dense(units=25),
            Dense(units=1)
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size,
                            verbose=1)

        return model, history
    except Exception as e:
        print(f"Error during LSTM model training: {e}")
        return None, None


def forecast_with_lstm(model, X_test, scaler, df, feature_col='Price of last transaction', time_steps=30):
    try:
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)

        actual_data = df[feature_col].values[-len(predictions):].reshape(-1, 1)
        mse = mean_squared_error(actual_data, predictions)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=np.arange(len(actual_data)), y=actual_data.flatten(),
            mode='lines', name='Actual Prices', line=dict(color='blue')
        ))

        fig.add_trace(go.Scatter(
            x=np.arange(len(predictions)), y=predictions.flatten(),
            mode='lines', name='Predicted Prices', line=dict(color='red')
        ))

        fig.update_layout(
            title='Stock Price Prediction with LSTM',
            xaxis=dict(title='Time'),
            yaxis=dict(title='Price'),
            legend=dict(orientation="h", x=0, y=-0.2),
            template="plotly_white"
        )

        chart_html = fig.to_html(full_html=False)

        print(f"Mean Squared Error (MSE): {mse}")

        return chart_html, mse
    except Exception as e:
        print(f"Error during forecasting: {e}")
        return None, None


@app.route('/lstm/<filename>')
def lstm_prediction(filename):
    try:
        file_path = os.path.join(DATA_FOLDER, filename + ".csv")
        df = clean_data(file_path)

        if df.empty:
            return f"No valid data found for {filename}.", 404

        time_steps = 30
        X, y, scaler = prepare_data_for_lstm(df, time_steps=time_steps)

        if X is None or y is None:
            return f"Error preparing data for LSTM model for {filename}.", 500

        train_size = int(len(X) * 0.7)
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:], y[train_size:]

        if len(X_train) == 0 or len(X_val) == 0:
            return f"Insufficient data to train/test the LSTM model for {filename}.", 500

        model, history = train_lstm_model(X_train, y_train, X_val, y_val)

        chart_html, mse = forecast_with_lstm(model, X_val, scaler, df, time_steps=time_steps)

        return render_template(
            'lstm_results.html',
            filename=filename,
            chart_html=chart_html,
            mse=mse
        )
    except Exception as e:
        print(f"Error in LSTM prediction route: {e}")
        return "Error occurred while processing LSTM prediction.", 500



if __name__ == '__main__':
    app.run(debug=True)
