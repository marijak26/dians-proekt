import os
import pandas as pd
import numpy as np
from flask import Flask, render_template
import ta
import plotly.graph_objects as go
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from translate import Translator
import pdfplumber
import pytesseract
from sklearn.preprocessing import MinMaxScaler
from keras.api.models import Sequential
from keras.api.layers import LSTM, Dense, Input, Dropout
from sklearn.metrics import mean_squared_error, r2_score

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
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format="%d.%m.%Y")
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


def prepare_data_for_lstm(df, time_steps, feature_col='Price of last transaction'):
    try:
        if feature_col not in df.columns:
            raise ValueError(f"Feature column '{feature_col}' not found in DataFrame.")

        df.set_index(keys=["Date"], inplace=True)
        df.sort_index(inplace=True)

        df = df[[feature_col]]
        df = df.copy()

        for i in range(1, time_steps + 1):
            df.loc[:, f'lag_{i}'] = df[feature_col].shift(i)

        df.dropna(axis=0, inplace=True)

        X, y = df.drop(columns=feature_col, axis=1), df[feature_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        scaler = MinMaxScaler()
        y_train = scaler.fit_transform(y_train.to_numpy().reshape(-1, 1))

        X_train = X_train.reshape(X_train.shape[0], time_steps, (X_train.shape[1] // time_steps))
        X_test = X_test.reshape(X_test.shape[0], time_steps, (X_test.shape[1] // time_steps))

        if len(df) <= time_steps:
            raise ValueError(f"Not enough data to create sequences with {time_steps} time steps.")

        return X_train, y_train, X_test, y_test, scaler
    except Exception as e:
        print(f"Error in prepare_data_for_lstm: {e}")
        return None, None, None, None, None


def train_lstm_model(X_train, y_train, epochs=50, batch_size=32):
    try:
        model = Sequential([
            Input((X_train.shape[1], X_train.shape[2],)),
            LSTM(units=32, return_sequences=True, activation="relu"),
            Dropout(0.2),
            LSTM(units=8, return_sequences=False, activation="relu"),
            Dense(units=1, activation="linear")
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')
        history = model.fit(X_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size,
                            verbose=1, shuffle=False)

        return model, history
    except Exception as e:
        print(f"Error during LSTM model training: {e}")
        return None, None


def forecast_with_lstm(model, X_test, y_test, scaler, df):
    try:
        predictions = model.predict(X_test)
        predictions = predictions.reshape(-1, 1)
        predictions = scaler.inverse_transform(predictions)
        actual_data = y_test
        mse = mean_squared_error(actual_data, predictions)
        score = r2_score(actual_data, predictions)

        x_values = df.index[-len(actual_data):]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=x_values, y=actual_data,
            mode='lines', name='Actual Prices', line=dict(color='blue')
        ))

        fig.add_trace(go.Scatter(
            x=x_values, y=predictions.flatten(),
            mode='lines', name='Predicted Prices', line=dict(color='red', dash='dash')
        ))

        fig.update_layout(
            title='Actual vs Predicted Prices',
            xaxis_title='Date',
            yaxis_title='Price',
            template="plotly_white"
        )

        chart_html = fig.to_html(full_html=False)

        return chart_html, mse, score
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

        time_steps = 3
        X_train, y_train, X_test, y_test, scaler = prepare_data_for_lstm(df, time_steps=time_steps)

        if X_train is None or y_train is None:
            return f"Error preparing data for LSTM model for {filename}.", 500

        model, history = train_lstm_model(X_train, y_train)

        chart_html, mse, score = forecast_with_lstm(model, X_test, y_test, scaler, df)

        return render_template(
            'lstm_results.html',
            filename=filename,
            chart_html=chart_html,
            mse=mse,
            score=score
        )
    except Exception as e:
        print(f"Error in LSTM prediction route: {e}")
        return "Error occurred while processing LSTM prediction.", 500


if __name__ == '__main__':
    app.run(debug=True)
