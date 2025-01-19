import os

import pytesseract
import tensorflow
import requests
from flask import Flask, render_template

import pandas as pd

app = Flask(__name__)

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

@app.route('/run-scraping')
def update_data():
    response = requests.get('http://scraping-servi ce:5005/api/run-scraping')

    if response.status_code != 200:
        return f"Error collecting data", 500
    else:
        print(response.json())

@app.route('/company/all')
def list_files():
    try:
        files = [f[:-4] for f in os.listdir(f'data') if f.endswith('.csv')]
    except FileNotFoundError:
        files = []
    return render_template('companies.html', files=files)


@app.route('/company/<filename>')
def display_file(filename):
    try:
        file_path = os.path.join(f'data', filename + ".csv")

        data = clean_data(file_path, filename)

        daily_data = calculate_indicators(filename, resample_data(filename, data, 'D'))
        weekly_data = calculate_indicators(filename, resample_data(filename, data, 'W'))
        monthly_data = calculate_indicators(filename, resample_data(filename, data, 'M'))

        chart_html = plot_custom_chart(filename, daily_data)
        classification = analyze_sentiment(filename)

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
    except Exception as e:
        return f"Error: {str(e)}", 500


@app.route('/lstm/<filename>')
def lstm_prediction(filename):
    try:
        file_path = os.path.join(f'data', filename + ".csv")

        df = clean_data(file_path, filename)

        time_steps = 3
        X_train, y_train, X_test, y_test, scaler_data = prepare_data_for_lstm(filename, df, time_steps)

        model_path = train_lstm_model(filename, X_train, y_train)

        chart_html, mse, score = forecast_with_lstm(filename, model_path, X_test, y_test, scaler_data, df)

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


def clean_data(file_path, filename):
    with open(file_path, 'rb') as f:
        response = requests.post('http://data-processing-service:5001/api/clean_data', files={'file': f})

    if response.status_code != 200:
        return f"Error cleaning data for {filename}.", 500

    data = pd.DataFrame(response.json())

    if data.empty:
        return f"No valid data found for {filename}.", 404

    return data


def resample_data(filename, df, timeframe):
    data_to_process = {'data': df.to_dict(orient='records'),
                       'timeframe': timeframe}
    response = requests.post('http://data-processing-service:5001/api/resample_data', json=data_to_process)

    if response.status_code != 200:
        return f"Error resampling data for {filename}.", 500

    return pd.DataFrame(response.json())


def calculate_indicators(filename, df):
    response = requests.post('http://fundamental-analysis-service:5002/api/calculate_indicators', json=df.to_dict(orient='records'))

    if response.status_code != 200:
        return f"Error calculating indicators for {filename}.", 500

    return pd.DataFrame(response.json())


def plot_custom_chart(filename, df):
    df = df.dropna()
    data_to_process = {'data': df.to_dict(orient='records'),
                       'company_name': filename}
    response = requests.post('http://fundamental-analysis-service:5002/api/plot_chart', json=data_to_process)

    if response.status_code != 200:
        return f"Error plotting chart for {filename}.", 500

    return response.json()['chart_html']


def analyze_sentiment(filename):
    response = requests.post('http://sentiment-analysis-service:5003/api/analyze_sentiment',
                             json={'file': f'financial_reports/{filename}_report.pdf'})

    if response.status_code != 200:
        return f"Error analyzing sentiment for {filename}.", 500

    return response.json()['sentiment']


def prepare_data_for_lstm(filename, df, time_steps):
    data_to_process = {'data': df.to_dict(orient='records'),
                       'time_steps': time_steps}
    response = requests.post('http://lstm-prediction-service:5004/api/prepare_data',
                             json=data_to_process)
    if response.status_code != 200:
        return f"Error preparing lstm data for {filename}.", 500

    data = response.json()
    X_train = data['X_train']
    X_test = data['X_test']

    y_train = data['y_train']
    y_test = data['y_test']

    scaler_path = data['scaler']

    return X_train, y_train, X_test, y_test, scaler_path

def train_lstm_model(filename, X_train, y_train):
    data_to_process = {'X_train': X_train,
                       'y_train': y_train}
    response = requests.post('http://lstm-prediction-service:5004/api/train_model',
                             json=data_to_process)

    if response.status_code != 200:
        return f"Error training model for {filename}.", 500

    data = response.json()
    model_path = data['model_path']
    return model_path

def forecast_with_lstm(filename, model_path, X_test, y_test, scaler, df):
    data_to_process = {'X_test': X_test,
                       'y_test': y_test,
                       'model_path': model_path,
                       'scaler': scaler,
                       'data': df.to_dict(orient='records')}
    response = requests.post('http://lstm-prediction-service:5004/api/forecast',
                             json=data_to_process)

    if response.status_code != 200:
        return f"Error training model for {filename}.", 500

    return response.json()['chart_html'],response.json()['mse'], response.json()['r2_score']


if __name__ == '__main__':
    app.run(port=5000, debug=True, host='0.0.0.0')
