from flask import Flask, request, jsonify
import pandas as pd
from io import StringIO

DATA_FOLDER = 'data'
REPORTS_FOLDER = 'financial_reports'

app = Flask(__name__)

class DataService:
    def clean_data(self, file_path):
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

    def resample_data(self, df, timeframe):
        try:
            if df.empty:
                raise ValueError("DataFrame is empty. No data to resample.")
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format="%a, %d %b %Y %H:%M:%S GMT")
            df = df.dropna(subset=['Date'])


            df = df.set_index('Date').resample(timeframe).agg({
                'Price of last transaction': 'last',
                'Max': 'max',
                'Min': 'min',
                'Average price': 'mean',
                '%chg.': 'mean',
                'Volume': 'sum'
            }).dropna().reset_index()

            df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format="%a, %d %b %Y %H:%M:%S GMT").dt.strftime(
                "%d.%m.%Y")

            df = df.sort_index(ascending=False)
            return df
        except ValueError as e:
            print(f"Error: {e}")
            return pd.DataFrame()


data_service = DataService()


@app.route('/api/clean_data', methods=['POST'])
def clean_data():
    try:
        file = request.files['file']
        file_content = file.read().decode('utf-8')
        file_io = StringIO(file_content)

        df = data_service.clean_data(file_io)

        if df.empty:
            return jsonify({"error": "No valid data found in the file"}), 400

        return jsonify(df.to_dict(orient='records')), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/resample_data', methods=['POST'])
def resample_data():
    try:
        data = request.get_json()
        timeframe = data.get('timeframe', 'D')
        df = pd.DataFrame(data['data'])

        resampled_df = data_service.resample_data(df, timeframe)

        if resampled_df.empty:
            return jsonify({"error": "No valid data to resample"}), 400

        return jsonify(resampled_df.to_dict(orient='records')), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(port=5001, debug=True, host='0.0.0.0')