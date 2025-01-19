from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.api.models import Sequential, load_model
from keras.api.layers import LSTM, Dense, Input, Dropout
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import pickle


app = Flask(__name__)

class LSTM_Service:
    def prepare_data_for_lstm(self, df, time_steps, feature_col='Price of last transaction'):
        try:
            if feature_col not in df.columns:
                raise ValueError(f"Feature column '{feature_col}' not found in DataFrame.")

            df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format="%a, %d %b %Y %H:%M:%S GMT")
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

    def train_lstm_model(self, X_train, y_train, epochs=50, batch_size=32):
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

            model_path = 'model.h5'
            model.save(model_path)

            return model_path

        except Exception as e:
            print(f"Error during LSTM model training: {e}")
            return None, None

    def forecast_with_lstm(self, model, X_test, y_test, scaler, df):
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


lstm_service = LSTM_Service()

@app.route('/api/prepare_data', methods=['POST'])
def prepare_data():
    try:
        data = request.get_json()
        df = pd.DataFrame(data['data'])
        time_steps = data['time_steps']

        X_train, y_train, X_test, y_test, scaler = lstm_service.prepare_data_for_lstm(df, time_steps)

        if X_train is None:
            return jsonify({'error': 'Data preparation failed.'}), 400

        scaler_path = 'scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

        response = {
            'X_train': X_train.tolist(),
            'X_test': X_test.tolist(),
            'y_train': y_train.tolist(),
            'y_test': y_test.tolist(),
            'scaler': scaler_path
        }


        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/train_model', methods=['POST'])
def train_model():
    try:
        data = request.get_json()
        X_train = np.array(data['X_train'])
        y_train = np.array(data['y_train'])

        model_path = lstm_service.train_lstm_model(X_train, y_train)

        if model_path is None:
            return jsonify({'error': 'Model training failed.'}), 400

        return jsonify({
            'model_path': model_path,
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/forecast', methods=['POST'])
def forecast():
    try:
        data = request.get_json()
        model_path = data['model_path']
        model = load_model(model_path)

        X_test = np.array(data['X_test'])
        y_test = np.array(data['y_test'])

        scaler_path = data['scaler']
        scaler = MinMaxScaler()

        df = pd.DataFrame(data['data'])
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        chart_html, mse, score = lstm_service.forecast_with_lstm(model, X_test, y_test, scaler, df)

        if chart_html is None:
            return jsonify({'error': 'Forecasting failed.'}), 400

        return jsonify({'chart_html': chart_html, 'mse': mse, 'r2_score': score}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(port=5004, debug=True, host='0.0.0.0')
