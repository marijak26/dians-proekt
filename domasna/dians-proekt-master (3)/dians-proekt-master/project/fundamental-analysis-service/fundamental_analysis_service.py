import ta
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from flask import Flask, request, jsonify


app = Flask(__name__)

class IndicatorService:
    def calculate_indicators(self, df):
        try:
            if df.empty:
                raise ValueError("DataFrame is empty. No indicators to calculate.")

            df = df.sort_index(ascending=False)

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

    def plot_custom_chart(self, df, company_name):
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

indicator_service = IndicatorService()


@app.route('/api/calculate_indicators', methods=['POST'])
def calculate_indicators():
    try:
        data = request.get_json()
        df = pd.DataFrame(data)
        df_with_indicators = indicator_service.calculate_indicators(df)
        return jsonify(df_with_indicators.to_dict(orient='records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/plot_chart', methods=['POST'])
def plot_chart():
    try:
        data = request.get_json()
        df = pd.DataFrame(data['data'])
        company_name = data['company_name']
        chart_html = indicator_service.plot_custom_chart(df, company_name)
        return jsonify({'chart_html': chart_html})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(port=5002,debug=True, host='0.0.0.0')