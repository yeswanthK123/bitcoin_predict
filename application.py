from flask import Flask, render_template
from flask_socketio import SocketIO
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import requests
import collections
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'
socketio = SocketIO(app)

# Load the pre-trained LSTM model and scaler
model = load_model('bitcoin_lstm_model.h5')
scaler = MinMaxScaler()
scaler.min_, scaler.scale_ = np.load('scaler_params.npy', allow_pickle=True)

# Initialize a variable to store the last price
last_price = 0

# Define your CryptoCompare API key
api_key = 'd7e5542ec30c465824f676595f41c4fced2011b848fa8d19001af2d23bcc790a'

# Define the cryptocurrency pair and the exchange
pair = 'BTC-USD'
exchange = 'Coinbase'

# Define the endpoint for historical data
url = 'https://min-api.cryptocompare.com/data/v2/histoday'

# Initialize a deque to accumulate historical data
historical_data = collections.deque(maxlen=60)

def get_historical_data():
    global last_price, historical_data
    while True:
        params = {
            'fsym': 'BTC',
            'tsym': 'USD',
            'e': exchange,
            'api_key': api_key
        }

        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            if 'Data' in data and 'Data' in data['Data']:
                new_data = data['Data']['Data']
                historical_data.extend(new_data)

                if len(historical_data) >= 60:
                    last_close_price = historical_data[-1]['close']

                    scaled_input_data = scaler.transform(np.array(last_close_price).reshape(1, 1))
                    input_data = np.array([item['close'] for item in historical_data])
                    scaled_input_data = scaler.transform(input_data[:, np.newaxis])

                    X_test = np.reshape(scaled_input_data, (1, 60, 1))

                    predicted_prices = model.predict(X_test)
                    predicted_prices = scaler.inverse_transform(predicted_prices)
                    next_predicted_price = float(predicted_prices[0][-1])

                    last_price = last_close_price

                    socketio.emit('update_data', {
                        'last_price': last_close_price,
                        'predicted_price': next_predicted_price,
                    })
                    print(f'Last Price: {last_close_price}, Predicted Price: {next_predicted_price}')
            else:
                print(f'Invalid API response data structure')
        else:
            print(f'Request failed with status code: {response.status_code}')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.start_background_task(get_historical_data)
    socketio.run(app, debug=True)
