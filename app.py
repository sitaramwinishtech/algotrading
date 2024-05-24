from flask import Flask, render_template, request, jsonify
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import Trading as td
import requests

app = Flask(__name__)

API_KEY = 'demo'
URL = "https://www.alphavantage.co/query?function=LISTING_STATUS&apikey=" + API_KEY

# Your existing functions with slight modifications pip install Flask yfinance scikit-learn pandas
def fetch_historical_data(symbol):
    return yf.download(symbol, start="2010-01-01", end="2024-04-04")

def fetch_fundamental_data(symbol):
    stock = yf.Ticker(symbol)
    print(stock)
    info = stock.info

    current_price = stock.info['currentPrice']

    print(f"The current price of {symbol} is: ${current_price}")
    fundamentals = {}
    try:
        earnings = stock.earnings
        print(earnings)
        fundamentals['EarningsGrowth'] = earnings.iloc[-1]['Earnings'] > earnings.iloc[-2]['Earnings'] if len(earnings) > 1 else False
    except Exception as e:
        fundamentals['EarningsGrowth'] = False
    fundamentals['ROE'] = info.get('returnOnEquity', 0)
    fundamentals['DE_Ratio'] = info.get('debtToEquity', 0)
    fundamentals['PE_Ratio'] = info.get('forwardPE', 0)
    fundamentals['DividendYield'] = info.get('dividendYield', 0) > 0
    return fundamentals

def evaluate_fundamentals1(fundamentals):
    criteria = [
        fundamentals['EarningsGrowth'],
        fundamentals['ROE'] > 0.15,
        fundamentals['DE_Ratio'] < 1,
        fundamentals['PE_Ratio'] < 25,
        fundamentals['DividendYield']
    ]
    return all(criteria)

def evaluate_fundamentals(fundamentals):
    criteria = [
        # It's okay if Earnings Growth is not present
        fundamentals['ROE'] > 0,  # Adjusting to > 0, to not strictly require 15%
        fundamentals['DE_Ratio'] < 5,  # Adjusting to < 5, to be less strict than < 1
        fundamentals['PE_Ratio'] < 50,  # Adjusting to < 50, considering different industries
        # Keeping DividendYield as a positive criterion but not mandatory
    ]
    # Considering stocks that meet at least half of the adjusted criteria
    return sum(criteria) >= len(criteria) / 2


def prepare_dataset1(symbols):
    data = []
    labels = []
    for symbol in symbols:
        fundamentals = fetch_fundamental_data(symbol)
        if fundamentals:
            data.append(list(fundamentals.values()))
            labels.append(1 if evaluate_fundamentals(fundamentals) else 0)
    return pd.DataFrame(data, columns=list(fundamentals.keys())), labels

def prepare_dataset(symbols):
    data = []
    labels = []
    for symbol in symbols:
        fundamentals = fetch_fundamental_data(symbol)
        print(f"Fundamentals for {symbol}: {fundamentals}")  # Debugging print
        if fundamentals:
            data.append(list(fundamentals.values()))
            labels.append(1 if evaluate_fundamentals(fundamentals) else 0)
    print(f"Data Prepared: {data}")  # Debugging print
    print(f"Labels Prepared: {labels}")  # Debugging print
    return pd.DataFrame(data, columns=list(fundamentals.keys()) if fundamentals else []), labels

def train_model(features, labels):
    # Check if there are enough samples to split
    if len(features) <= 1:
        print("Not enough data for training and testing. Consider adjusting your criteria.")
        return None

    # Check for variability in labels
    if len(set(labels)) < 2:
        print("All labels are the same. The model cannot be trained on this data.")
        return None

    # Proceed with the train-test split and model training
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model trained successfully with accuracy: {accuracy}")
    return accuracy


def train_model2(features, labels):
    if len(features) > 1:
        # Proceed with train-test split if there are more than one samples
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return accuracy_score(y_test, y_pred)
    elif len(features) == 1:
        # Handle case with a single sample (e.g., skip test or use different validation approach)
        print("Dataset contains only one sample, skipping train-test split.")
        # You might choose to train on this single sample and note the limitation
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(features, labels)
        # Since we can't test, return a default or placeholder accuracy
        return None
    else:
        # Handle empty dataset case
        print("Dataset is empty.")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    symbols = request.form.get('symbols').split(',')
    features, labels = prepare_dataset(symbols)
    accuracy = train_model(features, labels)
    if accuracy is not None:
        return jsonify({'message': 'Analysis complete', 'accuracy': accuracy})
    else:
        return jsonify({'message': 'Analysis could not be completed due to insufficient or non-variable data', 'accuracy': None})


@app.route('/analyze1', methods=['POST'])
def analyze1():
    symbol = request.form.get('symbols')
    if symbol:
        data = td.fetch_data(symbol)
        print("data::: ", data)
        if not data.empty:
            breakout_signals = td.analyze_stock(data)
            if not breakout_signals.empty:
                return render_template('results.html', signals=breakout_signals.to_html(), symbol=symbol)
            else:
                return render_template('results.html', message="No breakout signals found.", symbol=symbol)
        else:
            return render_template('results.html', message="Data fetching was unsuccessful.", symbol=symbol)
    return render_template('index.html', message="Please enter a valid symbol.")

CSV_FILE_PATH = 'templates/MCAP28032024666.csv'

@app.route('/analyze_all', methods=['POST'])
def analyze_all():

    try:
        df = pd.read_csv(CSV_FILE_PATH)
        symbol_list = df['Symbol'].tolist()  # Adjust 'symbol' to the actual column name containing symbols
    except Exception as e:
        return f"Error reading CSV file: {e}"
    breakout_symbols = []
    for symbol in symbol_list[:10]:  # Limiting to first 10 for demo purposes
        stock_data = td.fetch_data(symbol)
        if not stock_data.empty:
            breakout_signals = td.analyze_stock(stock_data)
            if not breakout_signals.empty:
                breakout_symbols.append(symbol)

    if breakout_symbols:
        symbols_str = ', '.join(breakout_symbols)
        return render_template('results.html', message=f"Breakout symbols: {symbols_str}")
    else:
        return render_template('results.html', message="No breakout signals found across symbols.")


if __name__ == '__main__':
    app.run(debug=True)
