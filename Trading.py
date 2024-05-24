import yfinance as yf
import pandas_ta as ta


def fetch_fundamental_data(symbol):
    """Fetch basic fundamental data for the symbol."""
    stock = yf.Ticker(symbol)

    try:
        #earnings = stock.earnings
        financials = stock.financials
        print("DDDDDD::: ",financials)

        pe_ratio = stock.info.get('forwardPE', None)  # Using forward PE ratio

        # Check for positive earnings growth
        earnings_growth = 31.31
        #if len(earnings) > 1:  # Ensure there are at least two years of earnings data
            #earnings_growth = earnings.iloc[-1]['Earnings'] > earnings.iloc[-2]['Earnings']

        return earnings_growth, pe_ratio
    except KeyError:
        print(f"Data for {symbol} not found or incomplete.")
        return False, None


import pandas as pd

def analyze_stock(stock_data):
    """Analyze stock data with technical indicators."""
    # Calculate moving averages
    stock_data.fillna(method='ffill', inplace=True)  # Forward fill NaN values
    stock_data.dropna(inplace=True)  # Drop rows with NaN values
    stock_data['MA50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['MA200'] = stock_data['Close'].rolling(window=200).mean()

    # Calculate RSI
    stock_data['RSI'] = ta.rsi(stock_data['Close'], length=14)

    # Calculate MACD and its signal line
    macd = ta.macd(stock_data['Close'])
    stock_data = pd.concat([stock_data, macd], axis=1)  # This assumes MACD returns a DataFrame

    # Calculate Bollinger Bands
    bbands = ta.bbands(stock_data['Close'])
    stock_data = pd.concat([stock_data, bbands], axis=1)  # This assumes BBands returns a DataFrame

    # Calculate Average True Range (ATR) for volatility
    stock_data['ATR'] = ta.atr(stock_data['High'], stock_data['Low'], stock_data['Close'])

    # Identify Volume Increase
    stock_data['Volume_Increase'] = stock_data['Volume'] > stock_data['Volume'].shift(1)
    bbands = ta.bbands(stock_data['Close'])

    # Assuming you concatenate bbands to stock_data if not directly added
    stock_data = pd.concat([stock_data, bbands], axis=1)
    stock_data['ATR'].fillna(method='ffill', inplace=True)  # Forward fill first
    stock_data['ATR'].fillna(method='bfill', inplace=True)  # Then backward fill

    # Print column names to identify Bollinger Bands columns
    #print(stock_data.columns)
    # Define Stronger Breakout Condition
    #print((stock_data['Close'] > stock_data['MA50']).head())
   # print((stock_data['Close'] > stock_data['MA200']).head())
    # Continue this for each condition
    print(stock_data.index)
    try:
        condition1 = (stock_data['Close'] > stock_data['MA50'])
        print("Condition 1 OK")
        condition2 = (stock_data['Close'] > stock_data['MA200'])
        print("Condition 2 OK")

        condition3 = (stock_data['Volume_Increase'])
        print("Condition 3 OK")

        condition4 = (stock_data['RSI'] > 70)
        print("Condition 4 OK")

        condition5 = (stock_data['MACD_12_26_9'] > stock_data['MACDs_12_26_9'])
        print("Condition 5 OK")
        bbands = ta.bbands(stock_data['Close'], length=5, std=2.0)
        stock_data = pd.concat([stock_data, bbands], axis=1)

        print(stock_data[['Close', 'BBU_5_2.0']].isna().sum())
        stock_data.dropna(subset=['BBU_5_2.0'], inplace=True)
        print(stock_data[['Close', 'BBU_5_2.0']].isna().sum())
        print(stock_data.columns)
        stock_data = stock_data.loc[:, ~stock_data.columns.duplicated()]
        print(stock_data['BBU_5_2.0'].describe())
        print("NaN count in BBU_5_2.0:", stock_data['BBU_5_2.0'].isna().sum())

        print("Close Dtype: ",stock_data['Close'].dtype)
        print("BBU_5_2 Type: ",stock_data['BBU_5_2.0'].dtype)

        print(stock_data['Close'].dtype, stock_data['BBU_5_2.0'].dtype)
        print("Data size:", stock_data.shape)
        print(stock_data[['Close', 'BBU_5_2.0']].head())

        condition6 = (stock_data['Close'] >= stock_data['BBU_5_2.0'])
        print("Condition 6 OK")

        condition6 = (stock_data['Close'] >= stock_data['BBU_5_2.0'])
        print("Condition 6 OK")
        condition7 = (stock_data['ATR'].shift(1) < stock_data['ATR'])
        print("Condition 7 OK")

        # Continue this for each part of the breakout_condition
    except Exception as e:
        print("An error occurred in condition evaluation:", e)



    breakout_condition = (
            (stock_data['Close'] > stock_data['MA50']) &
            (stock_data['Close'] > stock_data['MA200']) &
            (stock_data['Volume_Increase']) &
            (stock_data['RSI'] > 70) &  # Higher threshold for RSI
            (stock_data['MACD_12_26_9'] > stock_data['MACDs_12_26_9']) &
            (stock_data['Close'] >= stock_data['BBU_5_2.0']) &  # Adjusted to correct Bollinger Band Upper column name
            (stock_data['ATR'].shift(1) < stock_data['ATR'])  # Increasing ATR indicates rising volatility
    )
    print("zxczxc")
    print(breakout_condition.index)
    print("Indexes are equal:", stock_data.index.equals(breakout_condition.index))

    breakout_signals = stock_data[breakout_condition]
    return breakout_signals


def fetch_data(symbol):
    """Fetch historical data for the symbol."""
    # Define the date range for historical data. Adjust as needed.
    start_date = "2015-01-01"
    end_date = "2024-04-05"

    # Download the historical data using yfinance
    data = yf.download(symbol+".NS", start=start_date, end=end_date)

    # Check if the data is empty (symbol may not exist or network issues)
    if data.empty:
        print(f"No data found for {symbol}")
        return None

    return data