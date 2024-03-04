import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping
from keras.regularizers import l1_l2
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

tickers = []


def rsi(df, column="close", period=14):
    delta = df[column].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def macd(df, fast_period=12, slow_period=26, signal_period=9, column="close"):
    fast_ma = df[column].ewm(span=fast_period, adjust=False).mean()
    slow_ma = df[column].ewm(span=slow_period, adjust=False).mean()
    macd_line = fast_ma - slow_ma
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line

def stochastic_oscillator(df, k_period=14, d_period=3, column="close"):
    low_min = df["low"].rolling(window=k_period).min()
    high_max = df["high"].rolling(window=k_period).max()
    k = 100 * (df[column] - low_min) / (high_max - low_min)
    d = k.rolling(window=d_period).mean()
    return k, d

def bollinger_bands(df, window=20, no_of_stds=2):
    rolling_mean = df['close'].rolling(window=window).mean()
    rolling_std = df['close'].rolling(window=window).std()
    df['bollinger_upper'] = rolling_mean + (rolling_std * no_of_stds)
    df['bollinger_lower'] = rolling_mean - (rolling_std * no_of_stds)

def add_stock(stock_name):
    tickers.append(stock_name)

def delete_stock(stock_name):
    tickers.remove(stock_name)


def predict_signals(stocks, start_date, end_date, funds):
    start_date = start_date
    end_date =  end_date
    scaler = StandardScaler()
    signals_data = {}
    tickers = stocks

    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date)
        data.columns = data.columns.str.lower()
        data['rsi'] = rsi(data)
        data['macd'], data['macdsignal'], data['macdhist'] = macd(data)
        data['k'], data['d'] = stochastic_oscillator(data)
        bollinger_bands(data)
        data.dropna(inplace=True)
        features = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'macdhist', 'k', 'd', 'bollinger_upper', 'bollinger_lower']
        X = data[features]
        y = np.where(data['close'].shift(-1) > data['close'], 1, 0)
        X_scaled = scaler.fit_transform(X)
        X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        split = int(len(X_scaled) * 0.7)
        X_train, X_test = X_scaled[:split], X_scaled[split:]
        y_train, y_test = y[:split], y[split:]
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]), kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
            Dropout(0.2),
            LSTM(units=50),
            Dropout(0.2),
            Dense(units=1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0, callbacks=[early_stopping])
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        rmse = sqrt(mean_squared_error(y_test, predictions))

        print(f'Mean Absolute Error (MAE): {mae}')
        print(f'Root Mean Square Error (RMSE): {rmse}')
        signals = (predictions > 0.5).astype(int).flatten()
        data['signal'] = np.nan
        data.iloc[split:, data.columns.get_loc('signal')] = signals
        signals_data[ticker] = data

        
    for ticker in tickers:
        data = signals_data[ticker]
        
        # Assume 'data' already includes technical indicators used for signal generation
        # Let's simplify by saying a buy signal is generated based on a specific condition, 
        # and similarly for sell, storing the close price as the intended trade price
        
        # Generate signals and store trade price
        data['trade_signal'] = 0  # Default to no action
        data['trade_price'] = np.nan
        
        # Example signal generation (simplified for demonstration)
        data.loc[data['rsi'] < 30, 'trade_signal'] = 1  # Buy signal on low RSI
        data.loc[data['rsi'] > 70, 'trade_signal'] = -1  # Sell signal on high RSI
        data.loc[data['trade_signal'] != 0, 'trade_price'] = data['close']
        
        signals_data[ticker] = data
        
    #change funds    
    initial_funds = float(funds)
    portfolio_value = initial_funds
    holdings = {ticker: 0 for ticker in tickers}
    trade_history = []
    portfolio_history = []

    # Use a union of all dates from all tickers to ensure we cover every trading opportunity
    all_dates = sorted(set.union(*(set(signals_data[ticker].index) for ticker in tickers)))

    for date in all_dates:
        for ticker in tickers:
            if date in signals_data[ticker].index:
                row = signals_data[ticker].loc[date]
                signal = row['trade_signal']
                trade_price = row['trade_price']
                
                if pd.notnull(signal) and pd.notnull(trade_price):
                    if signal == 1 and portfolio_value >= trade_price:  # Buy condition
                        shares_to_buy = int(portfolio_value // trade_price)
                        if shares_to_buy > 0:
                            portfolio_value -= shares_to_buy * trade_price
                            holdings[ticker] += shares_to_buy
                            trade_history.append({'Date': date, 'Ticker': ticker, 'Action': 'BUY', 'Shares': shares_to_buy, 'Price': trade_price})
                            # print(f"Bought {shares_to_buy} shares of {ticker} at {trade_price} on {date}")
                    elif signal == -1 and holdings[ticker] > 0:  # Sell condition
                        portfolio_value += holdings[ticker] * trade_price
                        # print(f"Sold {holdings[ticker]} shares of {ticker} at {trade_price} on {date}")
                        trade_history.append({'Date': date, 'Ticker': ticker, 'Action': 'SELL', 'Shares': holdings[ticker], 'Price': trade_price})
                        holdings[ticker] = 0
        
        # Daily portfolio value update
        daily_value = portfolio_value + sum(holdings[ticker] * signals_data[ticker].loc[date, 'close'] if date in signals_data[ticker].index else 0 for ticker in tickers)
        portfolio_history.append({'Date': date, 'Portfolio Value': daily_value})

    # Convert trade and portfolio history to DataFrames and save to CSV
    trade_history_df = pd.DataFrame(trade_history)
    portfolio_history_df = pd.DataFrame(portfolio_history)
    trade_history_df.to_csv('trade_history.csv', index=False)
    portfolio_history_df.to_csv('portfolio_history.csv', index=False)

    print("Trading and portfolio history have been saved to csv files namely trade_history and portfolio_history.")


    final_value = portfolio_history[-1]['Portfolio Value']
    total_days = len(portfolio_history)
    years = total_days / 252
    annual_return = ((final_value / initial_funds) ** (1 / years)) - 1
    daily_returns = portfolio_history_df['Portfolio Value'].pct_change().dropna()
    sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)

    print(f"Final Portfolio Value: ${final_value:.2f}")
    print(f"Annualized Return: {annual_return * 100:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")





