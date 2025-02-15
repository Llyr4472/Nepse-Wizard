from Stock import *
import pandas as pd
from datetime import datetime
import pandas_ta as ta
from termcolor import cprint
import sys

def macd(df, lag=0):
    """
    Calculates the MACD (Moving Average Convergence Divergence) signal for a given DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the 'close' column.
        lag (int): The number of periods to look back for the MACD signal.

    Returns:
        int: The MACD signal, where 1 indicates a buy signal, -1 indicates a sell signal, and 0 indicates no signal.
    """
    if len(df) < 26 + 9:  # Ensure sufficient data
        return 0
    
    # Calculate MACD
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    hist_col = 'MACDh_12_26_9'
    
    signal = 0
    # Check for crossovers in the last (lag+1) periods
    for i in range(-1 - lag, -1 + 1):
        if i < -len(df) or i >= 0:
            continue  # Skip out-of-bounds indices
        
        # Bullish crossover (histogram crosses above zero)
        if df[hist_col].iloc[i] > 0 and df[hist_col].iloc[i-1] <= 0:
            signal = 1
            break
        
        # Bearish crossover (histogram crosses below zero)
        if df[hist_col].iloc[i] < 0 and df[hist_col].iloc[i-1] >= 0:
            signal = -1
            break
    
    return signal

def rsi(df, lag=0):
    """
    Calculates the Relative Strength Index (RSI) for the given DataFrame `df` and returns a signal based on the RSI values.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the 'close' column.
        lag (int): The number of periods to look back when calculating the signal.

    Returns:
        int: The RSI signal, where -1 indicates a sell signal, 0 indicates no signal, and 1 indicates a buy signal.
    """
    rsi_length = 14
    if len(df) < rsi_length + lag:
        return 0
    
    df['RSI'] = ta.rsi(df['close'], length=rsi_length)
    
    buy_signal = False
    sell_signal = False
    
    # Check last (lag+1) periods
    for i in range(-1 - lag, -1 + 1):
        if i < -len(df) or i >= 0:
            continue
        
        prev_i = max(i-1, -len(df))
        current_rsi = df['RSI'].iloc[i]
        prev_rsi = df['RSI'].iloc[prev_i]
        
        # Cross above 30
        if current_rsi >= 30 and prev_rsi < 30:
            buy_signal = True
        
        # Cross below 70
        if current_rsi <= 70 and prev_rsi > 70:
            sell_signal = True
    
    if buy_signal and not sell_signal:
        return 1
    elif sell_signal and not buy_signal:
        return -1
    return 0

def ema_crossover(df, lag=0):
    """
    Calculates the Exponential Moving Average (EMA) crossover signal for a given DataFrame `df`.

    The function calculates the EMA for various periods (3, 5, 8, 10, 12, 15, 30, 35, 40, 45, 50, 60) and checks if the minimum of the shorter EMAs is greater than the maximum of the longer EMAs. This is used as a signal for a potential buy or sell decision.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the 'close' column.
        lag (int, optional): The number of periods to look back for the EMA crossover signal. Defaults to 1.

    Returns:
        int: 1 if the EMA crossover signal is a buy signal, -1 if the signal is a sell signal, 0 if there is no signal.
    """
    short_period = 9
    long_period = 21
    if len(df) < long_period + lag:
        return 0
    
    df.ta.ema(length=short_period, append=True)
    df.ta.ema(length=long_period, append=True)
    
    short_ema = f'EMA_{short_period}'
    long_ema = f'EMA_{long_period}'
    
    signal = 0
    for i in range(-1 - lag, -1 + 1):
        if i < -len(df) or i >= 0:
            continue
        
        prev_i = max(i-1, -len(df))
        # Bullish crossover
        if df[short_ema].iloc[i] > df[long_ema].iloc[i] and \
           df[short_ema].iloc[prev_i] <= df[long_ema].iloc[prev_i]:
            signal = 1
            break
        
        # Bearish crossover
        if df[short_ema].iloc[i] < df[long_ema].iloc[i] and \
           df[short_ema].iloc[prev_i] >= df[long_ema].iloc[prev_i]:
            signal = -1
            break
    
    return signal

def bbands(df, lag=0):
    """
    Calculates the Bollinger Bands (BBands) for the given DataFrame `df` and returns a signal based on the current price relative to the BBands.

    Args:
        df (pandas.DataFrame): The DataFrame containing the price data.

    Returns:
        int: 1 if the current price is below the lower BBand, -1 if the current price is above the upper BBand, 0 otherwise.
    """
    bb_length = 20
    if len(df) < bb_length + lag:
        return 0
    
    df.ta.bbands(length=bb_length, std=2, append=True)
    df['BB_%b'] = (df['close'] - df[f'BBL_{bb_length}_2.0']) / \
                 (df[f'BBU_{bb_length}_2.0'] - df[f'BBL_{bb_length}_2.0'])
    
    signal = 0
    for i in range(-1 - lag, -1 + 1):
        if i < -len(df) or i >= 0:
            continue
        
        prev_i = max(i-1, -len(df))
        current = df['BB_%b'].iloc[i]
        prev = df['BB_%b'].iloc[prev_i]
        
        # Buy signal: %b crosses above 0
        if current > 0 and prev <= 0:
            signal = 1
            break
        
        # Sell signal: %b crosses below 1
        if current < 1 and prev >= 1:
            signal = -1
            break
    
    return signal

def hybrid(df, lag=0):
    """
    Calculates a hybrid trading signal based on the combination of EMA crossover, MACD, RSI, and Bollinger Bands indicators.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the stock data.
        lag (int, optional): The number of periods to lag the indicators. Defaults to 0.

    Returns:
        int: The hybrid trading signal, where 1 indicates a buy signal, -1 indicates a sell signal, and 0 indicates a neutral signal.
    """
    df_copy = df.copy()
    
    signals = {
        'macd': macd(df_copy, lag),
        'ema': ema_crossover(df_copy, lag),
        'rsi': rsi(df_copy, lag),
        'bbands': bbands(df_copy, lag)
    }
    
    # Weighted scoring system
    weights = {
        'macd': 2,    # Most reliable
        'ema': 2,     # Important trend indicator
        'rsi': 1,     # Momentum confirmation
        'bbands': 1   # Volatility confirmation
    }
    
    total_score = sum(signals[indicator] * weights[indicator] 
                     for indicator in signals)
    
    # Require strong consensus for signals
    if total_score >= 3:
        return 1
    elif total_score <= -3:
        return -1
    return 0


def main():
    
    # Get the stock ticker symbol from the command line
    if len(sys.argv)>1 and sys.argv[1] != 'all':
        stocks = [Stock(sys.argv[1])]
    else:
        stocks = [Stock(x) for x in COMPANIES if Stock(x).trade]
    
    # Get the lag value from the command line
    if len(sys.argv)>2:
        lag = int(sys.argv[2])
    else:
        lag = 2
    
    # Get the intended action from the command line
    if len(sys.argv)>3 and sys.argv[3] in ['buy','sell']:
        intent = sys.argv[3]
    else:
        intent = None
    
    with open('data/info.json') as f:
        date = json.load(f)['updated_on']
    
    print(f"Lag: {lag}\tintent: {intent}\t Date:{date}")
    print("Stock\tMACD\tEMA\tRSI\tBBands\tAction\tChart")
    for stock in stocks:
        df =  pd.read_csv(stock.file)
        macd_value = macd(df,lag=lag)
        ema_value = ema_crossover(df,lag=lag)
        rsi_value = rsi(df,lag=lag)
        bbands_value = bbands(df,lag=lag)  # Added lag parameter here
        sum = ema_value + macd_value + rsi_value + bbands_value 
        action = "buy" if sum>=2 else "sell" if sum < -2 else ''
        if intent == None or intent == action:
            cprint(f'{stock()}\t{macd_value}\t{ema_value}\t{rsi_value}\t{bbands_value}\t{action}\thttps://nepsealpha.com/trading/chart?symbol={stock.symbol}',color='green' if sum>=2 else'red' if sum<=-2 else 'white')

if __name__ == "__main__":
    main()