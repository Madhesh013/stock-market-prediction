import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import datetime as dt

# Dictionary of popular companies with their ticker symbols and sectors
COMPANIES = {
    'AAPL': {'name': 'Apple Inc.', 'sector': 'Technology'},
    'MSFT': {'name': 'Microsoft Corporation', 'sector': 'Technology'},
    'AMZN': {'name': 'Amazon.com Inc.', 'sector': 'Consumer Cyclical'},
    'GOOGL': {'name': 'Alphabet Inc. (Google)', 'sector': 'Communication Services'},
    'META': {'name': 'Meta Platforms Inc. (Facebook)', 'sector': 'Communication Services'},
    'TSLA': {'name': 'Tesla Inc.', 'sector': 'Automotive'},
    'NVDA': {'name': 'NVIDIA Corporation', 'sector': 'Technology'},
    'JPM': {'name': 'JPMorgan Chase & Co.', 'sector': 'Financial Services'},
    'V': {'name': 'Visa Inc.', 'sector': 'Financial Services'},
    'WMT': {'name': 'Walmart Inc.', 'sector': 'Consumer Defensive'},
    'JNJ': {'name': 'Johnson & Johnson', 'sector': 'Healthcare'},
    'PG': {'name': 'Procter & Gamble Co.', 'sector': 'Consumer Defensive'},
    'DIS': {'name': 'The Walt Disney Company', 'sector': 'Communication Services'},
    'NFLX': {'name': 'Netflix Inc.', 'sector': 'Communication Services'},
    'KO': {'name': 'The Coca-Cola Company', 'sector': 'Consumer Defensive'},
    'PEP': {'name': 'PepsiCo Inc.', 'sector': 'Consumer Defensive'},
    'INTC': {'name': 'Intel Corporation', 'sector': 'Technology'},
    'AMD': {'name': 'Advanced Micro Devices Inc.', 'sector': 'Technology'},
    'BAC': {'name': 'Bank of America Corporation', 'sector': 'Financial Services'},
    'XOM': {'name': 'Exxon Mobil Corporation', 'sector': 'Energy'},
    'CVX': {'name': 'Chevron Corporation', 'sector': 'Energy'},
    'NKE': {'name': 'Nike Inc.', 'sector': 'Consumer Cyclical'},
    'MCD': {'name': 'McDonald\'s Corporation', 'sector': 'Consumer Cyclical'},
    'SBUX': {'name': 'Starbucks Corporation', 'sector': 'Consumer Cyclical'},
    'COST': {'name': 'Costco Wholesale Corporation', 'sector': 'Consumer Defensive'}
}

# Function to display available companies
def display_companies():
    """
    Display the list of available companies by sector
    """
    print("\n==== Available Companies ====")
    
    # Group companies by sector
    sectors = {}
    for ticker, info in COMPANIES.items():
        sector = info['sector']
        if sector not in sectors:
            sectors[sector] = []
        sectors[sector].append((ticker, info['name']))
    
    # Display companies by sector
    for sector, companies in sorted(sectors.items()):
        print(f"\n{sector}:")
        for ticker, name in sorted(companies):
            print(f"  {ticker}: {name}")
    
    print("\nYou can also enter any other valid stock ticker not in this list.")

# Function to get company ticker
def get_company_ticker():
    """
    Get company ticker from user input
    """
    display_companies()
    
    while True:
        ticker = input("\nEnter stock ticker symbol (e.g., AAPL) or 'list' to see companies again: ").upper()
        
        if ticker == 'LIST':
            display_companies()
            continue
        
        if ticker in COMPANIES:
            print(f"Selected: {COMPANIES[ticker]['name']} ({ticker})")
        else:
            confirm = input(f"'{ticker}' is not in our predefined list. Continue with this ticker? (y/n): ")
            if confirm.lower() != 'y':
                continue
        
        return ticker

# Function to compare multiple companies
def compare_companies(tickers, years=1):
    """
    Compare multiple companies' stock performance and predictions
    """
    if len(tickers) < 2:
        print("Need at least 2 tickers to compare")
        return
    
    # Calculate dates
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=365 * years)
    
    # Initialize results dictionary
    results = {}
    
    # Fetch data and make predictions for each ticker
    for ticker in tickers:
        print(f"\n--- Processing {ticker} ---")
        
        # Get company name if available
        company_name = COMPANIES.get(ticker, {}).get('name', ticker)
        
        # Get stock data
        data = get_stock_data(ticker, start_date, end_date)
        if data is None:
            print(f"Skipping {ticker} due to data retrieval failure")
            continue
        
        # Store closing prices for comparison
        results[ticker] = {
            'name': company_name,
            'data': data,
            'current_price': data['Adj Close'][-1],
            'start_price': data['Adj Close'][0],
            'return': (data['Adj Close'][-1] / data['Adj Close'][0] - 1) * 100
        }
    
    # Plot comparison
    plt.figure(figsize=(16, 8))
    plt.title(f'Stock Price Comparison ({start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")})')
    plt.xlabel('Date')
    plt.ylabel('Normalized Price (Start = 100)')
    
    # Plot normalized prices for easy comparison
    for ticker, result in results.items():
        normalized_price = result['data']['Adj Close'] / result['data']['Adj Close'][0] * 100
        plt.plot(result['data'].index, normalized_price, label=f"{ticker} ({result['name']})")
    
    plt.legend()
    plt.grid(True)
    plt.savefig('stock_comparison.png')
    
    try:
        plt.show()
    except Exception as e:
        print(f"Warning: Could not display plot ({str(e)}), but it was saved to file")
    
    # Print comparison results
    print("\n==== Company Comparison ====")
    print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print("\nPerformance Summary:")
    
    # Sort by return
    sorted_results = sorted(results.items(), key=lambda x: x[1]['return'], reverse=True)
    
    for ticker, result in sorted_results:
        print(f"{ticker} ({result['name']}):")
        print(f"  Starting Price: ${result['start_price']:.2f}")
        print(f"  Current Price: ${result['current_price']:.2f}")
        print(f"  Return: {result['return']:.2f}%")
        print()
    
    print("Comparison chart saved as 'stock_comparison.png'")

# Function to get stock data
def get_stock_data(ticker, start_date, end_date):
    """
    Download stock data from Yahoo Finance
    """
    print(f"Downloading data for {ticker} from {start_date} to {end_date}")
    data = yf.download(ticker, start=start_date, end=end_date)
    print(f"Downloaded {len(data)} rows of data")
    
    # Check if data is empty
    if len(data) == 0:
        print(f"Error: No data available for {ticker}")
        return None
    
    # Check if 'Adj Close' column exists
    if 'Adj Close' not in data.columns:
        print("'Adj Close' column not found. Using 'Close' column instead.")
        # If 'Adj Close' doesn't exist, use 'Close' and rename it
        if 'Close' in data.columns:
            data['Adj Close'] = data['Close']
        else:
            print("Error: Neither 'Adj Close' nor 'Close' columns found in data")
            print("Available columns:", data.columns.tolist())
            return None
    
    return data

# Function to prepare features for prediction
def prepare_features(data, window=14):
    """
    Create technical indicators as features for prediction
    """
    # Check if data is valid
    if data is None:
        print("Error: Invalid data. Cannot prepare features.")
        return None
    
    # Make a copy to avoid modifying original data
    df = data.copy()
    
    # Ensure we have the necessary columns
    required_columns = ['Adj Close', 'Volume']
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: Required column '{col}' not found in data")
            print("Available columns:", df.columns.tolist())
            return None
    
    try:
        # Calculate moving averages
        df['MA5'] = df['Adj Close'].rolling(window=5).mean()
        df['MA20'] = df['Adj Close'].rolling(window=20).mean()
        
        # Calculate price momentum
        df['Price_Change'] = df['Adj Close'].pct_change()
        df['Price_Change_5'] = df['Adj Close'].pct_change(periods=5)
        
        # Calculate volume features (check if Volume exists)
        if 'Volume' in df.columns:
            df['Volume_Change'] = df['Volume'].pct_change()
            df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
        else:
            # If Volume doesn't exist, create dummy features
            df['Volume_Change'] = 0
            df['Volume_MA5'] = 0
            print("Warning: 'Volume' column not found. Using dummy values.")
        
        # Calculate volatility
        df['Volatility'] = df['Price_Change'].rolling(window=window).std()
        
        # Calculate price relative to moving averages
        df['Price_to_MA5'] = df['Adj Close'] / df['MA5']
        df['Price_to_MA20'] = df['Adj Close'] / df['MA20']
        
        # Drop NaN values
        df = df.dropna()
        
        # Target variable: next day's price
        df['Target'] = df['Adj Close'].shift(-1)
        
        # Drop the last row which has NaN target
        df = df[:-1]
        
        return df
    
    except Exception as e:
        print(f"Error in prepare_features: {str(e)}")
        print("Columns in dataframe:", df.columns.tolist())
        return None

# Function to split data and scale features
def split_and_scale_data(df, test_size=0.2):
    """
    Split data into training and testing sets, and scale features
    """
    # Check if data is valid
    if df is None or len(df) == 0:
        print("Error: Invalid or empty dataframe for splitting/scaling")
        return None, None, None, None, None, None, None, None
    
    try:
        # Define features and target
        features = ['Adj Close', 'MA5', 'MA20', 'Price_Change', 'Price_Change_5', 
                    'Volume_Change', 'Volume_MA5', 'Volatility', 
                    'Price_to_MA5', 'Price_to_MA20']
        target = 'Target'
        
        # Check that all features exist
        for feature in features:
            if feature not in df.columns:
                print(f"Error: Feature '{feature}' not found in data")
                return None, None, None, None, None, None, None, None
        
        # Split data
        train_size = int(len(df) * (1 - test_size))
        train_data = df[:train_size]
        test_data = df[train_size:]
        
        # Scale features
        scaler = MinMaxScaler()
        
        # Fit scaler on training data
        X_train = scaler.fit_transform(train_data[features])
        y_train = train_data[target].values
        
        # Transform test data
        X_test = scaler.transform(test_data[features])
        y_test = test_data[target].values
        
        return X_train, y_train, X_test, y_test, train_data, test_data, features, scaler
    
    except Exception as e:
        print(f"Error in split_and_scale_data: {str(e)}")
        return None, None, None, None, None, None, None, None

# Function to train models
def train_models(X_train, y_train):
    """
    Train linear regression and random forest models
    """
    # Check if data is valid
    if X_train is None or y_train is None:
        print("Error: Invalid training data")
        return None, None
    
    try:
        # Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        
        # Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        return lr_model, rf_model
    
    except Exception as e:
        print(f"Error in train_models: {str(e)}")
        return None, None

# Function to evaluate models
def evaluate_models(lr_model, rf_model, X_test, y_test):
    """
    Evaluate models on test data
    """
    # Check if models and data are valid
    if lr_model is None or rf_model is None or X_test is None or y_test is None:
        print("Error: Invalid models or test data")
        return None, None
    
    try:
        # Make predictions
        lr_predictions = lr_model.predict(X_test)
        rf_predictions = rf_model.predict(X_test)
        
        # Calculate RMSE
        lr_rmse = np.sqrt(np.mean((lr_predictions - y_test) ** 2))
        rf_rmse = np.sqrt(np.mean((rf_predictions - y_test) ** 2))
        
        print(f"Linear Regression RMSE: ${lr_rmse:.2f}")
        print(f"Random Forest RMSE: ${rf_rmse:.2f}")
        
        # Choose the better model
        if lr_rmse <= rf_rmse:
            print("Linear Regression model performs better")
            return lr_model, lr_predictions
        else:
            print("Random Forest model performs better")
            return rf_model, rf_predictions
    
    except Exception as e:
        print(f"Error in evaluate_models: {str(e)}")
        return None, None

# Function to predict future prices
def predict_future(model, last_data, features, scaler, days=7):
    """
    Predict future stock prices
    """
    # Check if model and data are valid
    if model is None or last_data is None or features is None or scaler is None:
        print("Error: Invalid model or data for prediction")
        return None
    
    try:
        print(f"Predicting prices for the next {days} days...")
        
        future_prices = []
        current_features = last_data[features].iloc[-1:].values
        
        for _ in range(days):
            # Scale the features
            scaled_features = scaler.transform(current_features)
            
            # Make prediction
            next_price = model.predict(scaled_features)[0]
            future_prices.append(next_price)
            
            # Update features for next prediction
            new_features = current_features.copy()
            
            # Update close price
            new_features[0, 0] = next_price
            
            # Simple updates for other features - this is simplified
            # MA5 update: 80% of old MA5 + 20% of new price
            new_features[0, 1] = 0.8 * new_features[0, 1] + 0.2 * next_price
            
            # MA20 update: 95% of old MA20 + 5% of new price
            new_features[0, 2] = 0.95 * new_features[0, 2] + 0.05 * next_price
            
            # Price change
            new_features[0, 3] = (next_price / current_features[0, 0]) - 1
            
            # Other features - use last values (simplified)
            
            # Update current features for next iteration
            current_features = new_features
        
        return future_prices
    
    except Exception as e:
        print(f"Error in predict_future: {str(e)}")
        return None

# Function to plot results
def plot_results(data, test_data, predictions, future_dates, future_prices, ticker):
    """
    Plot historical data, test predictions, and future predictions
    """
    # Check if data is valid
    if (data is None or test_data is None or predictions is None or 
        future_dates is None or future_prices is None):
        print("Error: Invalid data for plotting")
        return
    
    try:
        # Get company name if available
        company_name = COMPANIES.get(ticker, {}).get('name', ticker)
        
        plt.figure(figsize=(16, 8))
        plt.title(f'{company_name} ({ticker}) - Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price USD ($)')
        
        # Plot historical data
        plt.plot(data.index, data['Adj Close'], color='blue', label='Historical Data')
        
        # Plot test predictions
        plt.plot(test_data.index, predictions, color='red', label='Test Predictions')
        
        # Plot future predictions
        plt.plot(future_dates, future_prices, 'go-', label='Future Predictions')
        
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{ticker}_prediction.png')  # Save the figure
        print(f"Plot saved as '{ticker}_prediction.png'")
        
        # Try to display the plot
        try:
            plt.show()
        except Exception as e:
            print(f"Warning: Could not display plot ({str(e)}), but it was saved to file")
    
    except Exception as e:
        print(f"Error in plot_results: {str(e)}")

# Main function to run the entire process
def main():
    try:
        print("\n===== Stock Market Prediction Tool =====")
        print("1. Single Stock Analysis")
        print("2. Compare Multiple Stocks")
        
        choice = input("\nSelect an option (1 or 2): ")
        
        if choice == '2':
            # Compare multiple stocks
            print("\nYou'll be able to select multiple stocks to compare")
            tickers = []
            
            while True:
                ticker = get_company_ticker()
                tickers.append(ticker)
                
                add_more = input("\nAdd another company to compare? (y/n): ")
                if add_more.lower() != 'y':
                    break
            
            years = int(input("\nEnter number of years of historical data to use (1-5): "))
            if years < 1 or years > 10:
                years = min(max(years, 1), 10)  # Clamp between 1 and 10
                print(f"Using {years} years of historical data")
            
            compare_companies(tickers, years)
            return
        
        # Single stock analysis
        ticker = get_company_ticker()
        years = int(input("\nEnter number of years of historical data to use (1-5): "))
        
        # Validate input
        if years < 1 or years > 10:
            years = min(max(years, 1), 10)  # Clamp between 1 and 10
            print(f"Using {years} years of historical data")
        
        # Calculate dates
        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=365 * years)
        
        # Get stock data
        data = get_stock_data(ticker, start_date, end_date)
        if data is None:
            print("Failed to retrieve stock data. Exiting.")
            return
        
        # Print data information
        print(f"Data shape: {data.shape}")
        
        # Prepare features
        df = prepare_features(data)
        if df is None:
            print("Failed to prepare features. Exiting.")
            return
        
        # Split and scale data
        split_result = split_and_scale_data(df)
        X_train, y_train, X_test, y_test, train_data, test_data, features, scaler = split_result
        
        if X_train is None:
            print("Failed to split and scale data. Exiting.")
            return
        
        # Train models
        lr_model, rf_model = train_models(X_train, y_train)
        if lr_model is None or rf_model is None:
            print("Failed to train models. Exiting.")
            return
        
        # Evaluate models
        eval_result = evaluate_models(lr_model, rf_model, X_test, y_test)
        if eval_result is None:
            print("Failed to evaluate models. Exiting.")
            return
        
        best_model, predictions = eval_result
        
        # Predict future prices
        days = 7  # Predict for next 7 days
        future_prices = predict_future(best_model, test_data, features, scaler, days)
        if future_prices is None:
            print("Failed to predict future prices. Exiting.")
            return
        
        # Create future dates
        last_date = data.index[-1]
        future_dates = [last_date + dt.timedelta(days=i+1) for i in range(days)]
        
        # Plot results
        plot_results(data, test_data, predictions, future_dates, future_prices, ticker)
        
        # Print future predictions
        company_name = COMPANIES.get(ticker, {}).get('name', ticker)
        print(f"\nFuture Price Predictions for {company_name} ({ticker}):")
        for i in range(len(future_dates)):
            print(f"{future_dates[i].strftime('%Y-%m-%d')}: ${future_prices[i]:.2f}")
        
        print("\nNote: These predictions are based on historical patterns and should not be used as financial advice.")
    
    except Exception as e:
        print(f"Unexpected error in main: {str(e)}")

if __name__ == "__main__":
    main()