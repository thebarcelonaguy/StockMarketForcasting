from add_stock import (
    create_portfolio,
    remove_stock_from_portfolio,
    add_stock_to_portfolio,
)
from add_stock import fetch_stock_data_for_portfolio
from add_stock import display_portfolios
import argparse
import yfinance as yf
import warnings
from LSTM import predict_signals, add_stock, delete_stock
import os

warnings.filterwarnings(
    "ignore", message="The 'unit' keyword in TimedeltaIndex construction is deprecated"
)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow warnings


def main():
    parser = argparse.ArgumentParser(description="Stock Portfolio Management")

    # Adding arguments for different functionalities
    parser.add_argument(
        "--create",
        help="Create a new portfolio",
        type=str,
        metavar=("<PORTFOLIO_NAME>"),
    )
    parser.add_argument(
        "--add",
        help="Add a stock to a portfolio",
        nargs=2,
        metavar=("<PORTFOLIO_NAME>", "<STOCK_NAME>"),
    )
    parser.add_argument(
        "--remove",
        help="Remove a stock from a portfolio",
        nargs=2,
        metavar=("<PORTFOLIO_NAME>", "<STOCK_NAME>"),
    )
    parser.add_argument("--display", help="Display all portfolios", action="store_true")
    parser.add_argument(
        "--fetch",
        help="Fetch stock price data for a portfolio for a date range [YYYY-MM-DD]",
        nargs=3,
        metavar=("<PORTFOLIO_NAME>", "<START_DATE>", "<END_DATE>"),
    )
    parser.add_argument(
        "--predict_signals",
        help="Predict buy/sell signals for a stock for a date range [YYYY-MM-DD]",
        nargs=4,
        metavar=("<PORTFOLIO_NAME>", "<START_DATE>", "<END_DATE>", "<BUDGET>"),
    )

    args = parser.parse_args()

    if args.create:
        create_portfolio(args.create)
    elif args.add:
        portfolio, stock = args.add
        add_stock_to_portfolio(portfolio, stock.upper())
    elif args.remove:
        portfolio, stock = args.remove
        remove_stock_from_portfolio(portfolio, stock.lower())
    elif args.display:
        display_portfolios()
    elif args.predict_signals:
        portfolio, start_date, end_date, budget = args.predict_signals
        stocks_data, stock = fetch_stock_data_for_portfolio(
            portfolio, start_date, end_date
        )
        predict_signals(stock, start_date, end_date, budget)
    elif args.fetch:
        portfolio, start_date, end_date = args.fetch
        data, stocks = fetch_stock_data_for_portfolio(portfolio, start_date, end_date)
        if data is not None and not data.empty:
            stock_symbols = data.columns.levels[1]
            for symbol in stock_symbols:
                # Select data for the current symbol
                stock_data = data.loc[:, data.columns.get_level_values(1) == symbol]
                print(f"\nData for {symbol}:")
                print(stock_data)
        else:
            print("No data to display for this portfolio.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
