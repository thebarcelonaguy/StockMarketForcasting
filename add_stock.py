import mysql.connector
from mysql.connector import errorcode
import yfinance as yf
import pandas as pd

def create_portfolio_table():
    try:
        cnx = mysql.connector.connect(user="root", password="aszx")
        cursor = cnx.cursor()
        cursor.execute("CREATE DATABASE IF NOT EXISTS stocksDB DEFAULT CHARACTER SET 'utf8'")
        cursor.execute("USE stocksDB")

        # Creating the 'portfolios' table
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS `portfolios` ("
            "  `Name` varchar(50) NOT NULL,"
            "  `CreationDate` date NOT NULL,"
            "  PRIMARY KEY (`Name`)"
            ") ENGINE=InnoDB"
        )

    except mysql.connector.Error as err:
        print(f"Failed creating table: {err}")
    finally:
        cursor.close()
        cnx.close()

#PLEASE RUN THIS FOR THE FIRST TIME

create_portfolio_table()

def connect_database():
    try:
        cnx = mysql.connector.connect(user="root", password="aszx", database="stocksDB")
        return cnx
    except mysql.connector.Error as err:
        print(f"Error connecting to database: {err}")
        exit(1)


def display_portfolios():
    cnx = connect_database()
    cursor = cnx.cursor()
    try:
        cursor.execute("SELECT Name, CreationDate FROM portfolios")
        portfolios = cursor.fetchall()
        if not portfolios:
            print("No portfolios found.")
            return
        for (name, creation_date) in portfolios:
            print(f"\nPortfolio: {name}, Created on: {creation_date}")
            cursor.execute(f"SELECT Symbol FROM `{name}`")
            stocks = [stock[0] for stock in cursor.fetchall()]
            print(f"Stocks: {', '.join(stocks) if stocks else 'No stocks'}")
    except mysql.connector.Error as err:
        print(f"Failed to display portfolios: {err}")
    finally:
        cursor.close()
        cnx.close()

def create_portfolio(portfolio_name):
    cnx = connect_database()
    cursor = cnx.cursor()
    try:
        cursor.execute(f"CREATE TABLE `{portfolio_name}` ("
                       f"  `Symbol` varchar(10) NOT NULL,"
                       f"  PRIMARY KEY (`Symbol`)"
                       f") ENGINE=InnoDB")

        cursor.execute("INSERT INTO portfolios (Name, CreationDate) VALUES (%s, CURDATE())", (portfolio_name,))
        cnx.commit()
        print(f"Portfolio '{portfolio_name}' created successfully.")
    except mysql.connector.Error as err:
        print(f"Error creating portfolio: {err}")
    finally:
        cursor.close()
        cnx.close()

def add_stock_to_portfolio(portfolio_name, stock_symbol):
    if not is_stock_valid(stock_symbol):
        print(f"Stock symbol {stock_symbol} is not valid.")
        return
    cnx = connect_database()
    cursor = cnx.cursor()
    try:
        cursor.execute(f"INSERT INTO `{portfolio_name}` (Symbol) VALUES (%s)", (stock_symbol,))
        cnx.commit()
        print(f"Stock {stock_symbol} added to portfolio {portfolio_name}.")
    except mysql.connector.Error as err:
        print(f"Failed to add stock: {err}")
    finally:
        cursor.close()
        cnx.close()

def remove_stock_from_portfolio(portfolio_name, stock_symbol):
    cnx = connect_database()
    cursor = cnx.cursor()
    try:
        cursor.execute(f"DELETE FROM `{portfolio_name}` WHERE Symbol = %s", (stock_symbol,))
        cnx.commit()
        print(f"Stock {stock_symbol} removed from portfolio {portfolio_name}.")
    except mysql.connector.Error as err:
        print(f"Failed to remove stock: {err}")
    finally:
        cursor.close()
        cnx.close()

def display_portfolios():
    cnx = connect_database()
    cursor = cnx.cursor()
    try:
        cursor.execute("SELECT Name, CreationDate FROM portfolios")
        portfolios = cursor.fetchall()
        for (name, creation_date) in portfolios:
            cursor.execute(f"SELECT Symbol FROM `{name}`")
            stocks = [stock[0] for stock in cursor.fetchall()]
            print(f"Portfolio: {name}, Created on: {creation_date}, Stocks: {stocks}")
    except mysql.connector.Error as err:
        print(f"Failed to display portfolios: {err}")
    finally:
        cursor.close()
        cnx.close()

def is_stock_valid(stock_symbol):
    stock = yf.Ticker(stock_symbol)
    hist = stock.history(period="1d")
    return not hist.empty



def fetch_stock_data_for_portfolio(portfolio_name, start_date, end_date):
    cnx = connect_database()
    cursor = cnx.cursor()
    try:
        cursor.execute(f"SELECT Symbol FROM `{portfolio_name}`")
        stocks = [row[0] for row in cursor.fetchall()]

        if not stocks:
            print(f"No stocks found in portfolio: {portfolio_name}")
            return None
        stock_data = yf.download(stocks, start=start_date, end=end_date)

        if len(stocks) == 1:
            stock_data.columns = pd.MultiIndex.from_product([stock_data.columns, [stocks[0]]])

        return stock_data, stocks
    except mysql.connector.Error as err:
        print(f"Error fetching data from database: {err}")
        return None
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return None
    finally:
        cursor.close()
        cnx.close()


 
