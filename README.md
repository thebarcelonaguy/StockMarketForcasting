Running the Code
Before you start, ensure you have all the necessary packages installed:

Run pip install -r requirements.txt to install the required packages.
Important Setup Instructions
MySQL Password: Please update the root password in the add_stock.py file to your MySQL root password. This needs to be done on line 8 and line 34.
Command Line Interface Commands
The application provides a command line interface with the following commands:

Help: --help - Show the help message and exit.
Create a New Portfolio: --create <PORTFOLIO_NAME>
Add a Stock to a Portfolio: --add <PORTFOLIO_NAME> <STOCK_NAME>
Remove a Stock from a Portfolio: --remove <PORTFOLIO_NAME> <STOCK_NAME>
Display All Portfolios: --display - Display all portfolios.
Fetch Stock Price Data: --fetch <PORTFOLIO_NAME> <START_DATE> <END_DATE>
Predict Buy/Sell Signals: --predict_signals <PORTFOLIO_NAME> <START_DATE> <END_DATE> <BUDGET_IN_USD>
Examples of How to Run Commands
To utilize the features of this program, follow the steps below:

Step 1: Check Python Installation
Ensure Python is installed on your system by typing python3 --version in the Terminal.
Step 2: Running Your Python Program
To run your program, type python3 stock_cli.py [options] in the terminal, replacing [options] with the specific commands you want to use.
Using the Program's Features
Here are detailed instructions on how to use each feature:

Create a New Portfolio: python3 stock_cli.py --create MyPortfolio
Add a Stock to a Portfolio: python3 stock_cli.py --add MyPortfolio AAPL
Remove a Stock from a Portfolio: python3 stock_cli.py --remove MyPortfolio AAPL
Display All Portfolios: python3 stock_cli.py --display
Fetch Stock Price Data: python3 stock_cli.py --fetch MyPortfolio 2014-01-01 2020-01-01
Predict Buy/Sell Signals: python3 stock_cli.py --predict_signals MyPortfolio 2014-01-01 2020-01-01 1000
Note: Replace python3 with python if that's how Python is invoked on your system.

Additional Information
You can only add and delete one stock at a time with the commands.
From our implemented models, the LSTM model performed better than Arima and moving averages.
Model Performance
After running the code, the data will be saved into two CSV files: trade_history.csv and portfolio_history.csv.

We create an LSTM model for every stock in the portfolio. The performance metrics used are:

RMSE (Root Mean Square Error): Measures the square root of the average squared differences between predicted and actual values, emphasizing larger errors.
MAE (Mean Absolute Error): Calculates the average absolute differences between predicted and actual values, representing the average prediction error.
Best Model Performance
Mean Absolute Error (MAE): 0.4978713148424343
Root Mean Square Error (RMSE): 0.49836931196550915
