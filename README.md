# Stock Price Prediction Dashboard

This project uses the Prophet model to predict stock prices and visualize the results in an interactive dashboard. It's designed for educational purposes to demonstrate time series forecasting and data visualization techniques.

## Disclaimer

**IMPORTANT:** This script is for educational purposes only. It is not intended for real-world trading or financial decision-making. The stock market is inherently unpredictable, and this model does not account for many factors that influence stock prices. Using this script for actual trading could lead to financial losses. Always consult with a qualified financial advisor before making investment decisions.

## Features

- Downloads historical stock data using yfinance
- Predicts future stock prices using Facebook's Prophet model
- Visualizes predictions and various components in an interactive dashboard
- Calculates and displays prediction errors

## Requirements

See `requirements.txt` for a full list of dependencies. Main libraries used:

- yfinance
- prophet
- pandas
- numpy
- matplotlib
- seaborn
- plotly
- scikit-learn

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/lordamdal/Basic-Prophet-Stock-Prediction.git
   cd stock-prediction-dashboard
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Open `prediction.py` and modify the `stock_symbol` variable to the stock you want to analyze.
2. Run the script:
   ```
   python prediction.py
   ```
3. The script will generate an interactive dashboard and display it in your default web browser.
4. A CSV file named `forecasted_table.csv` will be saved in the same directory, containing the forecast data.

## Output

The script generates:

1. An interactive dashboard with six plots:
   - Stock Price and Prediction
   - Trend Components
   - Prediction Error Distribution
   - Weekly Seasonality
   - Yearly Seasonality
   - Trend and Changepoints

2. Printed Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) in the console.

3. A CSV file (`forecasted_table.csv`) with the forecasted values.

## Limitations

- The model's accuracy is not guaranteed and can vary significantly depending on the stock and market conditions.
- It does not account for external factors such as company news, economic indicators, or market sentiment.
- The prediction is based solely on historical price data and does not consider fundamental or technical analysis.

## Contributing

Contributions to improve the model or extend its capabilities are welcome. Please ensure that any modifications maintain the educational nature of the project and include appropriate disclaimers.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
