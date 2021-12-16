![enter image description here](https://raw.githubusercontent.com/Deniz-Jasa/stock_prediction_model/main/images/predictCropped.png)

  # ✨ Inspiration

Wanting to find success in the stock market that would be backed up by historical data and relatively accurate predictions in order to mitigate risks and increase return rates.

    

#  📈 What Predict does

**Predict** is a stock prediction model that uses a technical analysis approach to predict a stock's daily price movement by examining trends and daily price movements the model uses this data to attempt to predict future stock price movements.
Four indicators that can give signals about the next price movement:
- **Volume**: Chaikin Money Flow
- **Volatility**: Bollinger Bands
- **Trend**: MACD
- **Momentum**: Relative Strength Index


# 💾 The Data
![The Data](https://raw.githubusercontent.com/razlevio/razlevio/main/resources/presentation/the-data.png)
 
 # ♻️ First Model 
- We use the Random Forest Regression model to try to predict.
- The X value consists of the indicator's value.
- The y prediction consists of close price in the future.

## Did not work so well
![DWSW](https://raw.githubusercontent.com/razlevio/razlevio/main/resources/presentation/dwsw.png)

# ♻️ Second Model 
- We found out that the data provide many noises, so we have to figure out a way to eliminate it "legally."
- We use divergence definition to eliminate the noises.

## 💹 Divergence Definition
![divergence](https://raw.githubusercontent.com/razlevio/razlevio/main/resources/presentation/divergence.jpg)

## 📊 Second Data Set 
Using the bearish divergence to filter out important data points
This is our second dataset:
![second-data-set](https://raw.githubusercontent.com/razlevio/razlevio/main/resources/presentation/second%20model.png)

# 💿 The Result
![the-result](https://raw.githubusercontent.com/razlevio/razlevio/main/resources/presentation/theresult.png)

# 🏗️ How we built it
-  Python
-  Numpy
-  Pandas
-  Seaborn
-  Streamlit

## 🛑 Challenges we ran into
Primarily the challenges we have ran into have been clean and noisy data sets that have affected performance and accuracy.

## 🏅 Accomplishments we're proud of

Further strengthening our skills in python and data science.

## 💡 What we learned

Various financial python libraries, metrics, terms, methods of stock analysis (technical vs. fundamental analysis).

# ⏭️ What's next for Predict

-   More accurate predictions (higher R^2 value).
-   Adding the rest of the two divergences to see if we can sort more important data.
-   Use more indicators to see if it works better for the prediction.
-   Further measurements to see if it's actually effective or not.
-   Options analysis based on historical data.

# Our Group:
Khuong Tran, Deniz Jasarbasic, Eric Karpovits, Raz Levi

# 👨🏻‍💻 Try Predict
[GitHub Repository](https://github.com/Deniz-Jasa/stock_prediction_model)
