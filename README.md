# Bike Sharing Demand Forecasting

**Dataset Link** - [Bike Sharing Dataset on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/bike-sharing-dataset)
### I have used the hour dataset to get to work on more precise features, this dataset also contains a day wise data. 

Predicting inventory demand is a classic operations problem. For this project, the goal was to forecast exactly how many bikes would be rented in any given hour across a city, relying purely on environmental factors like temperature, humidity, and the time of day. 

Instead of basic regression or standard Random Forests, this project steps up to **XGBoost** (Extreme Gradient Boosting) to handle the forecasting.

## The Build Process

Prepping the data for this one required catching a massive data leakage trap before training:

* **Killing Data Leakage:** The dataset included `casual` and `registered` user counts for each hour. Since those two columns mathematically add up to the target variable (`cnt`), leaving them in the feature matrix would have let the model "cheat" by just learning basic addition. I dropped both before training.
* **Skipping the Scaler:** Unlike neural networks or distance-based algorithms (like KNN) that break if data isn't scaled, tree-based models like XGBoost just look for threshold splits (e.g., `temp > 0.5`). Because of this, I was able to skip standardization entirely and feed the raw numerical features straight into the model.
* **The Algorithm:** I used `XGBRegressor`. Instead of building a bunch of independent decision trees, XGBoost builds them sequentially—each new tree aggressively targets the mistakes (residuals) made by the previous ones.

## Does it actually work?

Very well. The model nailed the hourly demand patterns:

* **R² Score : ~0.94** (The model successfully explains 94% of the variance in bike rentals based on the weather and time data).
* **RMSE :** **~41.6 bikes** (On average, the model's hourly forecast is only off by about 41 physical bikes).

For a city-wide logistics network, being able to predict hourly demand with that level of accuracy is highly actionable for rebalancing inventory. We can further improve the results by hyperparameter tuning

## Tech Stack
* Python
* Pandas 
* XGBoost
* Scikit-Learn 
