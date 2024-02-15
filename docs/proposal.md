# Real estate Price Prediction

**Prepared for UMBC Data Science Master Degree Capstone by Dr. Chaoji (Jay) Wang - SPRING 2024 Semester**

**Author: Shiva Sahith Gaddam**

[GitHub](https://github.com/Sahith4311)

[LinkedIn](https://www.linkedin.com/in/gaddam-shivasahith-290567175/)

## Background
1. **What is it about?**
   The project focuses on predicting real estate prices using machine learning techniques. Real estate pricing is a critical aspect of the housing market, impacting both buyers and sellers. By analyzing various features such as transaction date, house age, distance to amenities, and geographical coordinates, the project aims to develop a model that accurately predicts the price of residential properties.

2. **Why does it matter?**
   Real estate is a significant investment for individuals and families, making informed decisions crucial. Accurate price prediction helps prospective buyers assess the value of properties they are interested in, while sellers can set competitive prices based on market trends. Additionally, stakeholders like real estate agents, developers, and investors rely on reliable price forecasts for strategic decision-making and risk management.

3. **What are your research questions?**
   - How do different factors such as transaction date, house age, and proximity to amenities affect real estate prices?
   - Can machine learning models effectively predict the prices of residential properties based on these factors?
   - What preprocessing techniques and feature engineering methods optimize the model's predictive performance?
   - How does the chosen machine learning algorithm compare to alternative regression methods in terms of prediction accuracy and robustness?

4. **Why is it relevant?**
   
   Understanding the relationship between various property features and prices can provide valuable insights into the dynamics of the real estate market. Moreover, developing an accurate predictive model can assist stakeholders in making informed decisions regarding property investments, sales, and rental pricing strategies. By leveraging machine learning, the project aims to enhance transparency, efficiency, and fairness in real estate transactions.

## Data
### Data Sources
The datasets used for this project include a CSV file [Real estate.csv](https://www.kaggle.com/code/shreyasbagwe1015/realestate-price-prediction/input) containing information regarding houses and many more.

### Data Size
- Real estate.csv: 21 kB

### Data Shape
- Real estate.csv: 414 rows by 8 columns

### Dataset Description
**Each Row Represents:**

The dataset provided consists of information about various housing units, including transaction dates, house ages, distances to the nearest MRT (Mass Rapid Transit) station, the number of convenience stores nearby, latitude, longitude, and house prices per unit area. Each row represents a different housing unit, with corresponding details such as transaction dates ranging from 2012.667 to 2013.583, house ages from 0 to 43.8 years, distances to the nearest MRT station from 23.38284 to 6488.021, the number of convenience stores from 0 to 10, latitude ranging from 24.93293 to 25.01459, longitude from 121.49507 to 121.56627, and house prices per unit area from 7.6 to 117.5. The dataset showcases a variety of housing characteristics and can be used for various analytical purposes, such as predicting house prices based on the provided features or understanding the factors influencing house prices in a particular area.

### Dataset Columns
- No: Unique identifier for each data entry (int64)
- X1 Transaction Date: The date of the transaction in the format YYYY.MM (float64)
- X2 House Age: The age of the house in years (float64)
- X3 Distance to the Nearest MRT Station: The distance to the nearest Mass Rapid Transit (MRT) station in meters (float64)
- X4 Number of Convenience Stores: The number of convenience stores within a certain radius (int64)
- X5 Latitude: The latitude of the location (float64)
- X6 Longitude: The longitude of the location (float64)
- Y House Price of Unit Area: The price of the house per unit area in square meters (float64)

## Potential Values
In the provided dataset, the potential values for each feature are as follows:

1. Transaction date: Values ranging from 2012.667 to 2013.583, representing dates in the format YYYY.MM, indicating the year and month of the transaction.
2. House age: Values ranging from 0 to 43.8 years, indicating the age of the house in years at the time of the transaction.
3. Distance to the nearest MRT station: Values ranging from 23.38284 to 6488.021, representing distances in meters from the housing unit to the nearest Mass Rapid Transit (MRT) station.
4. Number of convenience stores: Values ranging from 0 to 10, indicating the count of convenience stores located near the housing unit.
5. Latitude: Values ranging from 24.93293 to 25.01459, representing the latitude coordinates of the housing unit's location.
6. Longitude: Values ranging from 121.49507 to 121.56627, representing the longitude coordinates of the housing unit's location.
7. House price per unit area: Values ranging from 7.6 to 117.5, indicating the price per unit area of the housing unit in terms of currency (e.g., dollars, euros) per square meter.

These potential values provide a range of characteristics for each housing unit, allowing for analysis and exploration of factors influencing house prices in the area.

### Target/Label
The target variable is likely to be "House price per unit area." This means that the goal of any analysis or modeling performed on this dataset would be to predict the price per unit area of a housing unit.

### Features/Predictors
The features or predictors for the machine learning model in this dataset could include various factors that might influence the price per unit area of a housing unit. Some potential features could be:
- Transaction date: The date when the transaction occurred, which could potentially affect housing prices due to market trends and seasonality.
- Age: The age of the housing unit, which might influence its condition and desirability.
- Distance to the nearest MRT station: Proximity to public transportation can significantly impact property values.
- Number of nearby convenience stores: Access to amenities like convenience stores may affect housing prices.
- Latitude and longitude: Geographic coordinates could capture the location's desirability or accessibility.

