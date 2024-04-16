# Real estate Price Prediction

**Prepared for UMBC Data Science Master Degree Capstone by Dr. Chaoji (Jay) Wang - SPRING 2024 Semester**

**Author: Shiva Sahith Gaddam**

[GitHub](https://github.com/Sahith4311)

[LinkedIn](https://www.linkedin.com/in/shiva-sahith-gaddam-290567175/)

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

  ```markdown
# Exploratory Data Analysis (EDA)

## Overview
Exploratory Data Analysis (EDA) is a crucial step in understanding the dataset's structure, distribution, and relationships between variables. Let's explore the real estate dataset to gain insights into its features and potential patterns.

## Data Summary
- The dataset contains information about various housing units, including transaction dates, house ages, distances to the nearest MRT station, the number of convenience stores nearby, latitude, longitude, and house prices per unit area.
- It comprises 414 rows and 8 columns.

## Data Visualization
### Histograms:
- **X1 Transaction Date:** The histogram of transaction dates reveals that most transactions occurred in the latter half of the time period, suggesting potential temporal trends in housing market activity.
- <div>
    <img src="docs/Transaction_date.png" alt="X1 Transaction Date Histogram" style="width: 400px;"/>
</div>
 

- **X2 House Age:** The distribution of house ages shows that the majority of houses are relatively new, with ages concentrated around 0 to 20 years. This indicates a predominantly modern housing market.
- **X3 Distance to the Nearest MRT Station:** The histogram displays variations in distances, with some houses located very close to MRT stations and others farther away. Proximity to public transportation may influence housing prices.
- **X4 Number of Convenience Stores:** The histogram illustrates that the majority of houses have a small number of convenience stores nearby, with a few outliers having a higher count. This suggests that access to amenities may vary across locations.
- **X5 Latitude and X6 Longitude:** The histograms of latitude and longitude values show distributions corresponding to the geographical area covered by the dataset. These features represent the spatial coordinates of housing units, which may affect their desirability and prices.
- **Y House Price of Unit Area:** The histogram of house prices per unit area reveals variability, with prices ranging from 7.6 to 117.5 currency units per square meter. This variation underscores the importance of understanding factors driving price differences across properties.

### Insights
- **Temporal Trends:** Transaction dates span two years, indicating data collected over a period with potential seasonal variations. Further analysis of temporal patterns may provide insights into seasonal fluctuations in housing market activity and their impact on prices.
- **Housing Age Profile:** The concentration of house ages in the range of 0 to 20 years suggests a relatively new housing market with a significant portion of modern properties. This information is valuable for understanding the housing stock's composition and its implications for buyers and sellers.
- **Proximity to Amenities:** The distribution of distances to the nearest MRT station and the number of convenience stores highlights spatial variations in accessibility to amenities. Properties located closer to transportation hubs and with more nearby convenience stores may command higher prices due to increased convenience and accessibility.
- **Spatial Dependencies:** Strong correlations between house prices and the number of convenience stores, as well as latitude and longitude, indicate spatial dependencies in housing prices. Analyzing these spatial patterns can provide valuable insights into local market dynamics and help identify areas of high demand or potential investment opportunities.

## Data Cleansing:
**Handling Missing Values:**
- No missing values were detected in the dataset, ensuring that all records contain complete information.

**Handling Duplicate Rows:**
- No duplicate rows were found in the dataset, indicating data consistency and preventing redundancy.

## Data Preprocessing:
**Date Column:**
- **Convert to DateTime Format:** The 'X1 transaction date' column was successfully converted to datetime format, enabling temporal analysis and manipulation.
- **Extract Features:** From the 'X1 transaction date' column, the month and year were extracted to facilitate time-based analysis and trend identification.

**Tidying Data:**
- **Tidy Data Format:** The dataset adheres to the principles of tidy data, with each row representing a single observation and each column representing a unique variable. This structured format facilitates efficient data analysis and modeling processes.

## Machine Learning Model Building:

**Data Splitting:**
- The dataset was divided into training and testing sets using a 70-30 split, with 70% of the data allocated for training the model and 30% for evaluating its performance. This split ensures that the model is trained on a sufficiently large portion of the data while retaining unseen data for unbiased evaluation.

**Model Selection:**
- Linear Regression Model: A linear regression model was chosen for its simplicity and interpretability, making it suitable for predicting house prices based on various features.
  
**Model Training:**
- The linear regression model was trained using the training dataset, where the features (independent variables) were used to predict the target variable (house price per unit area).
  
**Model Evaluation:**
- **Coefficient Analysis:** The coefficients of the linear regression model were analyzed to understand the impact of each feature on the predicted house prices. Positive coefficients indicate a positive relationship with the target variable, while negative coefficients suggest a negative relationship.
  
**Model Performance Metrics:**
- **Mean Absolute Error (MAE):** The MAE was calculated to measure the average absolute difference between the predicted and actual house prices. A lower MAE indicates better model performance.
  
- **Mean Squared Error (MSE):** The MSE quantifies the average squared difference between the predicted and actual house prices, providing insight into the variance of prediction errors.
  
- **Root Mean Squared Error (RMSE):** The RMSE, derived from the MSE, represents the square root of the average squared difference between predicted and actual house prices. It provides a more interpretable measure of error in the same units as the target variable.

**Model Persistence:**
- The trained linear regression model was serialized using the pickle library, allowing it to be stored as a binary file for future use without the need for retraining. This enables easy deployment and integration of the model into production environments for

 real-time predictions.

## Model Performance:
After training the linear regression model to predict house prices based on various features, the following performance metrics were calculated on the testing dataset:

- **Mean Absolute Error (MAE):** MAE = 6.485337
- **Mean Squared Error (MSE):** MSE = 72.908600
- **Root Mean Squared Error (RMSE):** RMSE = 8.538653

These metrics serve as indicators of the model's predictive performance, with lower values indicating better accuracy and precision in predicting house prices. While the model demonstrates reasonable performance, further optimization and fine-tuning may be explored to improve its predictive capabilities.

## Results:
The results of the real estate price prediction model indicate promising performance, with the following key metrics:

- **Mean Absolute Error (MAE):** The MAE, representing the average absolute difference between predicted and actual prices, is approximately 6.49 units.
  
- **Mean Squared Error (MSE):** The MSE, which measures the average squared difference between predicted and actual prices, is approximately 72.91.

- **Root Mean Squared Error (RMSE):** The RMSE, indicating the square root of the MSE and providing a more interpretable measure of error, is approximately 8.54 units.

These metrics provide insights into the model's predictive accuracy and performance. The relatively low values of MAE, MSE, and RMSE suggest that the model is capable of making reasonably accurate predictions of real estate prices based on the provided features.

Moreover, visual analysis of the scatter plot comparing predicted and actual prices reveals a generally positive correlation between the two, indicating that the model captures the underlying patterns in the data effectively. Additionally, the residual plot shows no clear patterns or trends, suggesting that the model's predictions exhibit random errors and do not display systematic bias.

Overall, the results demonstrate the feasibility and effectiveness of using machine learning techniques for real estate price prediction, providing valuable decision support for stakeholders in the housing market.

## Application of Trained Model:
The application of the trained model for real estate price prediction via Streamlit offers a seamless user experience, allowing stakeholders to interactively input property features and obtain predictions for its price. This user-friendly interface streamlines the process of estimating property values, catering to users with varying levels of technical expertise. By providing input fields for each feature and a "Predict" button to trigger model predictions, the application ensures accessibility and ease of use.

Upon inputting feature values and clicking the "Predict" button, users receive real-time predictions for the property's price per unit area. This instant feedback enables users to explore different scenarios by adjusting feature values and observing their impact on the predicted price. The application's interactivity fosters a deeper understanding of how various property characteristics influence its value, empowering users to make informed decisions in the real estate market.

Accessible via a web browser, the application offers flexibility and convenience, allowing users to access it from any device with internet connectivity. This accessibility enhances usability and expands the application's reach to a wider audience of stakeholders, including real estate agents, buyers, sellers, and investors. The intuitive interface and real-time feedback mechanism streamline the process of property valuation, facilitating quicker decision-making and informed negotiations.

While the current version of the application provides valuable functionality for estimating property prices, there is potential for future enhancements to further improve its utility and effectiveness. Incorporating additional features such as property size, number of bedrooms, and neighborhood characteristics could enhance the model's predictive accuracy and relevance. Exploring advanced machine learning techniques and ensemble methods may also improve prediction performance, especially in complex real estate market scenarios.

Furthermore, integrating interactive geospatial visualization tools could enhance data exploration and decision-making by providing visual insights into spatial patterns and property attributes. Developing dynamic pricing models that adapt to changing market conditions in real-time could offer valuable decision support for stakeholders by providing personalized pricing recommendations and optimizing pricing strategies based on current market trends.

In conclusion, the Streamlit application for real estate price prediction serves as a valuable tool for stakeholders in the real estate market, offering an intuitive and accessible platform for estimating property values based on specific features. With its user-friendly interface, real-time feedback, and potential for future enhancements, the application empowers users to make informed decisions and navigate the complexities of the real estate market with confidence.

## Conclusion:

The real estate price prediction project aimed to develop a machine learning model capable of accurately predicting house prices based on various features such as transaction date, house age, distance to amenities, and geographical coordinates. The dataset contained information on 414 housing units with eight features, including transaction date, house age, distance to the nearest MRT station, number of convenience stores, latitude, longitude, and house price per unit area. Exploratory data analysis (EDA) revealed insights into the distribution of features and their correlations with house prices, providing valuable information for model development. Data cleansing involved handling missing values and duplicate rows, ensuring the dataset's integrity and quality. Preprocessing steps included converting the transaction date to datetime format, extracting additional features like month and year, and ensuring the dataset adhered to tidy data principles.

A linear regression model was trained using the preprocessed data to predict house prices. Performance metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) were calculated to evaluate the model's predictive accuracy. The model demonstrated reasonable performance, with MAE of 6.49, MSE of 72.91, and RMSE of 8.54, indicating moderate accuracy in predicting house prices. Moving forward, the model's performance could be further improved by exploring advanced machine learning algorithms, feature engineering techniques, and hyperparameter tuning. Additional features such as property size, number of bedrooms and bathrooms, and neighborhood characteristics could enhance the model's predictive capabilities. External factors like economic indicators, market trends, and demographic data could be incorporated for a more comprehensive analysis of housing market dynamics.

Accurate real estate price prediction models have significant implications for buyers, sellers, real estate agents, developers, and investors. Stakeholders can use predictive models to make informed decisions regarding property investments, pricing strategies, and risk management. Transparency and efficiency in real estate transactions can be enhanced through data-driven approaches, leading to fairer and more equitable housing markets. Overall, while the linear regression model achieved reasonable performance, there is ample room for refinement and enhancement through leveraging advanced techniques and incorporating additional data sources in future iterations of the model.

## Limitations:

1. **Limited Feature Set:** The model's predictive power is constrained by the features included in the dataset. Factors such as property size, amenities, and neighborhood characteristics could significantly influence housing prices but are not accounted for in this analysis.

2. **Assumption of Linearity:** Linear regression assumes a linear relationship between the independent variables and the target variable. However, in reality, the relationship may be more complex, leading to potential inaccuracies in predictions.

3. **Overfitting or Underfitting:** The model may suffer from overfitting, where it captures noise in the training data rather than underlying patterns, or underfitting, where it fails to capture the true relationship between variables, both of which can compromise its performance on unseen data.

4. **Data Quality Issues:** The accuracy and reliability of predictions are contingent upon the quality of the data. Errors, outliers, or missing values in the dataset could adversely affect the model's performance and predictive accuracy.

5. **Limited Generalization:** The model's ability to generalize to new, unseen data may be limited by the specificity of the dataset. If the dataset is not representative of the broader population or if there are inherent biases in the data, the model's predictions may not be applicable in real-world scenarios.

6. **External Factors:** The housing market is influenced by a myriad of external factors such as economic conditions, government policies, and sociopolitical events, which are not accounted for in the model. Failure to consider these external factors could lead to inaccurate predictions and unreliable insights.

7. **Ethical Considerations:** Predictive models in real estate raise ethical concerns related to fairness, transparency, and discrimination. Biases present in the data or the model itself could result in inequitable outcomes, disadvantaging certain groups or communities. Additionally, the use of predictive analytics in real estate decision-making may raise privacy concerns regarding the use of personal data and the potential for algorithmic discrimination.

Addressing these limitations requires a holistic approach, including the incorporation of additional relevant features, the use of more sophisticated modeling techniques, rigorous data validation and cleansing procedures, and ethical considerations in model development and deployment.

## Lessons Learned:

1. **Feature Importance:** Understanding the importance of different features in predicting real estate prices was a key takeaway. Features such as proximity to amenities, transaction date, and property age emerged as significant predictors, highlighting the importance of considering multiple factors in price estimation.

2. **Model Evaluation:** Evaluating model performance using appropriate metrics such as Mean Absolute Error (MAE),

 Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) provided valuable insights into the accuracy and reliability of predictions. This process emphasized the importance of selecting robust evaluation metrics tailored to the specific problem domain.

3. **Data Preprocessing:** The importance of data preprocessing techniques, including handling missing values, scaling features, and encoding categorical variables, became evident during the project. Effective data preprocessing not only improves model performance but also ensures the integrity and quality of the analysis results.

4. **Model Selection:** Experimenting with different machine learning algorithms, including linear regression, helped in understanding the strengths and limitations of each approach. It became clear that the choice of the appropriate model depends on various factors such as data complexity, interpretability, and computational efficiency.

5. **Interpretability vs. Complexity:** Balancing model interpretability with complexity was a recurring theme throughout the project. While simpler models like linear regression offer interpretability, more complex models such as ensemble methods or neural networks may provide higher predictive accuracy but at the cost of interpretability. Understanding this trade-off is crucial in selecting the most suitable model for a given scenario.

6. **Domain Knowledge:** Incorporating domain knowledge, such as understanding the real estate market dynamics and relevant economic indicators, proved invaluable in refining the model and interpreting the results effectively. Combining data-driven insights with domain expertise enhances the robustness and practical relevance of the analysis.

7. **Continuous Improvement:** The iterative nature of the data science process highlighted the importance of continuous improvement. Revisiting and refining the model based on feedback, new data, or changing requirements is essential for ensuring the relevance and efficacy of predictive models in dynamic environments.

By embracing these lessons learned, future projects in real estate price prediction or similar domains can benefit from a more informed and systematic approach, leading to more accurate predictions and actionable insights.

## Future Scope:

1. **Advanced Modeling Techniques:** Exploring advanced machine learning techniques such as gradient boosting, random forests, or neural networks could potentially improve prediction accuracy. These models have the capacity to capture nonlinear relationships and interactions between features, which may lead to better performance, especially in complex real estate market scenarios.

2. **Feature Engineering:** Further refinement of feature engineering methods could enhance model interpretability and predictive power. Incorporating additional relevant features, creating interaction terms, or engineering new variables based on domain knowledge may uncover hidden patterns and improve the model's ability to generalize to unseen data.

3. **Spatial Analysis:** Leveraging spatial analysis techniques could provide deeper insights into geographical patterns and spatial dependencies in real estate pricing. Spatial autocorrelation analysis, hotspot detection, and spatial interpolation methods can help identify clusters of high or low-value properties, contributing to a more nuanced understanding of market dynamics.

4. **Temporal Analysis:** Conducting temporal analysis to capture seasonality, trends, and cyclic patterns in real estate prices could enhance predictive models. Time series analysis techniques such as seasonal decomposition, trend analysis, and forecasting models can aid in forecasting future price trends and identifying market fluctuations over time.

5. **Ensemble Methods:** Investigating ensemble methods, such as stacking or model averaging, can potentially improve model robustness and stability. By combining predictions from multiple models, ensemble methods mitigate individual model biases and variance, leading to more reliable predictions and reduced risk of overfitting.

6. **Data Integration:** Integrating additional datasets, such as demographic data, economic indicators, or urban development plans, can enrich the analysis and provide a more comprehensive understanding of real estate market dynamics. Incorporating external data sources allows for a broader perspective and may uncover novel insights into factors influencing property prices.

7. **Dynamic Pricing Models:** Developing dynamic pricing models that adapt to changing market conditions in real-time could provide valuable decision support for buyers, sellers, and investors. Utilizing real-time data feeds, machine learning algorithms, and predictive analytics, these models can offer personalized pricing recommendations and optimize pricing strategies based on current market trends and customer preferences.

8. **Geospatial Visualization:** Implementing interactive geospatial visualization tools can facilitate data exploration and decision-making for stakeholders in the real estate industry. Heatmaps, choropleth maps, and interactive dashboards enable users to visualize spatial patterns, explore property attributes, and make informed decisions based on spatial insights.

By embracing these future scope areas, researchers and practitioners in the field of real estate price prediction can advance the state-of-the-art models, uncover deeper insights, and develop innovative solutions to address the evolving challenges and opportunities in the real estate market.

## References:
https://www.geeksforgeeks.org/ml-linear-regression/

https://towardsdatascience.com/exploratory-data-analysis-8fc1cb20fd15

https://baotramduong.medium.com/regression-rivalry-mse-vs-mae-vs-rmse-which-is-the-best-metric-9f898167e773#:~:text=to%20use%20each.-,Mean%20Squared%20Error%20(MSE)%2C%20Mean%20Absolute%20Error%20(MAE,between%20predicted%20and%20actual%20values.


