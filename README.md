

# Project Description
This project focuses on analyzing and modeling housing data in India. Using Python, the notebook performs data exploration, visualization, and machine learning to uncover patterns in the dataset and build predictive models. Steps are taken to mitigate overfitting, ensuring better model generalization on unseen data.

# Data Import
Libraries Used:
NumPy for numerical computations.
Pandas for data manipulation.
Matplotlib and Seaborn for data visualization.
The dataset (Intern Housing Data India.csv) is loaded using pandas.read_csv.

# Data Exploration
The structure of the dataset is examined using:
data.info() to view data types and non-null counts.
data.head() to preview the first few rows.
data.shape to understand the dimensions.

# Data Cleaning and Preprocessing
Missing values are handled 
![image](https://github.com/user-attachments/assets/fc225b35-9a40-482e-ab78-d34fb2c7ae08)

Purpose: Helps visualize the extent of missing data and decide on handling strategies.
Insights:

Only total_bedrooms has missing values, which are imputed to maintain dataset integrity.

# Data Visualization
Boxplot:

Description: Shows the distribution of numeric features (total_rooms, median_income, etc.), highlighting outliers.
Purpose: Detects extreme values that may skew the model.
Insights: Outliers are capped at the 90th percentile for robust modeling.

![image](https://github.com/user-attachments/assets/72b1044b-6f65-4750-8c30-181f645899d1)

Count Plot for ocean_proximity:

![image](https://github.com/user-attachments/assets/f19e817c-d3be-422d-925c-8557f76aacec)


Description: A bar plot displays the frequency of different categories in the ocean_proximity feature.
Purpose: Understands the distribution of houses based on their proximity to the ocean.
Insights: Most houses are near the ocean, with very few located on islands.

Heatmap (Correlation Matrix):

![image](https://github.com/user-attachments/assets/f8187c6b-b7e5-44ca-b644-f1a72c43190b)


Description: A heatmap shows the correlation between numeric features.
Purpose: Identifies relationships, e.g., high correlations between total_rooms and total_bedrooms suggest multicollinearity.
Insights: median_income is strongly correlated with median_house_value, indicating its predictive strength.


Scatter Plot of median_income vs. median_house_value:

![image](https://github.com/user-attachments/assets/6adbc7b2-6761-4d56-bb0d-e36935652580)


Description: A scatter plot visualizes the relationship between income and house prices.
Purpose: Detects trends and data patterns.
Insights: Higher incomes generally correspond to higher house prices, with some saturation near the upper price limit.


Geographical Scatter Plot (latitude and longitude):

![image](https://github.com/user-attachments/assets/c8c795bf-952b-47ab-863a-49de516f09cb)


Description: Plots house locations based on latitude and longitude, with marker sizes representing population and color representing house prices.
Purpose: Visualizes geographic distribution and trends.
Insights: Coastal areas have higher house prices and denser populations.

# Feature Engineering
Categorical variables (ocean_proximity) are one-hot encoded.
Outliers in numeric columns are capped at the 90th percentile to reduce their influence.


# Model Building and Evaluation
Model: Random Forest Regressor

Selected for its robustness against overfitting and ability to capture non-linear relationships.
Metrics:

RMSE: Measures prediction errors, with lower values indicating better performance.
R²: Explains variance in house prices, with values closer to 1 being ideal.
Results:

RMSE: 49,005.17
R²: 0.75

# How Each Plot Addresses Overfitting
Boxplot: Detects and removes outliers to prevent models from focusing on extreme values.
Correlation Heatmap: Identifies highly correlated features to reduce redundancy.
Scatter Plot:Highlights saturation and non-linear trends for targeted preprocessing.
Geographical Plot: Visualizes spatial patterns to inform feature engineering.

# web application
![Screenshot (382)](https://github.com/user-attachments/assets/85e378de-805f-4f82-ba68-d963a448f73c)






