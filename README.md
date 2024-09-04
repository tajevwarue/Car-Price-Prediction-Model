# Car Price Prediction Model

## Overview
This project aims to predict car prices using machine learning techniques. The dataset, provided by our tutor, contains 8128 rows and 13 columns with attributes relevant to car pricing. The project involves several key steps, including data cleaning, feature engineering, data visualization, encoding, scaling, feature selection, hyperparameter tuning, and model combination.

## Data Overview
The dataset contains the following columns:

| #   | Column         | Non-Null Count | Dtype   |
|-----|----------------|----------------|---------|
| 0   | name           | 8128 non-null  | object  |
| 1   | year           | 8128 non-null  | int64   |
| 2   | selling_price  | 8128 non-null  | int64   |
| 3   | km_driven      | 8128 non-null  | int64   |
| 4   | fuel           | 8128 non-null  | object  |
| 5   | seller_type    | 8128 non-null  | object  |
| 6   | transmission   | 8128 non-null  | object  |
| 7   | owner          | 8128 non-null  | object  |
| 8   | mileage        | 7907 non-null  | object  |
| 9   | engine         | 7907 non-null  | object  |
| 10  | max_power      | 7913 non-null  | object  |
| 11  | torque         | 7906 non-null  | object  |
| 12  | seats          | 7907 non-null  | float64 |

There are missing values in the `mileage`, `engine`, `max_power`, `torque`, and `seats` columns. Additionally, the data types of `mileage`, `max_power`, `engine`, and `torque` were initially of type `object` rather than `float`.

## Data Cleaning and Preprocessing
The data cleaning process involved:
1. Removing duplicates.
2. Formatting columns.
3. Handling missing values in key columns such as `mileage`, `engine`, `max_power`, `torque`, and `seats`.

Two methodologies were employed for handling missing values:
1. Imputing missing values using reference rows that had the same `name` and `year`.
2. Using the mean value of each brand to fill the missing values for each column.

## Feature Engineering
Significant features were engineered to enhance the model's predictive power:
1. Conversion of data types for `mileage`, `engine`, `max_power`, and `torque` to float.
2. Splitting `torque` into numerical values and RPM, converting kgm to Nm.
3. Deriving `engine_power` from torque values.
4. Extracting `brand_name` from the `name` column.
5. Removing outliers from `max_power`, `engine_power`, and `mileage`.

## Exploratory Data Analysis (EDA)
Key insights from EDA include:
1. A positive correlation between engine capacity and selling price.
2. Higher max power is associated with higher selling prices.
3. Premium brands like Lexus and Volvo have the highest selling prices, while budget brands such as Opel and Daewoo have the lowest.
4. Diesel and petrol cars are priced higher than other cars.
5. Dealer-sold cars command higher prices than those sold by individuals.
6. Automatic transmission cars are more expensive than manual ones.
7. First-owner and test drive cars are valued higher than cars with multiple previous owners.
8. Newer cars tend to be more expensive.

## Encoding and Scaling
- Applied Binary Encoding for the `name` column.
- Applied OneHot Encoding for the `fuel`, `seller_type`, `transmission`, and `owner` columns.
- Scaled the data using StandardScaler.

## Feature Selection
- Utilized Recursive Feature Elimination (RFE) to select the most important features.

## Hyperparameter Tuning
- Leveraged Optuna for efficient hyperparameter optimization.

## Model Combination
- Combined multiple models using VotingRegressor for improved accuracy.