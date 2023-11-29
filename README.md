# Store_Sales_Data_Predict
Store Sales in Weekly sales &amp; Markdowns and regional activity
# INTRODUCTION
Sales data analysis & sales forecasting for each department and stores, then build multiple regression ML models and compare their performance based on model accuracy and RMSE in Python to implement business insights to improve sales and customer interactions.
# Requirements 
- Streamlit
- Streamlit_lottie
- Python
- Pandas
- Numpy
- Matplotlib
- Seaborn
- Sklearn
# Dataset Feature Descriptions
# stores_data_set.csv:
This file contains anonymized information about the 45 stores, indicating the type and size of store.

The 45 Store in Type A,B,C is the category of each department

A - high in weekly sales

B - Average of weekly sales

C - decrease in weekly sales

# sales_data_set.csv:
This is the historical training data, which covers to 2010-02-05 to 2012-11- 01, Within this file you will find the following fields:

Store – the store number

Dept – the department number

Date – the week

Weekly_Sales – sales for the given department in the given store

IsHoliday – whether the week is a special holiday week
# features.csv:
This file contains additional data related to the store, department, and regional activity for the given dates. It contains the following fields:

Store – the store number

Date – the week

Temperature – average temperature in the region

Fuel_Price – cost of fuel in the region

MarkDown1-5 – MarkDown data is only available 2010, 2011,2012 and is not available for all stores all the time. Any missing value is marked with an NA.

CPI – the consumer price index

Unemployment – the unemployment rate

IsHoliday – whether the week is a special holiday week

# Merching Three Csv file:

While looking at the features it is evident that stores CSV files have “Store” as a repetitive column so it’s better to merge those columns to avoid confusion and to add the clarification in the dataset for future visualization.

Using the merge function to merge and we are merging along the common column named Store

Let’s develop a machine learning model for further analysis.

![Intro GUI]()
