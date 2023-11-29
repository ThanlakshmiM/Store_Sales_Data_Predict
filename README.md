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
# Overview : 
Sales data analysis & sales forecasting for each department and stores, then build multiple regression ML models and compare their performance based on model accuracy and RMSE in Python to implement business insights to improve sales and customer interactions.

# Conclusion
We examined the store’s sales forecasting dataset by applying various statistical and visualization techniques.

We trained and developed four ML models. We also concluded that for this problem, DecisionTree Regressor works best.


![image](https://github.com/ThanlakshmiM/Store_Sales_Data_Predict/assets/111423676/04163fe9-8183-4ba9-a99b-f60f613ad768)
user select a algorthm of pridict the weekly sales
# Accuracy & Metrics of DecisionTree
The DecisionTree Regressor 100 % of Accuracy works.
![image](https://github.com/ThanlakshmiM/Store_Sales_Data_Predict/assets/111423676/94e0b825-2b98-4ab3-aca8-4ff6a9a2c23a)

# Describe the dataset
![image](https://github.com/ThanlakshmiM/Store_Sales_Data_Predict/assets/111423676/19413301-e0b4-4db0-83f6-ee14cc4e2e67)

# Total of Type each Store
Here from the above pie chart it is clearly visible that Type c has the minimum number of stores while Type A has the maximum number of stores.
![image](https://github.com/ThanlakshmiM/Store_Sales_Data_Predict/assets/111423676/38d777f4-c1d3-4d87-8ed0-e02e9316c6ee)

# Feature Importance
![image](https://github.com/ThanlakshmiM/Store_Sales_Data_Predict/assets/111423676/4b6955eb-c459-426a-9c39-fa33fb8195d0)
# Markdown effect on Holidays
select one markdown negative sales on Holidays show in scatter plot

![image](https://github.com/ThanlakshmiM/Store_Sales_Data_Predict/assets/111423676/abbb2d86-d475-4a44-99d7-7bcd40492f8c)




