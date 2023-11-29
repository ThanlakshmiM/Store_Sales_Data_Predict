#install & import the libraries
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
import plotly.express as px
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score    
from PIL import Image 
import time  #spinner

#page congiguration
st.set_page_config(page_title= "Sales Data set",
                   page_icon= 'random',
                   layout= "wide",)
st.markdown("<h1 style='text-align: center; color: white;'> Sales Pridiction </h1>", unsafe_allow_html=True)


#application background
def app_bg():
    st.markdown(f""" <style>.stApp {{
                        background: url("https://cdn.wallpapersafari.com/7/90/BFUQb1.jpg");
                        background-size: cover}}
                     </style>""",unsafe_allow_html=True)
app_bg()


def get_profile_pic():
     return Image.open(r'C:\Users\user\Desktop\python\project_4\analysis.png')

def main():

  # SIDEBAR

  st.sidebar.image(get_profile_pic(), use_column_width=False, width=250)
  st.sidebar.header("Welcome!")

  st.sidebar.markdown(" ")
  st.sidebar.markdown("*I am a Data Science enthusiast with interest in Python, Machine Learning, Data Analysis.*")
  st.sidebar.markdown("**Author**: Thanalakshmi")
  st.sidebar.markdown("**Mail**: thanalakshmi7558@gmail.com")

  st.sidebar.markdown("- [Linkedin](https://www.linkedin.com/in/thanalakshmi-m-7633ba235/)")
  st.sidebar.markdown("- [Github](https://github.com/ThanlakshmiM)")

# Importing store,sales,feature data set
df1=pd.read_csv(r'C:\Users\user\Desktop\python\stores_data_set.csv')
df2=pd.read_csv(r'C:\Users\user\Desktop\python\sales_data_set - sales_data_set.csv')
df3=pd.read_csv(r'C:\Users\user\Desktop\python\Features_data_set.csv')

df4=df2.merge(df3, on=['Store', 'Date','IsHoliday'], how='left')
df5=df1.merge(df4, on=['Store'], how='left')
df5.drop('Date', axis=1, inplace=True)
df5.fillna(0,inplace=True)
           
eng=OrdinalEncoder()
df5['Type']=eng.fit_transform(df5[['Type']])
df5['IsHoliday']=eng.fit_transform(df5[['IsHoliday']])
#df5['Type']=df5['Type'].map({'A': 0, 'B': 1, 'C': 2})               # Type---> 0 - A, 1 - B, 2 - C  
#df5['IsHoliday']=df5['IsHoliday'].map({'False':0,'True':1})         # IsHoliday---> 0 - False, 1 - True


y=df5['Weekly_Sales']
x=df5.drop(['Weekly_Sales'],axis=1)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.15) # test min 15% max 20%


def Type(type):
   if type=="A":
      return float(0)
   elif type=="B":
      return float(1)
   elif type=="C":
      return float(2)
   
def holiday(holiday):
   if holiday=="False":
      return float(0)
   else:
      if holiday=="True":
        return float(1) 
def np_append(value):

   return float(value)
 
def algorithm(algorithm):
    
    if algorithm == 'DecisionTreeRegressor':
        return DecisionTreeRegressor()
    elif algorithm == 'LinearRegression':
        return LinearRegression()
    elif algorithm == 'RandomForestRegressor':
        return RandomForestRegressor(n_estimators=100,max_depth=2,random_state=0)
    elif algorithm == 'AdaBoostRegressor':
        return AdaBoostRegressor()
    elif algorithm == 'GradientBoostingRegressor':
        return GradientBoostingRegressor()
    elif algorithm == 'ElasticNet':
        return ElasticNet()
    

def algorithm_train_test_accuracy(x_train, x_test, y_train, y_test, algo):
         
         algorith=algorithm(algo)
         model = algorith.fit(x_train, y_train)
         y_pred_train = model.predict(x_train)
         y_pred_test = model.predict(x_test)
         r2_train = r2_score(y_train, y_pred_train)
         r2_test = r2_score(y_test, y_pred_test)
         

         # algo = str(algorithm).split("'")[1].split(".")[-1]
         accuracy = {'algorithm': algorith,
                'R2_train' : r2_train,
                'R2_test'  : r2_test}

         return accuracy

def score_error(x_train, y_train,x_test,y_test,algo):

         regressor=algorithm(algo)  
         result=regressor.fit(x_train, y_train)
         y_pred=result.predict(x_test) 

         mse = mean_squared_error(y_test, y_pred)
         rmse = np.sqrt(mse)
         r2 = r2_score(y_test, y_pred)
         mae = mean_absolute_error(y_test, y_pred)
         acc_rf= round(result.score(x_train, y_train) * 100, 2)

         metrics = {'R2': r2,
           'Mean Absolute Error': mae,
           'Mean Squared Error': mse,
           'Root Mean Squared Error': rmse,
           "Accuracy " : str(acc_rf)+"%" }

         return metrics
dff4=df2.merge(df3, on=['Store', 'Date','IsHoliday'], how='left')
df=df1.merge(dff4, on=['Store'], how='left')

def Markdown_effect(MarkDown):
     col1,col2=st.columns(2,gap='medium')

     col1.write(f'### { MarkDown } effect on Holidays')
     fig=plt.figure(figsize=(8,8))
    
     hol=df5.query(" IsHoliday == 1.0 ")
     eff=hol[hol['Weekly_Sales']<=0]
     
     fig = px.scatter(data_frame=eff, x="Weekly_Sales", y=MarkDown, color="IsHoliday")
     fig.update_yaxes(title=MarkDown)
     fig.update_xaxes(title="Weekly Sales")
     col1.markdown('   Select any one of markdown effect on Holidays ')
     col1.plotly_chart(fig,use_container_width=True)
     f=eff.IsHoliday.value_counts()
     d=pd.DataFrame(f)
     #st.write(qry)
     col2.markdown('### Markdown effect on Holidays')
     col2.write(eff)
     col1.markdown('### IsHolidays Counts')
     col1.write(d)
        


#Creating option menu in the menu bar
selected = option_menu(None,['Home',"About","Predict","Visualization & Insights"],
                        icons=["house","at","toggles","bar-chart"],
                        default_index=0,
                        orientation="horizontal")

if selected == 'Home':
   col1,col2=st.columns(2,gap='medium')

   col1.markdown("### :blue[Domain] : 👉 Store Sales in Weekly sales & Markdowns and regional activity")
   col1.markdown("### :blue[Technologies used] :👉 Python, Pandas,ML, Plotly, Streamlit")
   col1.markdown("### :blue[Overview] : 👉Sales data analysis & sales forecasting for each department and stores, then build multiple regression ML models and compare their performance based on model accuracy and RMSE in Python to implement business insights to improve sales and customer interactions.")
   sales=Image.open(r'C:\Users\user\Desktop\python\final_project\sales forcasting.jfif')
   col2.image(sales)
   pic=Image.open(r'C:\Users\user\Desktop\python\final_project\sales growth.jfif')
   col2.image(pic)

if selected == 'Predict': 
   with st.form("my_form"):
                col1,col2,col3=st.columns([5,5,5])
                col1.write(' ')
                col1.write(' ')
                col1.write(' ')
                col1.markdown("Select any store number from 1 to 45")
                store= col1.selectbox("Select a Store",(df5['Store'].unique()))
                col1.markdown("Select any dept number from 1 to 99")
                dept=col1.selectbox("Select a Dept",(df5['Dept'].unique()))
                col1.markdown("A,B,C is the category of each department")
                col1.markdown("A -  high in weekly sales")
                col1.markdown("B - Average of weekly sales")
                col1.markdown("C - decrease in weekly sales")
                Type=Type(col1.selectbox("Select a Type",(df1['Type'].unique())))
                Holiday=holiday(col1.selectbox("Select a IsHoliday",('False','True')))

            
            
                col2.markdown("Enter the size of the department in the store")
                Size = col2.text_input("Enter Size")
                col2.markdown("Enter the at Temperature ")
                Temperature =col2.text_input("Enter Temperature")
                col2.markdown("Enter the price of Feul at that time")
                Fuel_Price =col2.text_input("Enter a Fuel Price")
                col2.markdown("Enter the your 5 Type of markdown")
                MarkDown1 = col2.text_input("Enter MarkDown1")
                MarkDown2 = col2.text_input("Enter MarkDown2")

            
                MarkDown3 = col3.text_input("Enter MarkDown3")
                MarkDown4 = col3.text_input("Enter MarkDown4")
                MarkDown5 = col3.text_input("Enter MarkDown5")
                col3.markdown("Enter a Customer Price Index Counts")
                CPI=col3.text_input("Enter a CPI")
                col3.markdown("Enter a Unemployment counts")
                Unemployment=col3.text_input("Enter a Unemployment")
                col3.markdown(" ")
                weekend=Image.open(r'C:\Users\user\Desktop\python\final_project\week end sales.jfif')
                col3.image(weekend)
               
         
                # Handle the form submission
                st.markdown('### Select algorithm of pridict your Weekly Sales')
                     # select any one algorithm # find Training and Testing accuracy of algorithm
                select=st.selectbox('SELECT ALGORITHM ',['None','DecisionTreeRegressor','LinearRegression','RandomForestRegressor','AdaBoostRegressor','GradientBoostingRegressor','ElasticNet'])
       
                if st.form_submit_button(label="PREDICT WEEKLY SALES"):
                     
                     if select != 'None':
                      try:
                        # Training and Testing different algorithms using pridict your sales
                        regressor=algorithm(select)
                     #regressor=DecisionTreeRegressor()
                     # fit the regressor with x_train and y_train data
                     # y_pred=result.predict(x_test)
                        result=regressor.fit(x_train, y_train)
                        lst=[]
                        lst.append(store)
                        lst.append(Type)
                        lst.append(np_append(Size))
                        lst.append(dept)
                        lst.append(Holiday)
                        lst.append(np_append(Temperature))
                        lst.append(np_append(Fuel_Price))
                        lst.append(np_append(MarkDown1))
                        lst.append(np_append(MarkDown2))
                        lst.append(np_append(MarkDown3))
                        lst.append(np_append(MarkDown4))
                        lst.append(np_append(MarkDown5))
                        lst.append(np_append(CPI))
                        lst.append(np_append(Unemployment))
                        predict=np.array([lst])
               
                        sales_predict=result.predict(predict)
                        succuss=Image.open(r'final_project/invesment sucusss.png')
                        loss=Image.open(r'C:\Users\user\Desktop\python\final_project\invesment loss.jfif')
                        growth=Image.open(r'C:\Users\user\Desktop\python\final_project\sales growth.jfif')
                     
                        st.write("Your Weekly Sales ","Rs.",sales_predict[0])

                        if sales_predict <= 40000 and sales_predict >= 30000 :
                            st.image(succuss)
                            st.write("Your weekly sales Good")
                        elif sales_predict >= 40000 :
                            st.image(growth)
                            st.write("Your weekly sales Excellent")
                        elif sales_predict >= 10000  and sales_predict <= 30000 :
                            st.image(succuss)
                            st.write("Your weekly sales Medium")
                        elif sales_predict<=10000 and sales_predict>=0:
                            st.image(loss)
                            st.write("Your weekly sales bad")
                        else:
                            st.image(loss)
                            st.write("Your weekly sales bad")
                     
                      except:
                         st.write('Please enter the correct value...')
                #else:
                 #   st.write('Please enter the correct value...')
                
               

if selected=='Visualization & Insights':
    with st.spinner('Connecting...'):
            time.sleep(1)
        
    tab1,tab2,tab3 = st.tabs(["$\huge Accuracy 🚀 metrics $", "$\huge 🚀 Visualization$",'$\huge 🚀 Insights$'])

    with tab1:
     col1,col2=st.columns(2)
     # select any one algorithm # find Training and Testing accuracy of algorithm
     select=col1.selectbox('SELECT ALGORITHM ',['None','DecisionTreeRegressor','LinearRegression','RandomForestRegressor','AdaBoostRegressor','GradientBoostingRegressor','ElasticNet'])
       
     if select != 'None':
        # calculate Training and Testing accuracy of different algorithms and evaluate model performance using metrics
        accuracy=algorithm_train_test_accuracy(x_train, x_test, y_train, y_test, select)
        col1.markdown('### Accuracy of Train and Test')
        col1.write(accuracy)
        col2.markdown('### MSE & MAE & RMSE')
        col2.write(score_error(x_train, y_train,x_test,y_test,select))

     else:
         st.write('Choose Algorithm....')
    
    with tab2:
      col1,col2 = st.columns(2,gap='medium')
      with col1:
         st.markdown('### Data Visualization')
         st.markdown('Visualizing the Type of the Stores along with their percentage')
         dff=df2.merge(df3, on=['Store', 'Date','IsHoliday'], how='left')
         df=df1.merge(dff, on=['Store'], how='left')
         df.fillna(0, inplace=True)    
         count=df.Type.value_counts()
         # Defining colors for the pie chart 
         colors = ['pink', 'silver', 'steelblue'] 
         # Define the ratio of gap of each fragment in a tuple 
         explode = (0.05, 0.05, 0.05) 
         fig, ax = plt.subplots() 
  
         # Plotting the pie chart for above dataframe 
         ax=count.plot( 
         kind='pie',ylabel='Type_of_each_Store',autopct='%1.0f%%', colors=colors,explode=explode) 
         st.pyplot(fig)
         st.markdown('### Inference: ')
         st.markdown('    Here from the above pie chart it is clearly visible that Type c has the minimum number of stores while Type A has the maximum number of stores.')
         regressor=DecisionTreeRegressor()
        # fit the regressor with x_train and y_train data
         result=regressor.fit(x_train, y_train)
         importance_df = pd.DataFrame({
                   'feature': x.columns,
                   'importance': regressor.feature_importances_
                     }).sort_values('importance', ascending=False)
        
         fig=plt.figure(figsize=(10,6))
         plt.title('Feature Importance')
         sns.barplot(data=importance_df, x='importance', y='feature',palette='magma');
         st.markdown('### Feature Importance')
         st.pyplot(fig)
         
         fig = plt.figure(figsize = (12, 8), dpi=80)
         ax = fig.add_subplot(111, projection='3d')
         pnt3d = ax.scatter3D(df5['Store'],df5['Type'], df5['Size'],c=df5['Weekly_Sales'])
         cbar=plt.colorbar(pnt3d)
         cbar.set_label("Weekly sales")
         fig.set_facecolor('white')
         ax.set_facecolor('white')
         ax.set_xlabel('Store')
         ax.set_ylabel('Type')
         ax.set_zlabel('Size')
         st.markdown('### Weekly sales of each Store,Type,Size')
         st.pyplot(fig)

         # Top 10 of Store in Weekly sales
         df1 = df.groupby(["Store"]).size().reset_index(name="Weekly_Sales").sort_values(by='Weekly_Sales',ascending=False)[:10]
         fig = px.bar(df1,
                    title='Top 10 High Weekly_Sales',
                         x='Store',
                         y='Weekly_Sales',
                         orientation='h',
                         color='Store',
                         color_continuous_scale=px.colors.sequential.Agsunset)
         st.plotly_chart(fig,use_container_width=True) 

         fig = px.box(data_frame=df.groupby(['Store','Dept','IsHoliday']).size().reset_index(name='Temperature'),
                     x='Dept',
                     y='Temperature',
                     color='IsHoliday',
                     title='Holidays time Temperature each Store & Dept'
                    )
         st.plotly_chart(fig,use_container_width=True)
        
    
        
      with col2:
        def scatter(column):
           fig= plt.figure()
           df=df5.query('Weekly_Sales <=0')
           plt.scatter(df[column] , df['Weekly_Sales'],c=df['Weekly_Sales'])
           plt.ylabel('Weekly_Sales of Negative')
           plt.xlabel(column)
           st.pyplot(fig)
        st.markdown('### Weekly sales of Negatives each Fuel price,size,CPI,Type,Isholidays,Unemployment')
        scatter('Fuel_Price')  # with respect to Fuel_Price
        scatter('Size')  # with respect to Size
        scatter('CPI')  # with respect to CPI
        scatter('Type')  # with respect to Type
        scatter('IsHoliday') # with respect to IsHoliday
        scatter('Unemployment')  # with respect to Unemployment
        st.markdown('#### In the above diagram, we have seen the relationship between:')

        st.markdown('Weekly sales vs Fuel price')
        st.markdown('Weekly sales vs size of the store')
        st.markdown('Weekly sales vs CPI')
        st.markdown('Weekly sales vs type of departments')
        st.markdown('Weekly sales vs Holidays')
        st.markdown('Weekly sales vs Unemployment')
        st.markdown('Weekly sales vs Temperature')
        st.markdown('Weekly sales vs store')
        st.markdown('Weekly sales vs Departments')

    with tab3:
         select=st.selectbox('Select Markdowns',[None,'MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5'])
         if select=='MarkDown1':
             Markdown_effect(select)
         elif select=='MarkDown2':
              Markdown_effect(select)
         elif select=='MarkDown3':
              Markdown_effect(select)
         elif select=='MarkDown4':
              Markdown_effect(select)
         elif select=='MarkDown5':
              Markdown_effect(select)
        
if selected == 'About':
   col1,col2=st.columns(2)
   col1.markdown('### Store Sales Prediction – Forecasting')
   col1.markdown('Sales forecasting is the process of estimating the future sales of a product or service. It is a crucial part of any company and its business plan, as it helps businesses make informed decisions about resource allocation, marketing strategy, and investment.')
   col1.markdown('### Data Availability:')
   col1.markdown('### stores_data_set.csv: ')
   col1.markdown('    This file contains anonymized information about the 45 stores, indicating the type and size of store.')
   col1.markdown("The 45 Store in Type A,B,C is the category of each department")
   col1.markdown("A -  high in weekly sales")
   col1.markdown("B - Average of weekly sales")
   col1.markdown("C - decrease in weekly sales")
   col1.markdown("### sales_data_set.csv: ")
   col1.markdown("    This is the historical training data, which covers to 2010-02-05 to 2012-11- 01, Within this file you will find the following fields:")
   col1.markdown("Store – the store number")
   col1.markdown("Dept – the department number")
   col1.markdown("Date – the week")
   col1.markdown("Weekly_Sales – sales for the given department in the given store")
   col1.markdown("IsHoliday – whether the week is a special holiday week")
   
   col2.markdown('### features.csv: ')
   col2.markdown('   This file contains additional data related to the store, department, and regional activity for the given dates. It contains the following fields:')
   col2.markdown('Store – the store number')
   col2.markdown('Date – the week')
   col2.markdown("Temperature – average temperature in the region")
   col2.markdown("Fuel_Price – cost of fuel in the region")
   col2.markdown("MarkDown1-5 – MarkDown data is only available 2010, 2011,2012  and is not available for all stores all the time. Any missing value is marked with an NA.")
   markdown=Image.open(r'final_project/markdown1.jfif')
   col2.image(markdown)
   col2.markdown('CPI – the consumer price index')
   col2.markdown('Unemployment – the unemployment rate')
   col2.markdown("IsHoliday – whether the week is a special holiday week")
   col2.markdown('### Merching Three Csv file:')
   col2.markdown('  ')
   col2.markdown('While looking at the features it is evident that stores CSV files have “Store” as a repetitive column so it’s better to merge those columns to avoid confusion and to add the clarification in the dataset for future visualization.')
   col2.markdown('Using the merge function to merge and we are merging along the common column named Store')
   # check negative weekly_sales count
   col1.markdown("### Negative weekly_sales")
   qry=df[df['Weekly_Sales']<=0]
   col1.write(qry.sort_values(by='Weekly_Sales',ascending=False))
   col1.markdown("### Negative weekly_sales Count")
   col1.write(len(qry))
   
   col1.markdown('### Conclusion')
   col1.markdown('   We examined the store’s sales forecasting dataset by applying various statistical and visualization techniques.')
   col1.markdown('   We trained and developed four ML models. We also concluded that for this problem, DecisionTree Regressor works best.')
   col1.markdown('-----------------------------------------------------')
   col1.text('Developed by M.Thanalakshmi - 2023')
   col1.text('Mail: thanalakshmi7558@gmail.com')
   
  # check positive weekly_sales 
   col2.markdown("### Positive Weekly_Sales")
   qry=df[df['Weekly_Sales']>0]
   col2.write(qry.sort_values(by='Weekly_Sales',ascending=False))
   col2.markdown("### Positive Weekly_Sales Count")
   col2.write(len(qry))
   col2.markdown('### Describing the dataset')
   des=df.describe().T
   col2.write(des)
   col2.markdown('Let’s develop a machine learning model for further analysis.')
   

if __name__ == '__main__':
	      main()


 

   
   
                     
                     
