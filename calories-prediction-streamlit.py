import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import json
import requests
from streamlit_lottie import st_lottie
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import time

from matplotlib import style
style.use("seaborn-v0_8")
from IPython.display import HTML
import plotly.express as px

def load_lottiefile(filepath:str):
  with open(filepath,"r") as f:
    return json.load(f)

def load_lottieurl(url:str):
  r=requests.get(url)
  if r.status_code!=200:
    return None
  return r.json()

lottie_good=load_lottiefile("lottiefiles/good.json")
lottie_greatjob=load_lottiefile("lottiefiles/greatwork.json")
lottie_yourock=load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_tzgci2yi.json")

def predict(dataframe):
  st.header("Prediction : ")
  latest_iteration = st.empty()
  bar = st.progress(0)
  for i in range(100):
   # Update the progress bar with each iteration
    bar.progress(i + 1)
    time.sleep(0.01)
  pre=round(prediction[0],2)
  st.write(pre , " 🎉**kilocalories**")
  if pre>=25 and pre<=60:
    st_lottie(lottie_good,key="good")
  elif pre>60 and pre<=80:
    st_lottie(lottie_greatjob,key="greatjob")
  elif pre>80:
    st_lottie(lottie_yourock,key="yourock")

def model(dataframe):
  st.title("Random Forest Regression")
  st.write("** Random forest is a supervised machine learning algorithm that uses an esemble learning method for classification.The trees in random forest run in parallel,meaning is no interaction between these trees while building the trees.")
  st.write("## RandomForest Mean Absolute Error(MAE) :" , round(metrics.mean_absolute_error(y_test , random_reg_prediction) , 2))
  #st.write("##Linear Regression Mean Absolute Error(MAE) : " , round(metrics.mean_absolute_error(y_test , linreg_prediction) , 2))


import warnings
warnings.filterwarnings('ignore')

st.write("## Calories burned Prediction")
st.image("https://weightowellnessllc.com/wp-content/uploads/2018/10/caloriesblogWWW-1024x480-1-1280x720.png" , use_column_width=True)
st.write("In this WebApp you will be able to observe your predicted calories burned in your body.The Only thing you have to do is pass your parameters such as `Age` , `Gender` , `BMI` , `Body temperature` ,`Heart beat` into this WebApp and then you will be able to see the predicted value of kilocalories that burned in your body, and you can also see the similar results for your data.")
st.write("**Normal Heart_beat of a person is`74`beats per minute**")
st.write("**Normal Body Temperature of a person is `37` (C)**")

option =st.selectbox('Contents', ('Home','Prediction','Model','Similar Results'))

st.sidebar.header("User Input Parameters : ")

def user_input_features():
    global age , duration , heart_rate , body_temp, height,weight
    age=st.sidebar.text_input("Age")
    st.sidebar.write(age)
    height=st.sidebar.text_input("Height(in cm)")
    st.sidebar.write(height)
    weight=st.sidebar.text_input("Weight")
    st.sidebar.write(weight)
    duration=st.sidebar.text_input("Duration(in min)")
    st.sidebar.write(duration)
    heart_rate = st.sidebar.text_input("Heart_beat")
    st.sidebar.write(heart_rate)
    body_temp = st.sidebar.text_input("Body_Temp(in celsius")
    st.sidebar.write(body_temp)
    gender_button = st.sidebar.radio("Gender : ", ("Male" , "Female"))

    if gender_button == "Male":
        gender = 1
    else:
        gender = 0

    data = {
    "age" : age,
    "height" :height,
    "weight" :weight,
    "duration" : duration,
    "heart_rate" : heart_rate,
    "body_temp" : body_temp,
    "gender" : ["Male" if gender_button == "Male" else "Female"]
    }

    data_model = {
    "age" : age,
    "height" :height,
    "weight" :weight,
    "duration" : duration,
    "heart_rate" : heart_rate,
    "body_temp" : body_temp,
    "gender" : gender
    }

    features = pd.DataFrame(data_model, index=[0])
    data = pd.DataFrame(data, index=[0])
    return features , data

df , data = user_input_features()
st.write("---")
st.header("Your Parameters : ")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
  # Update the progress bar with each iteration
  bar.progress(i + 1)
  time.sleep(0.01)
st.write(data)

calories = pd.read_csv("calories.csv")
exercise = pd.read_csv("exercise.csv")

exercise_df = exercise.merge(calories , on = "User_ID")
# st.write(exercise_df.head())
exercise_df.drop(columns = "User_ID" , inplace = True)

exercise_train_data , exercise_test_data = train_test_split(exercise_df , test_size = 0.2 , random_state = 1)


exercise_train_data = exercise_train_data[["Gender" , "Age" , "Height" ,"Weight" , "Duration" , "Heart_Rate" , "Body_Temp" , "Calories"]]
exercise_test_data = exercise_test_data[["Gender" , "Age" ,"Height" , "Weight" , "Duration" , "Heart_Rate" , "Body_Temp" , "Calories"]]
exercise_train_data = pd.get_dummies(exercise_train_data, drop_first = True)
exercise_test_data = pd.get_dummies(exercise_test_data, drop_first = True)

X_train = exercise_train_data.drop("Calories" , axis = 1)
y_train = exercise_train_data["Calories"]

X_test = exercise_test_data.drop("Calories" , axis = 1)
y_test = exercise_test_data["Calories"]

random_reg = RandomForestRegressor(n_estimators = 1000 , max_features = 3 , max_depth = 6)
random_reg.fit(X_train , y_train)
random_reg_prediction = random_reg.predict(X_test)

#st.write("RandomForest Mean Squared Error(MSE) : " , round(metrics.mean_squared_error(y_test , random_reg_prediction) , 2))
#st.write("RandomForest Root Mean Squared Error(RMSE) : " , round(np.sqrt(metrics.mean_squared_error(y_test , random_reg_prediction)) , 2))
prediction = random_reg.predict(df)

#linreg = LinearRegression()
#linreg.fit(X_train , y_train)
#linreg_prediction = linreg.predict(X_test)

#st.write("Linear Regression Mean Absolute Error(MAE) : " , round(metrics.mean_absolute_error(y_test , linreg_prediction) , 2))

def similar(dataframe):
  st.header("Similar Results : ")
  latest_iteration = st.empty()
  bar = st.progress(0)
  for i in range(100):
    # Update the progress bar with each iteration
    bar.progress(i + 1)
    time.sleep(0.01)

  range1 = [prediction[0] - 10 , prediction[0] + 10]
  ds = exercise_df[(exercise_df["Calories"] >= range1[0]) & (exercise_df["Calories"] <= range1[-1])]
  st.write(ds.sample(4))

if  option== 'Prediction':
  predict(df)
elif option=='Model':
  model(df)
elif option=='Similar Results':
  similar(df)
