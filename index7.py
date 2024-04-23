import streamlit as st
import numpy as np
import pandas as pd
import pickle

file11=open("scale3.pkl","rb")
file12=open("model3.pkl","rb") 

scale3=pickle.load(file11)
model3=pickle.load(file12)

st.write('# Insurance prediction')
age=st.number_input("Enter age: (Age of insured person)",step=1,format='%d')
gender=st.number_input("Enter Gender 1-Male/0-Female : ",format='%d',value=0)
try:
    if gender>=0 and gender<=1 :
        st.write('valid')
except:
    st.write('Please enter 1/0')

st.write('###### Below 18.5=Underweight, 18.5-24.9=HealthyWeight, 25.0-29.9=Overweight, 30.0andAbove=Obesity')    
bmi=st.number_input("Enter bmi:") # float
children=st.number_input("Enter children 1-5 :",step=1,format='%d')
smoker=st.number_input("Enter if smoking 0:No / 1:yes :",step=1,format='%d',value=0)

st.write('###### 0: northeast, 1:northwest, 2:southeast, 3:southwest')
region=st.number_input("Enter region 0-3 (From which region of the city does the person belong from)",step=1,format='%d',value=0)
charges=st.number_input("Enter charges:") 
            

if st.button('Predict'):                                                                                                                     
    #create a list 
    features=[age,gender,bmi,children,smoker,region,charges]
    # convert list features into 2D numpy array
    features=np.array([features])                # convert to 2D array        
    # Apply standard scaler on input features
    features=scale3.transform(features)
    # predict the model , use inbuilt method predict()
    Y_pred=model3.predict(features)[0]     # giving index to avoid metrics
    #st.write(Y_pred)
    if Y_pred==0:
        st.write('Customer will not claim for insurance')
    else:
        st.write('Customer will claim for insurance')