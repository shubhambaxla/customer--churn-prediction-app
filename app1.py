import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder
import pandas as pd
import pickle

#loading the trained model
model=tf.keras.models.load_model('model.h5')
###load all our encoders
label_encoder_gender=pickle.load(open('label_encoder_gender.pkl','rb'))

dummy_columns=pickle.load(open('dummy_columns.pkl','rb'))

with open('scalar.pkl','rb')as file:
    scalar=pickle.load(file)
###starting with streamlit app


st.title('customer churn prediction')
st.header('enter customer details')

###for providing input by the 
#user i am gonna make 

# '''input_data = {
# 'CreditScore': 600,
# 'Geography': 'France',
# 'Gender': 'Male',
# 'Age': 40,
# 'Tenure': 3,
# 'Balance': 60000,
# 'NumOfProducts': 2,
# 'HasCrCard': 1,
# 'IsActiveMember': 1,
# 'EstimatedSalary': 50000} making moderration 
# like input data here in this format
# '''

# ---------------- USER INPUT ----------------
CreditScore=st.number_input('CreditScore',300,900,600)
Geography=st.selectbox('Geography',['France','Germany','Spain'])
Gender=st.selectbox('Gender',label_encoder_gender.classes_)
Age=st.slider('Age',18,100,35)
Tenure= st.slider('Tenure',0,10,3)
Balance=st.number_input('Balance',0.0,200000.0,50000.0)
NumOfProducts=st.slider('number of products',1,4,2,key='num_products_slider')
HasCrCard=st.selectbox('has cards',[0,1])
IsActiveMember=st.selectbox('is active member',[0,1])
EstimatedSalary=st.number_input('estimated Salary',0.0,200000.0,70000.0)


####flow if preiction BUtton in app
if st.button('predict churn'):
    #create input dictionary exaclty like traing columns
    input_data={
            'CreditScore':CreditScore,
            'Geography':Geography,
            'Gender':Gender,
            'Age':Age,
            'Tenure':Tenure,
            'Balance':Balance,
            'NumOfProducts':NumOfProducts ,
            'HasCrCard':HasCrCard,
            'IsActiveMember':IsActiveMember,
            'EstimatedSalary':EstimatedSalary}
    
#convert to data frame
    df=pd.DataFrame([input_data])
    df['Gender']=label_encoder_gender.transform(df['Gender'])
    df=pd.get_dummies(df)
    df=df.reindex(columns=dummy_columns,fill_value=0)

    if 'Exited' in df.columns:
        df=df.drop(columns=['Exited'])
        
    #scale
    df_scaled=scalar.transform(df)

    #predict
    prediction=model.predict(df_scaled)[0][0] ###output neuron shaped is 1 by 1 and indexing 0 by 0 row and col

    st.subheader('result')
    if prediction>=0.5:
        st.error(f"customer is likely to churn")
    else:
        st.success('customer will not churn')