import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
from sklearn.model_selection import train_test_split
@st.cache 
def loaddata():
        return pd.read_csv("house.csv")
@st.cache
def train_model(data):
        X=data.drop("id",axis=1)
        X=X.drop("date",axis=1)
        X=X.drop("sqft_above",axis =1)
        X=X.drop("zipcode",axis =1)
        X=X.drop("lat",axis =1)
        X=X.drop("long",axis =1)
        X=X.drop("price",axis=1)
        X=X.drop("sqft_lot15",axis=1)
        Y=data.price
        Xtrain,Xtest,Ytrain,Ytest=train_test_split(X, Y, test_size=0.1, random_state=2) 
        sc= StandardScaler()
        Xtrain = sc.fit_transform(Xtrain)
        Xtest = sc.transform(Xtest)
        reg = RandomForestRegressor(n_estimators=20, random_state=0)
        reg.fit(Xtrain, Ytrain)
        return(reg)
def main():
        dat=loaddata()
        reg=train_model(dat)
        st.sidebar.header('House price predictor')
        st.title('House price')
        bedrooms = st.number_input("No. of Bedrooms:")
        bathroom = st.number_input("No of bathrooms :")
        sqft_living =  st.number_input("Sqft living:")
        sqft_lot = st.number_input("Sqft lot:")
        floors = st.number_input("NO of floors:")
        waterfront = st.number_input("No of watefront:")
        view = st.number_input("No of view:")
        condition = st.number_input("Condition:")
        grade=st.number_input("grade")
        sqft_basement = st.number_input('Sqft Basement')
        yr_built= st.number_input('year of built')
        yr_renovated = st.number_input('Year of Renovated')
        submit = st.button('submit')
        if submit:
                prediction = reg.predict([[bedrooms,bathroom,sqft_living,sqft_lot,floors,waterfront,view ,condition,grade,sqft_basement,yr_built,yr_renovated]])
                st.write(prediction)
if __name__=="__main__":
        main()