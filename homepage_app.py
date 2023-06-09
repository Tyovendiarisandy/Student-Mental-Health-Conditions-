import pandas as pd
import numpy as np
import streamlit as st
import sklearn 
import imblearn
import pickle 
from PIL import Image

# load dataset
df = pd.read_csv('new_clean_dataset.csv')

# load model
model = pickle.load(open('random_forest_fix_tuned.pkl','rb'))
    
# create title (homepage)
def main():
    load_image = Image.open('./image.png')
    st.image(load_image)
    st.title('The Student Mental Health Conditions Prediction')
    st.subheader('Please input in the option box below!')

    # choose menu input - Selectbox
    # st.sidebar.subheader('Select Your Input')
    gender = st.selectbox('Select Your Gender', df['gender'].unique())
    age = st.selectbox('Select Your Age', df['age'].unique())
    study_years = st.selectbox('Select Your Study Years', df['study_years'].unique())
    cgpa = st.selectbox('Select Your CGPA', df['cgpa'].unique())
    marital_status = st.selectbox('Select Your Marital Status', df['marital_status'].unique())

    # subtitle for symptoms
    st.subheader('Select Your Symptoms Below')
    
    # choose menu input - selectbox for symptoms
    depression = st.selectbox('Are You Feeling Depressed?', df['depression'].unique())
    anxiety = st.selectbox('Are You Feeling Anxious?', df['anxiety'].unique())
    panic_attack = st.selectbox('Are You Having a Panic Attack?', df['panic_attack'].unique())
    seeking_treatment = st.selectbox('Are You Looking for Treatment?', df['seeking_treatment'].unique())

    # prediction - button for predict
    if st.button('Predict'):
    # input the data in dataframe
        input_data = pd.DataFrame({
        'gender': [gender],
        'age': [age],
        'study_years': [study_years],
        'cgpa': [cgpa],
        'marital_status': [marital_status],
        'depression': [depression],
        'anxiety': [anxiety],
        'panic_attack': [panic_attack],
        'seeking_treatment': [seeking_treatment]
        
        })

        # do predict with model
        prediction = model.predict(input_data)

        st.subheader('Prediction Result')
        if prediction[0] == 0:
            st.success('The condition of your mental health problem is Normal')
        elif prediction[0] == 1:
            st.success('The condition of your mental health problem is Mild.')
        elif prediction[0] == 2:
            st.success('The condition of your mental health problem is Moderate.')
        elif prediction[0] == 3:
            st.success('The condition of your mental health problem is Severe.')
        else:
            st.success('Unknown Condition.')
    
    st.write('----')
    st.write('''
    Dashboard Created by [Tyovendi Arisandy](https://www.linkedin.com/in/tyovendiarisandy/)
    ''')

if __name__=='__main__':
    main()
