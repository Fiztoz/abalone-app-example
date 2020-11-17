import streamlit as st
import pandas as pd
import pickle


st.write("""
# Hello!
""")

st.sidebar.header('User Input')
st.sidebar.subheader('Please enter your data:')

def get_input():

    #Widgets
    v_sex = st.sidebar.radio('Sex', ['Male','Female','Infant'])
    v_length = st.sidebar.slider('Length',0.075,0.745,0.506)
    v_diameter = st.sidebar.slider('Diameter',0.055,0.6,0.4)
    v_height = st.sidebar.slider('Height',0.01,0.24,0.13)
    v_whole_weight = st.sidebar.slider('Whole Weight',0.002,2.55,0.78)
    v_shucked_weight = st.sidebar.slider('Shucked Weight',0.001,1.0705,0.3)
    v_Viscera_weight = st.sidebar.slider('Viscera Weight', 0.0005, 0.54, 0.17)
    v_Shell_weight = st.sidebar.slider('Shell Weight', 0.0015, 1.0, 0.24)


    #Condion change
    if v_sex == 'Male': v_sex = 'M'
    elif v_sex == 'Female': v_sex = 'F'
    else: v_sex = 'I'

    #dictionary
    data = {'Sex': v_sex,
            'Length':v_length,
            'Diameter':v_diameter,
            'Height':v_height,
            'Whole_weight':v_whole_weight,
            'Shucked_weight':v_shucked_weight,
            'Viscera_weight':v_Viscera_weight,
            'Shell_weight':v_Shell_weight}


    #create data frame
    data_df = pd.DataFrame(data, index=[0])
    return data_df


df = get_input()
# st.write(df) ##dataframe from widgets

# Combines user input data with sample dataset
# The sample data contains unique values for each nominal features
# This will be used for the One-hot encoding
data_sample = pd.read_csv('abalone_sample_data.csv')
df = pd.concat([df, data_sample],axis=0)
st.write(df) ## Dataframe after concat
#One-hot encoding for nominal features
cat_data = pd.get_dummies(df[['Sex']])
# st.write(cat_data) ##Dataframe OneHand for Sex
#Combine all transformed features together
X_new = pd.concat([cat_data, df], axis=1)
X_new = X_new[:1] # Select only the first row (the user input data)
#Drop un-used feature
X_new = X_new.drop(columns=['Sex'])
# st.write(X_new) ##One-Hand Code Dataframe for widgets


# -- Reads the saved normalization model
load_nor = pickle.load(open('normalization.pkl', 'rb'))
#Apply the normalization model to new data
X_new = load_nor.transform(X_new)
# st.write(X_new) ##Transform_dataframe


# -- Reads the saved classification model
load_knn = pickle.load(open('best_knn.pkl', 'rb'))
# Apply model for prediction
prediction = load_knn.predict(X_new)
st.write("""
# Prediction:
""")
st.write(prediction)