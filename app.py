import pandas as pd
import joblib
import streamlit as st
from pycaret.regression import *

#@app.route('/')
def welcome():
    return "Welcome To Car Price Prediction"

#@app.route('/predict')
def car_price_predictor(test_data):
    GBR = joblib.load('mymodel.pkl')
    print(GBR)
    prediction = GBR.predict(test_data)
    print(prediction) 
    return prediction

def preprocess_test_data(df):
    df = df.drop("Unnamed: 0",axis=1)
    df = df.drop("car_name",axis=1)
    df['avg_car_price'] = (df['min_cost_price'] + df['max_cost_price'])/2
    df.drop(columns= ['min_cost_price', 'max_cost_price'], inplace = True)
    data = df
    data = data[data['km_driven'] < 1000000]
    data = data[data['mileage'] < 100]
    data = data[data['engine'] < 6100]
    data = data[data['max_power'] < 530]
    data = data.reset_index(drop=True)
    return data
    
    
def convert_df(result_df):
    return result_df.to_csv().encode('utf-8')


def read_file():
    uploaded_file = st.file_uploader("Choose csv file")
    if uploaded_file is not None:
        test_csv = pd.read_csv(uploaded_file)
        st.write(test_csv)  
    return test_csv
 
 
def main():
    st.title("CAR PRICE PREDICTION")
    html_temp = """
    <div style="background-color:red;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Car Price Prediction  ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)  
    st.write(welcome()) 
    

    df_csv = read_file()
    test_data = preprocess_test_data(df_csv)
    st.write(test_data)

    predictions = car_price_predictor(test_data)
    st.write(predictions)
    
    result_df = pd.DataFrame(predictions)
    csv = convert_df(result_df)
    st.download_button("Press to Download",csv,"result.csv","text/csv",key='download-csv')

if __name__=='__main__':
    main()
    
