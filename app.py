import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

st.title('BostonHouse Prediction App')
indus = st.slider('proportion of non-retail business acres per town',min_value = 0.378436	, max_value=3.358290)
nox  =  st.slider( 'nitric oxides concentration (parts per 10 million) ',  min_value = 0.325700	, max_value=0.626473)  
rm  =  st.slider( 'average number of rooms per dwelling ',  min_value = 1.517542	, max_value=2.280339) 
age  =  st.slider( 'proportion of owner-occupied units built prior to 1940 ',  min_value = 2.900000	, max_value=100.000000)     
ptratio  =  st.slider( 'pupil-teacher ratio by town ',  min_value = 12.600000	, max_value=22.000000) 
lstat  =  st.slider(' % lower status of the population',  min_value = 1.004302	, max_value=3.662792	) 
submit = st.button('Predict')
if submit:
    input_list =[]
    input_list.append(indus)
    input_list.append(nox)
    input_list.append(rm)
    input_list.append(age)
    input_list.append(ptratio)
    input_list.append(lstat)
    
    #scaling the input values
    scaling= pickle.load(open('boston_scaler.pickle', 'rb'))
    input_scaled = scaling.transform([input_list])

    #converting input values as per polynomial features created with degree 2
    poly_values = pickle.load(open('boston_polyfeature.pickle', 'rb'))
    input_scaled_poly = poly_values.transform(input_scaled)

    input = np.array(input_scaled_poly).reshape(1,-1)
    
    #loading the stored model for prediction
    model = pickle.load(open('boston.pickle', 'rb'))
    output = model.predict(input)
    if output:
        st.write(output)
        
        
        