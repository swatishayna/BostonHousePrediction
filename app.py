import streamlit as st
from sklearn.preprocessing import StandardScaler
import pickle
from logisticregressionwiththresholdclass import LogisticRegressionwithThreshold
import numpy as np


st.title('1974Redbook Affair Prediction App')
occ_woman = st.radio('Occupation of Lady',  ('farming/semi-skilled/unskilled', '"white collar', 'teacher/nurse/writer/technician/skilled', 'managerial/business','professional with advanced degree','others'))
occ_man   = st.radio('Occupation of Man',  ('farming/semi-skilled/unskilled', '"white collar', 'teacher/nurse/writer/technician/skilled', 'managerial/business','professional with advanced degree','others'))
rate_marriage = st.radio('Rating Input of Marriage Life', ('VeryPoor','Poor','Good','VeryGood','Excellent'))
age = st.slider('Age of lady', min_value = 17, max_value=42)
yrs_married = st.slider('Number of Married Years', min_value = 0, max_value=23)
children = st.number_input('ChildrenCount', min_value =0, max_value =5)
religious = st.radio('Are You religious?',('Not at all', 'Yes but not much', 'Yes', 'Highly Religious'))
educ= st.radio('level of education', ('grade school','high school', 'some college', 'college graduate','some graduate school','advanced degree'))

predict =  st.button('Predict')




occupations = ['farming/semi-skilled/unskilled', 'white collar', 'teacher/nurse/writer/technician/skilled', 'managerial/business','professional with advanced degree','others']
d_woman_occ_value = [0,0,0,0,0]
d_man_occ_value = [0,0,0,0,0]
if predict:
    if occ_woman!='others':
        i = occupations.index(occ_woman)
        d_woman_occ_value[i]=1

    if occ_woman!='others':
        i = occupations.index(occ_woman)
        d_man_occ_value[i]=1

    d_rate_marriage = ['VeryPoor','Poor','Good','VeryGood','Excellent']
    rate_marriage_value = d_rate_marriage.index(rate_marriage) + 1
    d_religious = ['Not at all', 'Yes but not much', 'Yes', 'Highly Religious']
    religious_value = d_religious.index(religious) + 1
    d_edu =  ['grade school','high school', 'some college', 'college graduate','some graduate school','advanced degree']
    edu_value = d_edu.index(educ) + 1


    inputs =[]
    inputs.extend(d_woman_occ_value)
    inputs.extend(d_man_occ_value)
    inputs.append(rate_marriage_value)
    inputs.append(age)
    inputs.append(yrs_married)
    inputs.append(children)
    inputs.append(religious_value)
    inputs.append(edu_value)
    
    
    
    scaling= pickle.load(open('affair_scaler.pickle', 'rb'))
    input_scaled = scaling.transform([inputs])
    input = np.array(input_scaled).reshape(1,-1)
   
    output = LogisticRegressionwithThreshold('l1','liblinear').output(input)
    if output == 1:
        st.header('There are high chances that the woman would have extramarital affair')
    else:
        st.header('There are high chances that the woman would not have extramarital affair')
        
        