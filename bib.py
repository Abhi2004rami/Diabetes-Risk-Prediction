import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
import urllib.request
st.set_page_config(page_title="Diabetes ML Teaching Tool", layout="wide")
#data reading
data_path="diabetes.csv"
url="https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv"
if not os.path.exists(data_path):
    urllib.request.urlretrieve(url,data_path)
raw_data=pd.read_csv(data_path)
def clean_and_prepare_data(df):
    cleaned_df=df.copy()
    suspicious_columnns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in suspicious_columnns:
        cleaned_df[col] = cleaned_df[col].replace(0, np.nan)
    for col in suspicious_columnns:
         median_value = cleaned_df[col].median()
         cleaned_df[col] = cleaned_df[col].fillna(median_value)
    return cleaned_df
cleaned_data=clean_and_prepare_data(raw_data)
x=cleaned_data.drop('Outcome',axis=1)
y=cleaned_data['Outcome']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=RandomForestClassifier(n_estimators=100,max_depth=5,random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
st.title("🩺 Diabetes Risk Predictor ")
st.write("This app teaches you how to find 'hidden' missing values in medical data before training an AI.")
tab1, tab2 = st.tabs(["🧹 Data Cleaning Explained", "🔮 Patient Risk Predictor"])
with tab1:
    st.header("step 1:Transforming the data")
    coll,col2=st.columns(2)
    with coll:
        st.subheader("Raw Data(Notice the 0's)")
        st.dataframe(raw_data[['Glucose', 'BloodPressure', 'BMI']].head(10),
                        use_container_width=True)
        st.caption("Notice how some patients have a blood pressure or BMI of 0.That's impossible!")
    with col2:
        st.subheader("Cleaned Data(Ready for AI)")
        st.dataframe(cleaned_data[['Glucose', 'BloodPressure', 'BMI', 'Outcome']].head(10),
                    use_container_width=True)
        st.caption("The 0s were identified as missing data and replaced with the median values.")
    st.markdown("---")
    st.header("step 2:Model Accuracy")
    m_coll,m_col2=st.columns(2)
    m_coll.metric("Total patients Analyzed", len(cleaned_data))
    m_col2.metric("AI Accuracy(Test Data)",f"{accuracy:.2%}")
with tab2:
    st.header("Test the Model:Input patient vitals")
    st.write("Adjust the medical parameters below to see if the AI predicts a risk of diabetes.")
    p_coll,p_col2=st.columns(2)
    with p_coll:
         input_pregnancies=st.number_input("Number of Pregnancies",min_value=0,max_value=20,value=1)
         input_glucose=st.slider("Glucose Level:",min_value=50,max_value=250,value=120)
         input_bp=st.slider("Blood Pressure:",min_value=40,max_value=140,value=70)
         input_skin=st.slider("Skin Thickness(mm):",min_value=10,max_value=100,value=20)
    with p_col2:
            input_insulin=st.slider("Insulin Level(IU/mL):",min_value=14,max_value=846,value=79)
            input_bmi=st.slider("Body Mass Index(BMI):",min_value=15.0,max_value=60.0,value=25.0)
            input_dpf=st.slider("Diabetes Pedigree Function(Genetics):",min_value=0.05,max_value=2.5,value=0.5)
            input_age=st.slider("patient Age:",min_value=21,max_value=100,value=30)
    if st.button("Run Health Risk prediction",type="primary"):
        input_features=pd.DataFrame([[input_pregnancies,input_glucose,input_bp,input_skin,input_insulin,input_bmi,input_dpf,input_age]],columns=x.columns,)
        prediction=model.predict(input_features)[0]
        if prediction==1:
            st.error("The AI predicts a HIGH risk of diabetes. Please consult a healthcare professional.")
        else:
            st.success("The AI predicts a LOW risk of diabetes. Maintain a healthy lifestyle and regular check-ups.")