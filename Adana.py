import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import joblib
#import plotly.graph_objs as go
#import plotly.figure_factory as ff

from PIL import Image
from streamlit_option_menu import option_menu
from sklearn import metrics
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler








data = pd.read_csv (r'C:\Users\achum\Desktop\PythonDiabetics_Project\diabetes.csv')
df = pd.DataFrame( data['Outcome'].value_counts())




X = data.drop("Outcome",axis=1)
Y = data["Outcome"]
 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20,  random_state = 42 )


sc=StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
  
svc_model = SVC()
svc_model.fit(X_train, Y_train)
 
svc_pred = svc_model.predict(X_test)

  





EXAMPLE_NO = 1


def streamlit_menu(example=1):
    if example == 1:
        # 1. as sidebar menu
        with st.sidebar:
            selected = option_menu(
                menu_title="Main Menu",  # required
                options=["Home", "DataSet", "Data description", "Predict Data", "Contact"],  # required
                icons=["house", "book", "book","activity", "envelope"],  # optional
                menu_icon="cast",  # optional
                default_index=0,  # optional
            )
        return selected




	
#########################################################################





	

selected = streamlit_menu(example=EXAMPLE_NO)

if selected == "Home":
 st.title("Welcome to my web Application for diabetic prediction")

 st.markdown ( 'Through this Web App I want to show you how to solve a real world problem using Data analysis technique. The data used for this project is a Diabetics dataset. To understand little bit about diabetics it is a “Metabolic disease that causes high blood sugar. The hormone insulin moves sugar from the blood into the cells to be stored or used for energy. With diabetes the body either doesn’t make enough insulin or cant effectively use the insulin it does make.” There are different types of diabetes and can damage the nerves, eyes, kidneys and other organs.')
 
 
 image = Image.open('Diab1.jpg')
 st.image(image, use_column_width = True)
 
 
 st.subheader('Types of Diabetics')
 
 st.markdown ('  Type 1 Diabetes -The immune system attacks and destroys cells in the pancreas where insulin is made. Note exactly know the reason for the cause of type 1 diabetes. Currently there are no known methods of prevention.')
 st.markdown ('	Type 2 Diabetes forms from a combination of genetics and lifestyle factors. Person with overweight or obese condition makes cells more resistant to the effects of insulin in the blood sugar.')
 st.markdown ('	Prediabetes – This occurs when the blood sugar is higher than normal, and this is not the condition for diagnosis ')
 st.markdown ('	Gestational diabetes – Is a condition of high blood sugar during pregnancy time. The placenta produces hormones that make a pregnant woman’s cells less sensitive to the effects of insulin.')
 
 
 st.subheader('Symptoms of Diabetics ')
 
 

 
 st.markdown ('Weight loss') 
 st.markdown('	Increase thirst and hunger')
 st.markdown('	Frequent urination or UTI issue') 
 st.markdown('	Some changes in vision ')
 st.markdown('	Extreme tiredness and Fatigue')
 st.markdown('	Slow heeling of wound')

 
 
 
 
 
 
 
 
 



 
 
 
 
 
 
 
 
 
 
 
 
########################################################################### 
 
 
 
 
if selected == "DataSet":
  st.title("Diabetics Data Set  ")
  st.markdown (' These Datasets are originally taken from “National Institute of Diabetic and Digestive and kidney Diseases. It provided courtesy of the Pima Indians Diabetic Database” .' )

  st.markdown (' Using this Dataset we are predicting whether the person have diabetics or likely to get diabetic in future . ')
  
  st.markdown('The dataset contains variables such as number of pregnancies the person has had , their BMI , insulin Level, age, Blood pressure, skin thickness, Glucose level, Diabetes pedigree function , outcome.') 
  
  
  
  
  
  data = pd.read_csv (r'C:\Users\achum\Desktop\PythonDiabetics_Project\diabetes.csv')
  df = pd.DataFrame( data['Outcome'].value_counts())
  st.write(data)
  

  
  
  st.subheader('Visualise  First 10 Patient Data')
  
  st.markdown(' To understand more clearly here we are taking first 10 patient data from the dataset and visualizing the data . Based on the nine variables in the data set we can see that person with highest graph are more likely to get diabeics .') 
  
  st.markdown(' This can also confirm by cheking the outcome in the data set .As the outcom = 0 the person have no diabetics and outcome = 1 means the person is likely to have diabetics  ')
  
  #df = pd.DataFrame( data['Outcome'].value_counts())
  data.info()
  st.bar_chart(data.head(10))
  data= data.head(10)
  st.write(data.head(10)) 
  
  
  
 #sc=StandardScaler()
 


 #X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)


	
	
 
 
 
 
 
 
 
 
 
 ########################################################################## 
	
	
if selected == "Data description":
 st.title("Describing the Dataset ")
 st.markdown(' This part explains the variables inside the dataset ,which includes what are the variables ,how this is related to diabetic , and analysing each variables with the outcome') 
 st.markdown(' 	Number of pregnancies  -  Number of pregnancies in the past . In some women diabetics can develop suddenly during the pregnancy time due to hormones produced by the placenta can make the body more resistant to the effects of insulin. This conditions will go away after the delivery ,however there may be risk of getting it later in life.')
 st.markdown ('	Glucose – Measuring the body’s response to sugar ,so for that we carry out a test called glucose tolerance test .This test are given in milligrams per deciliter (mg/dL) or millimoles per litter (mmol/L).The normal glucose level is lower than 140 mg/dL and the glucose level of 200 mg/dL or higher may indicate diabetes.')
 st.markdown(' 	Blood Pressure – Diastolic blood pressure (mm/Hg) .BP should be below 140/80 Hg  for people with diabetics . ')
 st.markdown (' 	Skin thickness- Skin thickness is primarily determined by collagen content .Triceps skin fold thickness (mm) ')
 st.markdown (' 	Insulin – 2 hours serum insulin (mu U/ml) ')
 st.markdown (' 	BMI – body mass index (weight in kg/(height in m)^2. BMI chart will calculate your height and weight and tell you if your normal ,overweight or Obese category ')
 st.markdown(' 	Age -  Age (how old you are) ')
 st.markdown ('	Outcome – Class variables 0  or 1 . 1 represent that you have diabetic and 0 means no diabetics ')
 
 
  

  

 
 


 
 


	
if selected == "Predict Data":
	st.title("Diabetic predictor ")
	st.subheader ("Please select the below details for the diabetic prediction")
 
	pregnancies = st.slider('Pregnancies', 0,17, 3 )
	st.write( "Number of pregnancies",  pregnancies )
 
	glucose = st.slider('Glucose', 0,200, 120 )
	st.write( "Glucose Level",  glucose )
 
	bp = st.slider('Blood Pressure', 0,122, 70 )
	st.write( "Blood Pressure",  bp )
  
	skinthickness = st.slider('Skin Thickness', 0,100, 20 )
	st.write( "skinthickness is ",  skinthickness )
  
	insulin = st.slider('Insulin', 0,846, 79 )
	st.write( "insulin level ",  insulin )
  
	bmi = st.slider('BMI', 0,67, 20 )
	st.write( "Body Max index",  bmi )
  
	dpf = st.slider('Diabetes Pedigree Function', 0.0,2.4, 0.47 )
	st.write( "Diabetes Pedigree Function",  dpf )
  
	age = st.slider('Age', 21,88, 33 )
	st.write( " My Age is",  age  )
 	
	
	user_data = (pregnancies, glucose, bp, skinthickness, insulin, bmi, dpf, age)
	
	
	
	#user_report  = svc_model.predict(sc.transform(np.array.reshap([[int(pregnancies), int(glucose), int(bp), int(skinthickness), int(insulin),int( bmi) , float(dpf) , int(age)]])))
	
	st.text ( ' PATIENT DATASET FOR PREDICTION BASED ON THE INFORMATION ABOVE ') 
	
	st.text( 'Pregnancies, Glucose, Bp, Skinthickness, Insulin, Bmi, dpf, Age')
	st.write ( user_data )
	
	if st.button('Predict'):
	
		st.write('YES ')
		
	
		
	output = ''
	
	user_data = np.array(user_data).reshape(1 ,-1)
	
	print (svc_model.predict(user_data))
	
	
	if svc_model.predict(user_data)==0:
	
		output=  'This person NOT Diabetes'

	
	else:

		output = 'This person have Diabetes, please consult a Doctor'
	

	
	
	
	
	
	
	

	
	

	
	
	


