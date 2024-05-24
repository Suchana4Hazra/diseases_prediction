import pickle
import time as t
import streamlit as st

diabetes_model = pickle.load(open('diabetes_detection_model.pkl','rb'))
heart_model = pickle.load(open('heart_diseases_detection_model.pkl','rb'))
parkinsons_model = pickle.load(open('parkinsons_detection_model.pkl','rb'))

st.sidebar.image("header_img.jpeg")
st.sidebar.subheader('[Please use Desktop site for better experience]')
#sidebar for navigation

activities = ["diabetes diseases prediction","heart diseases prediction","parkinsons diseases prediction"]
option = st.sidebar.selectbox("Choose one disease: ",activities)
if option == 'diabetes diseases prediction':
    st.title('Diabetes Prediction')
    st.image('diabetes_img.webp')
    Pregnancies = st.text_input("Enter no of Pregnancies")
    st.caption('Number of times the person has been pregnant.')
    Glucose = st.text_input("Enter glucose level")
    st.caption('Plasma glucose concentration a 2 hours in an oral glucose tolerance test')
    BloodPressure = st.text_input("Enter Blood Pressure")
    st.caption('Diastolic blood pressure (mm Hg)')
    SkinThickness = st.text_input("Enter skin thickness")
    st.caption('Triceps skinfold thickness (mm)')
    Insulin = st.text_input("Enter Insulin level")
    st.caption('2-Hour serum insulin (mu U/ml)')
    BMI = st.text_input("Enter BMI")
    st.caption('Body mass index (weight in kg/(height in m)^2)')
    DiabetesPedigreeFunction = st.text_input("Enter DiabetesPedigreeFunction")
    st.caption('Diabetes pedigree function (a function which scores likelihood of diabetes based on family history)')
    Age = st.text_input("Enter age")
    st.caption('Age in years')

    diag_analysis = ''

    if st.button("Predict diabetes"):
        predicted_value = diabetes_model.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        if(predicted_value == 1):
          diag_analysis = 'The patient is diabetic'
        else:
          diag_analysis = 'The patient is not diabetic'
        with st.spinner("just wait..."):
           t.sleep(2)
           st.success(diag_analysis)
    st.markdown(
        '''
        <html>
        <a href="https://www.kaggle.com/datasets/suchana16/multi-diseases-dataset?select=diabetes_dataset.csv">Sample Data</a>
        </html>
        ''',unsafe_allow_html=True
    )
          
if option == 'heart diseases prediction':   
    st.title('Heart Diseases Prediction')
    st.image('heart_img.jpeg')   
    age = st.number_input('Age of the Person')
    sex = st.number_input('Sex of the Person')
    cp = st.number_input('Chest pain types')
    trestbps = st.number_input('Resting Blood Pressure')
    chol = st.number_input('Serum Cholestoral in mg/dl')
    fbs = st.number_input('Fasting blood sugar > 120 mg/dl')
    restecg = st.number_input('Resting Electrocardiographic results')
    thalach = st.number_input('Maximum Heart Rate achieved')
    exang = st.number_input('Exercise Induced Angina')
    oldpeak = st.number_input('ST depression induced by exercise')
    slope = st.number_input('Slope of the peak exercise ST segment')
    ca = st.number_input('Mjor vessels colored by flourosopy')
    thal = st.number_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    heart_diagnosis = ''
    
    #Creating a button for prediction
    
    if st.button('Heart Test Result'):
        heart_prediction = heart_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        
        if (heart_prediction[0]==1):
            heart_diagnosis = 'The person is suffering from Heart disease'
            
        else:
            heart_diagnosis = 'The person is Not suffering from Heart disease'
            
            
        st.success(heart_diagnosis)
    st.markdown(
        '''
        <html>
        <a href="https://www.kaggle.com/datasets/suchana16/multi-diseases-dataset?select=heart_dataset.csv">Sample Data</a>
        </html>
        ''',unsafe_allow_html=True
    )


if option == 'parkinsons diseases prediction':
    st.title('Parkinsons Prediction using ML')
    st.image('parkinsons_img.png')
    

    fo = st.text_input('MDVP:Fo(Hz)')
    fhi = st.text_input('MDVP:Fhi(Hz)')
    flo = st.text_input('MDVP:Flo(Hz)')
    Jitter_percent = st.text_input('MDVP:Jitter(%)')
    Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
    RAP = st.text_input('MDVP:RAP')
    PPQ = st.text_input('MDVP:PPQ')
    DDP = st.text_input('Jitter:DDP')
    Shimmer = st.text_input('MDVP:Shimmer')
    Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
    APQ3 = st.text_input('Shimmer:APQ3')
    APQ5 = st.text_input('Shimmer:APQ5')
    APQ = st.text_input('MDVP:APQ')
    DDA = st.text_input('Shimmer:DDA')
    NHR = st.text_input('NHR')
    HNR = st.text_input('HNR')
    RPDE = st.text_input('RPDE')
    DFA = st.text_input('DFA')
    spread1 = st.text_input('spread1')
    spread2 = st.text_input('spread2')
    D2 = st.text_input('D2')
    PPE = st.text_input('PPE')
    
    parkinsons_diagnosis = ''

    if st.button('Parkinsons Test Result'):
            parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]])
            
            if (parkinsons_prediction[0]==1):
                parkinsons_diagnosis = 'The person is suffering from Parkinsons disease'
                
            else:
                parkinsons_diagnosis = 'The person is Not suffering from Parkinsons disease'
                
                
            st.success(parkinsons_diagnosis)
    st.markdown(
        '''
        <html>
        <a href="https://www.kaggle.com/datasets/suchana16/multi-diseases-dataset?select=parkinsons_dataset.csv">Sample Data</a>
        </html>
        ''',unsafe_allow_html=True
    )



st.sidebar.image('healthcare_img.jpeg')
st.sidebar.markdown(
    '''
    <html>
        <a href="https://www.kaggle.com/datasets/suchana16/multi-diseases-dataset">Sample Data</a>
    </html>
    ''',unsafe_allow_html=True
)

st.sidebar.subheader('Accuracy of detecting diabetes disease')
st.sidebar.progress(77)
st.sidebar.subheader('Accuracy of detecting heart disease')
st.sidebar.progress(84)
st.sidebar.subheader('Accuracy of detecting parkinsons disease')
st.sidebar.progress(76)

st.sidebar.markdown('---') #separator
st.sidebar.subheader('Report a Bug')
st.sidebar.markdown(
    '''
    For any discrepancies, please [report a bug](mailto:suchanahazra99@gmail.com).
    '''
)

st.sidebar.markdown(
    
    """
       <style>
         .footer-text{
         
              color:Tomato;
              margin-left: 70%;
              font-style: italic;
         }
       </style>
    """,unsafe_allow_html = True
)
footer_html = """
        <footer>
            <p class="footer-text">Developed by <b>Suchana Hazra</b></p>
        </footer>
    """
st.sidebar.markdown(footer_html, unsafe_allow_html=True)
    
       
