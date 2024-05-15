
import pickle
import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu
from PIL import Image
import os

# loading the saved models
current_dir = os.path.dirname(os.path.abspath(__file__))
diabetes_model_path = os.path.join(current_dir,  "diabetes_model.sav")

heart_disease_model_path = os.path.join(current_dir,  "heart_disease_model.sav")

pneumonia_model_path = os.path.join(current_dir,  "xraymodel.sav")

# Now you can load your model
diabetes_model = pickle.load(open(diabetes_model_path, 'rb'))

heart_disease_model = pickle.load(open(heart_disease_model_path,'rb'))

pneumonia_model= pickle.load(open(pneumonia_model_path,'rb'))


# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('ML-Driven System for Predicting Multiple Diseases',
                        ['Diabetes', 'Heart Disease'],
                        icons=['activity','heart','lungs'],
                        default_index=0)

    
    
# Diabetes Prediction Page
        
if (selected == 'Diabetes'):
    
    # Title of the page
    st.title('Diabetes Prediction using ML')
    
    
    # Getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies', placeholder='Enter number of pregnancies')
        
    with col2:
        Glucose = st.text_input('Glucose Level (mg/dL)', placeholder='Enter glucose level')
    
    with col3:
        BloodPressure = st.text_input('Blood Pressure (mmHg)', placeholder='Enter blood pressure value')
    
    with col1:
        SkinThickness = st.text_input('Skin Thickness (mm)', placeholder='Enter skin thickness value')
    
    with col2:
        Insulin = st.text_input('Insulin Level (mu U/ml)', placeholder='Enter insulin level')
    
    with col3:
        BMI = st.text_input('BMI (kg/m^2)', placeholder='Enter BMI value')
    
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function', placeholder='Enter diabetes pedigree function value')
    
    with col2:
        Age = st.text_input('Age of the Person', placeholder='Enter age')
    
    
    # code for Prediction
    diab_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        
        if (diab_prediction[0] == 1):
          diab_diagnosis = 'The person is diabetic'
        else:
          diab_diagnosis = 'The person is not diabetic'
        
    st.success(diab_diagnosis)




# Heart Disease Prediction Page
if (selected == 'Heart Disease'):
    
    # page title
    st.title('Heart Disease Prediction using ML')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')
        
    with col2:
        sex_mapping = {'male': 0, 'female': 1}
        sex_selected = st.selectbox('Sex', ['male', 'female'], index=0)

        sex = sex_mapping[sex_selected]
        
    with col3:
        cp_mapping = {'Typical angina': 0, 'Atypical angina': 1, 'Non-anginal pain': 2, 'Asymptomatic': 3}
        cp_selected = st.selectbox('Chest Pain Type', ['Typical angina', 'Atypical angina', 'Non-anginal pain', 'Asymptomatic'], index=0)

        cp = cp_mapping[cp_selected]

        # cp = st.selectbox('Chest Pain Type', ['Typical angina', 'Atypical angina', 'Non-anginal pain', 'Asymptomatic'], index=0)
        
    with col1:
        trestbps = st.text_input('Resting Blood Pressure',placeholder="120 mmHg")
        
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl', placeholder="139 mg/dL")
        
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl', placeholder="120mg/dL")
        
    with col1:
        restecg_mapping = {'Normal': 0, 'Abnormality': 1}
        restecg_selected = st.selectbox('Resting Electrocardiographic results', ['Normal', 'Abnormality'], index=0)
        restecg = restecg_mapping[restecg_selected]
        
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved',placeholder="150 bpm")
        
    with col3:
        exang = st.radio('Exercise Induced Angina', ['0 : No', '1 : Yes'], index=0)
        
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise',placeholder="1.4 mm")
        
    with col2:
        slope = st.selectbox('Slope of the peak exercise ST segment', ['0: Up', '1: Flat', '2: Down'], index=0)
        
    with col3:
        ca_mapping = {'0 vessels': 0, '1 vessel': 1, '2 vessels': 2, '3 vessels': 3}
        ca_selected = st.selectbox('Major vessels colored by fluoroscopy', list(ca_mapping.keys()), index=0)
        ca = ca_mapping[ca_selected]

    with col1:
        thal_mapping = {'Normal': 0, 'Fixed damage': 1, 'Reversible damage': 2}
        thal_selected = st.selectbox('Thal', list(thal_mapping.keys()), index=0)
        thal = thal_mapping[thal_selected]

        # code for Prediction
    heart_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Get Result'):
        heart_prediction = heart_disease_model.predict([[float(age), float(sex), float(cp), float(trestbps), float(chol), float(fbs), float(restecg), float(thalach), float(exang), float(oldpeak), float(slope), float(ca), float(thal)]])
        
        if (heart_prediction[0] == 1):
          heart_diagnosis = 'The person is likely has a heart disease'
        else:
          heart_diagnosis = 'The person likely does not have heart diseases'
        
    st.success(heart_diagnosis)

if (selected == 'Pneumonia'):
    
        
    # def main():
    #     # pick=open('/Users/muzz/CodeCon-Team-20/Multiple Disease Prediction System/saved models/xraymodel.sav','rb')
    #     # modell=pickle.load(pick)
    #     st.title("PNEUMONIA PREDICTOR")
    #     test_image = st.file_uploader("Upload file: {}", type=['jpeg'])
    #     diagnose = ""
    #     # if test_image:
    #     #     diagnose= xray(test_image)
    #     if st.button("Diagnose"):
    #         diagnose= xray(test_image)
    #     st.success(diagnose)
        
    # if __name__ == "__main__":
    #     main()  
    
    def xray(test_image):
        image=Image.open(test_image)
        new_image = image.resize((50, 50))
        final_image = np.array(new_image).flatten().reshape(1, -1)

        predition =pneumonia_model.predict(final_image)


        categories = ['PNEUMONIA', 'NORMAL']

        if categories[predition[0]]=="NORMAL":
            return "Normal, Accuracy of prediction: 0.9624233128834356"
        else:
            return "PNEUMONIA, Accuracy of prediction: 0.9624233128834356"
        
    def main():
        # pick=open('/Users/muzz/CodeCon-Team-20/Multiple Disease Prediction System/saved models/xraymodel.sav','rb')
        # modell=pickle.load(pick)
        st.title("PNEUMONIA PREDICTOR")
        test_image = st.file_uploader("Upload file: {}", type=['jpeg'])
        diagnose = ""
        # if test_image:
        #     diagnose= xray(test_image)
        if st.button("Diagnose"):
            diagnose= xray(test_image)
            # print("Accuracy : 0.9624233128834356")
        st.success(diagnose)
        
    if __name__ == "__main__":
        main() 

        
        

#DISCLAIMER MESSAGE

st.info('DISCLAIMER! : The results are not 100% accurate and the user should consult a doctor for further diagnosis')
