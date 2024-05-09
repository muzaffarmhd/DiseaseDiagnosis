
import pickle
import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu
import cv2
import matplotlib.pyplot as plt
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
                        ['Diabetes', 'Heart Disease', 'Pneumonia'],
                        icons=['activity','heart','lungs'],
                        default_index=0)

    
    
# Diabetes Prediction Page
if (selected == 'Diabetes'):
    
    # page title
    st.title('Diabetes Prediction using ML')
    
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
        
    with col2:
        Glucose = st.text_input('Glucose Level')
    
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    
    with col2:
        Insulin = st.text_input('Insulin Level')
    
    with col3:
        BMI = st.text_input('BMI value')
    
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    
    with col2:
        Age = st.text_input('Age of the Person')
    
    
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
        sex = st.text_input('Sex')
        
    with col3:
        cp = st.text_input('Chest Pain types')
        
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
        
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
        
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
        
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
        
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
        
    with col3:
        exang = st.text_input('Exercise Induced Angina')
        
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
        
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')
        
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')
        
    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    
        # code for Prediction
    heart_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Heart Disease Test Result'):
        heart_prediction = heart_disease_model.predict([[float(age), float(sex), float(cp), float(trestbps), float(chol), float(fbs), float(restecg), float(thalach), float(exang), float(oldpeak), float(slope), float(ca), float(thal)]])
        
        if (heart_prediction[0] == 1):
          heart_diagnosis = 'The person is having heart disease'
        else:
          heart_diagnosis = 'The person does not have any heart disease'
        
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
