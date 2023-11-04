import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
pick= open('C:/Users/mdmux/Desktop/muzaffarmhd/Projects/CodeCon-Team-20-main/Multiple Disease Prediction System/saved models/xraymodel.sav','rb')

modell=pickle.load(pick)
pick.close()

def xray(test_image):
    image=Image.open(test_image)
    new_image = image.resize((50, 50))
    final_image = np.array(new_image).flatten().reshape(1, -1)

    predition =modell.predict(final_image)


    categories = ['PNEUMONIA', 'NORMAL']

    if categories[predition[0]]=="NORMAL":
        return "Normal"
    else:
        return "PNEUMONIA"

    xray=final_image[0].reshape(50,50)

def main():
    st.title("PNEUMONIA PREDICTOR")
    test_image = st.file_uploader("Upload file: {}", type=['jpeg'])
    diagnose = ""
    # if test_image:
    #     diagnose= xray(test_image)
    if st.button("Diagnose"):
        diagnose= xray(test_image)
    st.success(diagnose)
        
if __name__ == "__main__":
    main()       

