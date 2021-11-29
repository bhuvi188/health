from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np

model = load_model('ment-pipeline')

def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions

def run():

        from PIL import Image
        image = Image.open('index.jpg')
        st.image(image,use_column_width=True)
        st.title("Mental Health Prediction App")

        Age = st.number_input('Enter Your Age', min_value=1, max_value=100, value=25)
        Gender = st.selectbox('Gender', ['male', 'female','Other'])
        tech_company=st.selectbox(' Is your employer primarily a tech company/organization?', ['Yes', 'No'])
        remote_work=st.selectbox('Do you work remotely (outside of an office) at least 50% of the time?', ['Yes', 'No'])
        family_history=st.selectbox('Do you have a family history of mental illness?', ['Yes', 'No'])
        obs_consequence=st.selectbox('Have you heard of or observed negative consequences for coworkers with mental health conditions in your workplace?', ['Yes', 'No'])
        mental_health_consequence=st.selectbox('Do you think that discussing a mental health issue with your employer would have negative consequences?', ['Yes', 'No','Maybe'])
        mental_health_interview=st.selectbox('Do you feel that your employer takes mental health as seriously as physical health?', ['Yes', 'No','Maybe'])
        no_employees=st.selectbox('How many employees does your company or organization have?',['1-5','6-25','26-100','100-500','500-1000','More than 1000'])
        leave=st.selectbox('How easy is it for you to take medical leave for a mental health condition?', ['Dont know','Somewhat easy','Very easy','Somewhat difficult','Very difficult'])
        self_employed=st.selectbox('Are you self-employed?', ['Yes', 'No'])
        seek_help=st.selectbox('Does your employer provide resources to learn more about mental health issues and how to seek help?', ['Yes', 'No','Dont know'])
        anonymity=st.selectbox('Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources?', ['Yes', 'No','Dont know'])
        phys_health_interview=st.selectbox('Would you bring up a physical health issue with a potential employer in an interview?', ['Yes', 'No','Maybe'])
        work_interfere=st.selectbox('If you have a mental health condition, do you feel that it interferes with your work?', ['Sometimes','unknown','Never','Rarely','Often'])
        care_options=st.selectbox('Do you know the options for mental health care your employer provides?', ['Yes', 'No','Not sure'])
        wellness_program=st.selectbox('Has your employer ever discussed mental health as part of an employee wellness program?', ['Yes', 'No','Dont know'])
        coworkers=st.selectbox('Would you be willing to discuss a mental health issue with your coworkers?', ['Yes', 'No','Some of them'])
        phys_health_consequence=st.selectbox('Do you think that discussing a physical health issue with your employer would have negative consequences?', ['Yes', 'No','Maybe'])
        supervisor=st.selectbox('Would you be willing to discuss a mental health issue with your supervisor?', ['Yes', 'No','Some of them'])
        benefits=st.selectbox('Does your employer provide mental health benefits?', ['Yes', 'No','Dont know'])
        mental_vs_physical=st.selectbox('Do you feel that your employer takes mental health as seriously as physical health?', ['Yes', 'No','Dont Know'])
     
        output=""

        input_dict = {'Age' : Age, 'Gender' : Gender,'remote_work':remote_work, 'tech_company':tech_company,
                      'family_history':family_history, 'obs_consequence':obs_consequence, 'mental_health_consequence':mental_health_consequence, 'mental_health_interview':mental_health_interview,
                      'no_employees':no_employees, 'leave':leave, 'self_employed':self_employed, 'seek_help':seek_help,
                      'anonymity':anonymity, 'phys_health_interview':phys_health_interview, 'work_interfere':work_interfere, 
                      'care_options':care_options, 'wellness_program':wellness_program, 'coworkers':coworkers, 'phys_health_consequence':phys_health_consequence, 
                      'supervisor':supervisor, 'benefits':benefits, 'mental_vs_physical':mental_vs_physical}
        input_df = pd.DataFrame([input_dict])

        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output = str(output)
        
        st.info('The output is {}'.format(output))
        
        if output=='1':
            st.warning('You need treatment')
        elif output=='0':
            st.success('You do not need any treatment and your mental health is good')
        else:
            st.error('Click On Predict')
        
if __name__ == '__main__':
    run() 
#streamlit run app.py   
    
