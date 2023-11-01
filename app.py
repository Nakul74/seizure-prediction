import joblib
import pandas as pd
import streamlit as st
import os
import aws_push
import matplotlib.pyplot as plt
import streamlit_authenticator as stauth
import yaml
import json
from datetime import datetime
from yaml.loader import SafeLoader
import warnings
warnings.filterwarnings("ignore")


with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
)

model = joblib.load('model.sav')
scaler = joblib.load('scaler.sav')

col = ['EEG.AF3', 'EEG.F7', 'EEG.F3', 'EEG.FC5', 'EEG.T7', 'EEG.P7',
       'EEG.O1', 'EEG.O2', 'EEG.P8', 'EEG.T8', 'EEG.FC6', 'EEG.F4',
       'EEG.F8', 'EEG.AF4']
c = ['Mean', 'stddev', 'skew', 'kurt', 'iqr', 'mean ad']
class_dict = {0:'normal',1:'seizure'}

def get_ecg_plot(df):
    num_channels = len(col)

    fig, axes = plt.subplots(num_channels, 1, figsize=(15, 3*num_channels))

    for i, channel in enumerate(col):
        axes[i].plot(df[channel].values.tolist())
        axes[i].legend([channel],fontsize=12)
        # axes[i].set_title(f'EEG : {channel} vs Timestamp', fontsize=20)
        axes[i].set_yticks([])  # Remove y-axis labels
        axes[i].grid()

    plt.tight_layout()
    return fig


def preprocess_data(file_path):
    df = pd.read_excel(file_path,skiprows=[0,1])
    fig = get_ecg_plot(df)
    try:
        df = df[~(df.Sample.isin(c))]
    except:
        df = df[~(df.S.isin(c))]

    df = df[col]
    val = []
    for i in col:
        val.append(df[i].mean())
        val.append(df[i].std())
        val.append(df[i].max())
        val.append(df[i].min())
        val.append(df[i].quantile(.25))
        val.append(df[i].quantile(.50))
        val.append(df[i].quantile(.75))
        val.append(df[i].quantile(.75) - df[i].quantile(.25))
        val.append(df[i].skew())
        val.append(df[i].kurtosis())

    l1 = ['mean','std','max','min','25%','median','75%','iqr','skew','kurtosis']
    l2 = col.copy()
    final_col = []
    for i in l2:
        for j in l1:
            final_col.append(i+'_'+j)

    df = pd.DataFrame(data = [val],columns = final_col)
    return df,fig


def predict_class(df):
    arr = scaler.transform(df)
    prob = model.predict_proba(arr)[0]
    if prob[0] > prob[1]:
        prob = prob[0]
    else:
        prob = prob[1]
    cls = model.predict(arr)[0]
    return round(prob,2),class_dict[cls]


def main():
    name, authentication_status, username = authenticator.login('Login', 'main')
    if st.session_state["authentication_status"]:
        authenticator.logout('Logout', 'main')
        st.write(f'Welcome *{st.session_state["name"]}*')
        patient_usernames = list(config['credentials']['usernames'].keys())
        patient_usernames.remove('dralex')
        
        if os.path.isfile('patients.json'):
            with open('patients.json', 'r') as fp:
                patients_json = json.load(fp)
        else:
            patients_json = {i:['No message','Pending'] for i in patient_usernames}
            
        if st.session_state['username'] == 'dralex':
            st.title("Epilepsy Prediction")
            tab1, tab2 = st.tabs(["Current status", "Uploaded data"])
            with tab1:
                patient = st.selectbox("select patient", patient_usernames, index=0)
                st.write("Upload an Excel file and click 'Predict' to get predictions.")
                uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"])
                comment = st.text_input("Add a comment for patient")
                if st.button("Predict"):
                    if uploaded_file is not None:
                        df = pd.read_excel(uploaded_file)
                        st.write("Uploaded Data:")
                        st.dataframe(df)
                        
                        with open(uploaded_file.name, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        df,fig = preprocess_data(uploaded_file.name)
                        prob,cls = predict_class(df)
                        # s3_file_name = str(patient)+f"-{cls}-{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}."+uploaded_file.name.split('.')[-1]
                        s3_file_name = str(patient)+f"-{cls}."+uploaded_file.name.split('.')[-1]
                        s3_client = aws_push.get_aws_client()
                        aws_push.upload_file_to_s3(s3_client,uploaded_file.name,s3_file_name)

                        os.remove(uploaded_file.name)
                        st.markdown(f"**Prediction class = {cls}**")
                        st.markdown(f"**Prediction Probability = {round(float(prob),2)}**")
                        st.success(f'File {s3_file_name} pushed to s3')
                        
                        st.write('Plot')
                        st.pyplot(fig)
                    
                    if comment:
                        patients_json[patient] = [comment,cls]
                        with open('patients.json', 'w') as fp:
                            json.dump(patients_json, fp)
            with tab2:
                s3_client = aws_push.get_aws_client()
                files_dict = aws_push.get_public_links(s3_client)
                if files_dict:
                    for filename,s3_link in files_dict.items():
                        st.markdown(f'<a href="{s3_link}" target="_blank">{filename}</a>', unsafe_allow_html=True)
                else:
                    st.markdown('**No files found**')
        else:
            st.markdown(f"**Message from doctor: {patients_json[st.session_state['username']][0]}**")
            if patients_json[st.session_state['username']][1] == 'seizure':
                st.error(f"Report Status: seizure detected")
            elif patients_json[st.session_state['username']][1] == 'normal':
                st.success(f"Report Status: seizure not detected")
            else:
                st.warning(f"Report Status: pending")
    elif st.session_state["authentication_status"] == False:
        st.error('Username/password is incorrect')
    elif st.session_state["authentication_status"] == None:
        st.warning('Please enter your username and password')
    
    
    
            
if __name__ == "__main__":
    main()
    
#rm -rf __public_logs__ && mkdir __public_logs__ && nohup streamlit run app.py --server.port 1024 >> __public_logs__/out 2>> __public_logs__/error &

