import numpy as np
import streamlit as st
import joblib 

# Simpan model
joblib.dump(diabetes_model, 'random_forest_model.joblib')

# Muat model
diabetes_model = joblib.load('random_forest_model.joblib')

# judul web
st.title('Prediksi Penyakit Diabetes')

# membagi kolom
col1, col2 = st.columns(2)

with col1:
    Pregnancies = st.text_input('Input nilai Pregnancies', value='0')

with col2:
    Glucose = st.text_input('Input nilai Glucose', value='0')

with col1:
    BloodPressure = st.text_input('Input nilai Blood Pressure', value='0')

with col2:
    SkinThickness = st.text_input('Input nilai Skin Thickness', value='0')

with col1:
    Insulin = st.text_input('Input nilai Insulin', value='0')

with col2:
    BMI = st.text_input('Input nilai BMI', value='0')

with col1:
    DiabetesPedigreeFunction = st.text_input('Input nilai Diabetes Pedigree Function', value='0')

with col2:
    Age = st.text_input('Input nilai Age', value='0')

# code untuk prediksi
diab_diagnosis = ''

# membuat tombol untuk prediksi
if st.button('Test Prediksi Diabetes'):
    try:
        # konversi input menjadi numerik
        input_data = [float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness),
                      float(Insulin), float(BMI), float(DiabetesPedigreeFunction), float(Age)]
        
        # ubah menjadi array numpy
        input_array = np.array(input_data).reshape(1, -1)

        # prediksi
        diab_prediction = diabetes_model.predict(input_array)

        # Menampilkan hasil prediksi
        if diab_prediction[0] == 1:
            st.success('Pasien terkena Diabetes')
        else:
            st.success('Pasien tidak terkena Diabetes')
    except ValueError:
        st.error('Masukkan nilai numerik untuk semua fitur.')
