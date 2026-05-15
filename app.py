import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


st.set_page_config(page_title="Hanta Virüsü Risk Analizi", layout="centered", page_icon="🦠")
st.title("🦠 Hanta Virüsü Klinik Risk Tahmin Sistemi")
st.write("Bu sistem, hastaların klinik semptomlarını ve demografik verilerini analiz ederek yapay zeka tabanlı bir hayati risk tahmini sunar.")


@st.cache_resource
def train_model():
    np.random.seed(42)
    n_samples = 3000
    symptom_cols = ['Fever', 'Myalgia', 'Headache', 'Cough', 'Dyspnea', 'Nausea',
                    'Tachycardia', 'Hypotension', 'Pulmonary_edema', 'Thrombocytopenia',
                    'Back_pain', 'Abdominal_pain', 'Blurred_vision', 'Petechiae',
                    'Oliguria', 'Proteinuria', 'Hemorrhage', 'icu_admission', 'ventilator_used']
    
   
    data = {sym: np.random.choice([0, 1], size=n_samples, p=[0.65, 0.35]) for sym in symptom_cols}
    data['Age'] = np.random.randint(15, 85, size=n_samples)
    df = pd.DataFrame(data)
    
   
    risk_score = (df['Dyspnea'] * 2.5 + df['Pulmonary_edema'] * 3.0 + 
                  df['ventilator_used'] * 2.0 + (df['Age'] / 40))
    df['target'] = (risk_score > 4.5).astype(int)
    
    X = df[symptom_cols + ['Age']]
    y = df['target']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, symptom_cols


model, symptom_cols = train_model()


st.subheader("📋 Hasta Bilgileri ve Semptomlar")


age = st.slider("Hastanın Yaşı", 15, 85, 40)

st.write("Aşağıdaki klinik belirtilerden hastada mevcut olanları işaretleyiniz:")
user_inputs = {}


col1, col2 = st.columns(2)
for i, sym in enumerate(symptom_cols):
    label = sym.replace("_", " ").title() 
    if i % 2 == 0:
        with col1:
            user_inputs[sym] = 1 if st.checkbox(label, key=sym) else 0
    else:
        with col2:
            user_inputs[sym] = 1 if st.checkbox(label, key=sym) else 0


if st.button("🔴 Risk Analizini Başlat", type="primary"):
   
    input_data = [user_inputs[sym] for sym in symptom_cols] + [age]
    
  
    prediction = model.predict([input_data])[0]
    probabilities = model.predict_proba([input_data])[0]
    risk_percentage = probabilities[1] * 100
    
    st.markdown("---")
    st.subheader("📊 Analiz Sonucu")
    
    if prediction == 1:
        st.error(f"⚠️ **Kritik Risk Durumu Tespiti!** \nHastanın klinik tablosu yüksek hayati risk taşımaktadır. Yoğun bakım ve acil müdahale gerekebilir. \n\n**Hesaplanan Ölüm Riski Oranı: %{risk_percentage:.2f}**")
    else:
        st.success(f"✅ **Stabil Durum (Düşük Risk).** \nHastanın mevcut semptom kombinasyonları kontrol edilebilir seviyededir. Standart tedavi ve rutin klinik takip önerilir. \n\n**Hesaplanan Ölüm Riski Oranı: %{risk_percentage:.2f}**")
