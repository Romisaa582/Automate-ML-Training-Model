import streamlit as st
import pandas as pd
import numpy as np
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# عنوان التطبيق
st.title("🧠 Alzheimer's Disease Prediction")

# تحميل البيانات
@st.cache_data
def load_data():
    df = pd.read_csv("alzheimers_disease_data.csv")  # تأكد من وضع الملف في نفس المجلد
    return df

df = load_data()

# عرض البيانات
st.subheader("📊 Dataset Overview")
st.write(df.head())

# عرض توزيع التشخيصات
st.subheader("🩺 Diagnosis Distribution")
st.write(df["Diagnosis"].value_counts())

# تحويل القيم النصية إلى أرقام
le = LabelEncoder()
df["Diagnosis"] = le.fit_transform(df["Diagnosis"])

# تقسيم البيانات إلى ميزات وأهداف
X = df.drop(columns=["Diagnosis"])
y = df["Diagnosis"]

# تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# زر تدريب النموذج
if st.button("🚀 Train Model"):
    with st.spinner("🔍 Training in Progress..."):
        clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
        models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    
    st.success("✅ Model Training Completed!")

    # عرض أفضل نموذج بناءً على الدقة
    st.subheader("🏆 Best Model Performance")
    best_model = models.iloc[0]
    st.write(f"**Best Model:** {best_model.name}")
    st.write(f"**Accuracy:** {best_model.Accuracy * 100:.2f}%")
    
    # عرض جميع النماذج
    st.subheader("📌 All Model Performance")
    st.dataframe(models)

# معلومات إضافية
st.markdown("🔹 This application uses AutoML to find the best predictive model for Alzheimer's disease.")


