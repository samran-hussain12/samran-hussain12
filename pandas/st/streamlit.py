import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.title("Machine Learning Classification Web App")

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview:")
    st.write(df.head())
    
    # Select target column
    target_column = st.selectbox("Select Target Column", df.columns)
    
    # Preprocess Data
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Encode categorical target if necessary
    if y.dtype == 'O':
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    # Handle categorical features
    X = pd.get_dummies(X, drop_first=True)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train Model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Performance Metrics
    acc = accuracy_score(y_test, y_pred)
    st.write(f"### Model Accuracy: {acc:.2f}")
    st.write("### Classification Report:")
    st.text(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    st.write("### Confusion Matrix:")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)
    
    # Feature Importance
    feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    st.write("### Feature Importance:")
    st.bar_chart(feature_importance)
    
    # Prediction on new data
    st.write("### Predict on New Data")
    input_data = []
    for col in X.columns:
        input_data.append(st.number_input(f"{col}", value=float(0)))
    if st.button("Predict"):
        prediction = model.predict([input_data])
        st.write(f"Predicted Class: {le.inverse_transform(prediction) if 'le' in locals() else prediction}")