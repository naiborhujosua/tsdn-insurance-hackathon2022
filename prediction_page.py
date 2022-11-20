import streamlit as st
import joblib 
import numpy as np
from PIL import Image
import pandas as pd 
import shap
from pathlib import Path
import streamlit.components.v1 as components

st.set_page_config(initial_sidebar_state='expanded')


def show_predict_page():
    image = Image.open('bg.jpg')
    st.image(image, caption='Prudential Life Insurance AssessmentS')
    st.title("Insurance Risk rate Prediction")

    df = pd.read_csv("insurance_data.csv")
    df = df.fillna(df.mean())
    feature_names = [col for col in df.columns if col not in ["Response"]]
  
    X = df[feature_names]
    y = df['Response']

    metadata ={}

    for num_col in feature_names:
        metadata[f"{num_col}_min"] = df[num_col].min()
        metadata[f"{num_col}_max"] = df[num_col].max()

    class_names = np.unique(df['Response']).tolist()

    user_input = []

    step_map = {
    'BMI': 0.1,
    'Medical_History_23': 1.0,
    'Medical_History_4': 1.0,
    'Product_Info_4': 0.1,

    'Wt': 0.1,
    'Ins_Age': 0.1,
    'Medical_Keyword_15': 0.1,
    'Medical_Keyword_3': 0.1,

    'InsuredInfo_6': 1.0,
    'Family_Hist_3': 1.0,
    'Family_Hist_4': 0.1,
    'Medical_History_30': 1.0
    }
    for col in feature_names:
        num_value = st.sidebar.number_input(
            f"Select {col}",
            step=step_map[col])
        user_input.append(num_value)

    user_df = pd.DataFrame([user_input], columns=feature_names)

    clf = joblib.load("insurance_clf.joblib")
    prediction = clf.predict(user_df)[0]
    class_prediction = class_names[prediction]

    # st.sidebar.write(f"## Prediction: {class_prediction}")
    proba = clf.predict_proba(user_df)
    proba_df = pd.DataFrame(proba, columns=class_names)

    st.write(f"## Explanation for Predicting: **{class_prediction}**")
    

    #user_encoded = clf[:-1].transform(user_df)
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)

    def st_shap(plot, height=None):
        shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
        components.html(shap_html, height=height)

    # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
    st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:]))

    st.markdown("""the bold value is the models score for this observation. The higher the scores lead to the model to predict 1(Risk) and lower scores lead to the model to predict 0(No risk)""")


    
    

   

