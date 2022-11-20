import streamlit as st 
import pandas as pd 
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns




def load_data():
    df = pd.read_csv("insurance_data.csv")
    df = df.fillna(df.mean())
    return df

df =load_data()

def show_explore_page():
    image = Image.open('bg.jpg')
    st.image(image, caption='Prudential Life Insurance AssessmentS') 

    # st.title("Explore Prudential Life Insurance Assessment Data")

    st.write("""
    ### Explore Prudential Life Insurance Assessment Data

    Product_Info : A set of normalized variables relating to the product applied for

    Ins_Age : Normalized age of applicant

    Ht : Normalized height of applicant

    Wt : Normalized weight of applicant

    BMI : Normalized BMI of applicant

    Employment_Info: A set of normalized variables relating to the employment history of the applicant.

    InsuredInfo: A set of normalized variables providing information about the applicant.

    Insurance_History : A set of normalized variables relating to the insurance history of the applicant.

    Family_Hist : A set of normalized variables relating to the family history of the applicant.

    Medical_History : A set of normalized variables relating to the medical history of the applicant.

    Medical_Keyword : A set of dummy variables relating to the presence of/absence of a medical keyword being associated with the application.

    Response : This is the target variable, an ordinal variable relating to the final decision associated with an application
    """)

    st.dataframe(df)

    data = df["Response"].value_counts()
    fig,ax = plt.subplots()
    ax.pie(data,labels=data.index,autopct="%1.1f%%",shadow=True,startangle=90)
    ax.axis("equal")
    
    st.write(""" #### The comparison of customers who are likely to get more risk over no risk""")
    st.pyplot(fig)

