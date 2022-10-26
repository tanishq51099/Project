import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

df = pd.read_csv('healthcare.csv')

df.drop('id',axis=1,inplace=True) # Dropping Id as it is not of any use

st.title('EDA on Healthcare Stroke dataset', anchor=None)
# str_text = ':Attribute Information:\n        - gender\n        - age\n        - hypertension\n        - heart_disease\n        - ever_married\n        - work_type\n        - Residence_type\n        - avg_glucose_level\n        - bmi\n        - smoking status\n      - stroke\n '
# st.text(str_text)

st.markdown("""
This app performs EDA on heart stroke data.
* **Python libraries:** pandas, streamlit, numpy, matplotlib, seaborn
* **Data source:** [Stroke Prediction Data](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset).
""")
from sklearn.impute import KNNImputer
my_imp = KNNImputer(missing_values=np.NaN,n_neighbors=1000)
fixed_df = pd.DataFrame(my_imp.fit_transform(df.drop(columns=["avg_glucose_level","gender","hypertension","heart_disease","ever_married","work_type","Residence_type","smoking_status","stroke"])))
fixed_df["gender"] = df["gender"]
fixed_df["hypertension"] = df["hypertension"]
fixed_df["heart_disease"] = df["heart_disease"]
fixed_df["ever_married"] = df["ever_married"]
fixed_df["work_type"] = df["work_type"]
fixed_df["Residence_type"] = df["Residence_type"]
fixed_df["avg_glucose_level"] = df["avg_glucose_level"]
fixed_df["smoking_status"] = df["smoking_status"]
fixed_df["stroke"] = df["stroke"]

df=fixed_df.rename(columns={0: 'age', 1: 'bmi'})

st.markdown("""
    For the preprocessing of the dataset KNN imputer was used. There was some missingness in the Body Mass Index (bmi) column of the dataset and upon observing the relationship with other columns it was found that bmi had some correlation with the age of patient. So, the KNN imputer was used considering only the age of the patient.
    """)

st.text("")
st.text("")
check = st.checkbox('Show the dataset as table data')

if check:
    st.dataframe(df)

st.text("")
st.text("")

# st.text("Select any one to see the plots:")

radio = st.radio("Select any one to see the plots:",('Only Stroke Data','Only Non Stroke Data','Both'))
# button1 = st.button('Only Stroke Data')
# button2 = st.button('Only Non Stroke Data')
# button3 = st.button('Both')

test1= df.loc[(df["stroke"] == 1)]
test2= df.loc[(df["stroke"] == 0)]

col1,col2 = st.columns(2,gap='large')

with col1:
    st.header("Distribution plot")
    column_name =  st.selectbox(
    'For which column would you like to see the distribution plot of Strokes (as the below plot displays the distribution of just having strokes)?',
    ("age","avg_glucose_level","bmi")
    )

    plt.figure(figsize=(15, 15))

    # if button1:
    if radio == 'Only Stroke Data':
        fig1 = sns.displot(test1,x=column_name,hue="stroke",element="step")
        st.pyplot(fig1)

    if radio == 'Only Non Stroke Data':
        fig1 = sns.displot(test2,x=column_name,hue="stroke",element="step")
        st.pyplot(fig1)

    if radio == 'Both':
        fig1 = sns.displot(df,x=column_name,hue="stroke",element="step")
        st.pyplot(fig1)

with col2:
    st.header("Categorial plot")
    column_name_3 =  st.selectbox(
    'Choose the categorical column',
    ("gender","hypertension","heart_disease","ever_married","work_type","Residence_type","smoking_status")
    )
    column_name_4 = st.selectbox(
        'Choose the y axis',
        ("age","avg_glucose_level","bmi")
        )
    plt.figure(figsize=(15, 15))

    if radio == 'Only Stroke Data':
        fig2 = sns.catplot(data=test1,x=column_name_3,y=column_name_4,hue="stroke", kind= "box")
        st.pyplot(fig2)

    if radio == 'Only Non Stroke Data':
        fig2 = sns.catplot(data=test2,x=column_name_3,y=column_name_4,hue="stroke", kind="box")
        st.pyplot(fig2)

    if radio == 'Both':
        fig2 = sns.catplot(data=df,x=column_name_3,y=column_name_4,hue="stroke", kind="box")
        st.pyplot(fig2)

    # if button1:
    #     fig2 = sns.catplot(data=test1,x=column_name_3,y=column_name_4,hue="stroke", kind= "box")
    #     st.pyplot(fig2)

    # if button2:
    #     fig2 = sns.catplot(data=test2,x=column_name_3,y=column_name_4,hue="stroke", kind="box")
    #     st.pyplot(fig2)

    # if button3:
    #     fig2 = sns.catplot(data=df,x=column_name_3,y=column_name_4,hue="stroke", kind="box")
    #     st.pyplot(fig2)

st.text("")
st.text("")
st.text("")
st.text("")
st.text("")

col3,col4 = st.columns(2,gap='large')

with col3:
    st.header("Simple Pair plot")
    options = st.multiselect(
        'Features',("age","avg_glucose_level","bmi","hypertension","heart_disease"),default=['age','avg_glucose_level'],key=10
        )

    st.set_option('deprecation.showPyplotGlobalUse', False)

    if radio == 'Only Stroke Data':
        fig3 = sns.pairplot(
            pd.concat([test1[options],
               test1['stroke']],
              axis=1),hue='stroke')    

        st.pyplot(fig3)

    if radio == 'Only Non Stroke Data':
        fig3 = sns.pairplot(
            pd.concat([test2[options],
               test2['stroke']],
              axis=1),hue='stroke')    

        st.pyplot(fig3)

    if radio == 'Both':
        fig3 = sns.pairplot(
            pd.concat([df[options],
               df['stroke']],
              axis=1),hue='stroke')    

        st.pyplot(fig3)

with col4:
    st.header("KDE plot")
    column_name_2 =  st.selectbox(
        'For which column would you like to see the KDE plot',
        ("age","hypertension","heart_disease","avg_glucose_level","bmi")
        )

    plt.figure(figsize=(20, 15))
    # fig4 = sns.displot(test1, x=column_name_2, hue="stroke", kind="kde", fill= True)
    # if button1:
    #     fig4 = sns.displot(test1,x=column_name,hue="stroke",kind="kde", fill= True)
    #     st.pyplot(fig4)

    # if button2:
    #     fig4 = sns.displot(test2,x=column_name,hue="stroke",kind="kde", fill= True)
    #     st.pyplot(fig4)

    # if button3:
    #     fig4 = sns.displot(df,x=column_name,hue="stroke",kind="kde", fill= True)
    #     st.pyplot(fig4)

    if radio == 'Only Stroke Data':
        fig4 = sns.displot(test1,x=column_name_2,hue="stroke",kind="kde", fill= True)
        st.pyplot(fig4)

    if radio == 'Only Non Stroke Data':
        fig4 = sns.displot(test2,x=column_name_2,hue="stroke",kind="kde", fill= True)
        st.pyplot(fig4)

    if radio == 'Both':
        fig4 = sns.displot(df,x=column_name_2,hue="stroke",kind="kde", fill= True)
        st.pyplot(fig4)

st.text("")
st.text("")
st.text("")
st.text("")

if st.checkbox('Show Intercorrelation Heatmap'):
    st.header('Intercorrelation Matrix Heatmap')

    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(7, 5))
        ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
    st.pyplot()

if st.checkbox('Show Pie charts'):
    st.header('Pie charts')
    column_name_5 =  st.selectbox(
        'For which column would you like to see the pie plot',
        ("hypertension","smoking_status","gender","ever_married","work_type","Residence_type","heart_disease")
        )
    # st.markdown("For which data would you like to see the pie?")
    # st.text("")
    radio2 = st.radio("For which data would you like to see the pie?",('Stroke','Non Stroke'))
    # col1, col2 = st.columns([1,6])

    # with col1:
    #     button1 = st.button('Stroke')
    # with col2:
    #     button2 = st.button('Non Stroke')
    # button1 = st.button('Stroke')
    # button2 = st.button('Non Stroke')
    st.text("")
    st.text("")

    if radio2 == 'Stroke':
        test1.groupby(column_name_5).count().plot(kind='pie', y='stroke', autopct='%1.0f%%', legend='False')
        plt.legend(
           prop = {'size' : 5},
           loc = 'upper right', shadow = False)
        st.pyplot()
    if radio2 == 'Non Stroke':
        test2.groupby(column_name_5).count().plot(kind='pie', y='stroke', autopct='%1.0f%%')
        plt.legend(
           prop = {'size' : 5},
           loc = 'upper right', shadow = False)
        st.pyplot()

