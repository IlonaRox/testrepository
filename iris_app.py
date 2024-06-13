import streamlit as st
import streamlit as st
import pandas as pd
import xlsxwriter
import io
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import joblib

from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer,OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

st.title('Eda - iris dataset')

imm = Image.open('iris.png')
st.image(imm, caption='Specie iris', width=500)

uploaded_file = st.file_uploader("Choose a file",type={"xlsx", "csv"})
if uploaded_file is not None:
    ###### transformation #####################################
    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file, index_col=0)
        st.dataframe(df)
    elif uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)


#    if st.button('Preprocessing', help="add column + palloncini"):
#       #st.header('Addes Column')
#        #df['new_col'] = 1
#        #st.dataframe(df)
#        #st.balloons()

buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
    # Write each dataframe to a different worksheet.
    df.to_excel(writer, index=False)
    # Close the Pandas Excel writer and output the Excel file to the buffer
    writer.close()
    st.download_button(
        label="Download Excel Result",
        data=buffer,
        file_name="trasnformed_file.xlsx",
        mime="application/vnd.ms-excel")
    

valori = df['class'].value_counts()
st.markdown("**count per classe**")
st.write(valori)

# PER ALCUNI PLOT SERVE plt.figure, PER ALTRI NON SERVE
fig0 = sns.pairplot(df, hue='class', height=2, aspect=1);
st.pyplot(fig0)

options = df.iloc[:, :-1].columns.to_list()
z = st.selectbox('Seleziona una feature per violin plot', options)
fig1 = plt.figure(figsize=(18,10))
sns.violinplot(data=df, x="class", y=z, palette='Set2')
st.pyplot(fig1)
# xlabel e ylabel size .....



loaded_model = joblib.load("logistic_reg_iris.pkl")

sepal_lenght = st.number_input('Inserisci un valore per sepal length 4,3 - 7,9', 0.5, 10.0, 5.0, 0.1)
sepal_width = st.number_input('Inserisci un valore per sepal width 2,0 - 4,4', 0.5, 6.0, 2.0, 0.1)
petal_length = st.number_input('Inserisci un valore per petal length 1,0 - 6,9', 0.5, 8.0, 5.0, 0.1)
petal_width = st.number_input('Inserisci un valore per petal width 0,1 - 2,5', 0.1, 3.0, 1.5, 0.1)



#data = {
#        "sepal length": sepal_lenght,
#        "sepal width": sepal_width,
#       "petal length": petal_length,
#        "petal width":petal_width
#        }

#input_df = pd.DataFrame(data)
#res = loaded_model.predict(input_df).astype(int)[0]

res = loaded_model.predict([[sepal_lenght,sepal_width,petal_length,petal_width]])[0]

def specie_iris(res):
    if res == 1:
        res = 'Setosa'
    elif res==2:
        res = 'Versicolor'
    else:
        res = 'Virginica'
    return res

if st.button('Predict', help="Indovina la specie"):
    st.write("La specie dell'iris è ", specie_iris(res))
    st.balloons()





