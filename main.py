import streamlit as st
from Utils.bdd import connection_database, get_data_to_df
from helpers.selection import getAlgorims
from Utils.train_test import train_test
import preprocessing
import Utils.plots as pt
df = None
table =  None
def intro():
    return st.set_page_config(
        page_title="ML Playground",
        layout="centered",
        initial_sidebar_state="auto"
    )

def header():
    st.title('Bienvenue')
    table = st.table(getData(st.session_state['data_sb'],None).head())

def sidebar():
    st.sidebar.selectbox('please select your dataset', options= ['Diabet inde','Vin'], key ='data_sb', on_change=changeData)
    st.sidebar.file_uploader("Choose a file", type=['csv'], key='uploaded_file',on_change= load)
    
    

def changeAlgo():
    pass
def changeData():
    pass
        
def load():
   pass

def getData(type, path):
    db = connection_database()
    return get_data_to_df(type,db,path)

if __name__ == '__main__':
    intro()
    sidebar()
    st.title('Bienvenue')
    if st.session_state['uploaded_file'] is not None:
        df = getData('load file',st.session_state['uploaded_file'])
        table = st.table(df.head())
    else :
        df = getData(st.session_state['data_sb'],None)
        table = st.table(df.head())
    if df is not None:
        algorithms = getAlgorims(df)
        st.sidebar.selectbox('please select your algorithm', options= algorithms.keys(), key ='algo', on_change=changeAlgo)
        model = algorithms[st.session_state['algo']]()
        preprocessor = preprocessing.DataPreprocessor(df)
        X_train, X_test, y_train, y_test,X,y = preprocessor.preprocess_data()
        if "SVC" in list(algorithms):
            clf = model.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            cm, acc,f1 = train_test(X_train, y_train,X_test, y_test,model,algorithms)
            st.pyplot(pt.conf_matrix(y_pred,y_test))
            st.caption(acc)
        else:
            clf = model.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            mse, r2 = train_test(X_train, y_train, X_test, y_test,model,algorithms)
            st.pyplot(pt.courbe_appr(model, X, y))
            st.caption(mse)