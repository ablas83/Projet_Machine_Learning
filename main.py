import streamlit as st
from Utils.bdd import connection_database, get_data_to_df
from helpers.selection import getAlgorims
from Utils.train_test import train_test
import preprocessing
import Utils.plots as pt
import pandas as pd
st.set_option('deprecation.showPyplotGlobalUse', False)

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
    with st.expander("DataFrame"):
        if st.session_state['uploaded_file'] is not None:
            df = getData('load file',st.session_state['uploaded_file'])
            table = st.table(df.head())
        else :
            df = getData(st.session_state['data_sb'],None)
            table = st.table(df.head())
    if df is not None:
        algorithms = getAlgorims(df)
        st.sidebar.selectbox('please select your algorithm', options= algorithms.keys(), key ='algo', on_change=changeAlgo)
        st.sidebar.toggle('Modifier les Hyperparamètres', key='on')
        st.sidebar.toggle('Hyperparamètres optimals', key='optimal')
        model = algorithms[st.session_state['algo']]()
        preprocessor = preprocessing.DataPreprocessor(df)
        X_train, X_test, y_train, y_test,X,y = preprocessor.preprocess_data()
        if "SVC" in list(algorithms):
            clf = model.fit(X_train, y_train)
            if st.session_state['optimal']:
                best_model = clf.best_estimator_
                y_pred = best_model.predict(X_test)
                best_params_df = pd.DataFrame(clf.best_params_, index=['Valeur'])
                cm, acc,f1 = train_test(y_pred, y_test,algorithms)
                pt.courbe_appr(best_model,X, y)
                st.caption(acc)
                st.dataframe(best_params_df)
            else:
                y_pred = clf.predict(X_test)
                cm, acc,f1 = train_test(y_pred, y_test,algorithms)
                st.pyplot(pt.conf_matrix(y_test,y_pred))
                st.pyplot(pt.courbe_appr(model, X, y))
                st.pyplot(pt.roc_class(X_train, X_test, y_train, y_test))
                st.caption(acc)
        else:
            clf = model.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            mse, r2 = train_test(y_pred, y_test,algorithms)
            st.pyplot(pt.courbe_appr(model, X, y))
            st.pyplot(pt.quant_quant(y_test, y_pred))
            st.plotly_chart(pt.histo_residu(y_test, y_pred))
            st.pyplot(pt.digramme_dispersion(y_test, y_pred))
            st.caption(mse)
            