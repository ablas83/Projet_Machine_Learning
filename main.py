import streamlit as st
from Utils.bdd import connection_database, get_data_to_df

def intro():
    return st.set_page_config(
        page_title="ML Playground",
        layout="centered",
        initial_sidebar_state="auto"
    )
df = None
def header():
    st.title('Bienvenue')
    

def sidebar():
    
    sellection=st.sidebar.selectbox('please select your dataset', options= ['Diabet inde','Vin', 'load csv file'], key ='data_sb', on_change=changeData)

def changeData():
    uploaded_file = None
    if st.session_state['data_sb'] == 'load csv file':
        uploaded_file = st.sidebar.file_uploader("Choose a file", type=['csv'], key='uploaded_file',on_change= load)
    else:
        db = connection_database()
        df = get_data_to_df(st.session_state['data_sb'],db,None)
        dataset = st.table(df.head())
def load():
    if st.session_state['uploaded_file'] is not None:
        db = connection_database()
        df = get_data_to_df(st.session_state['data_sb'],db,st.session_state['uploaded_file'])
        dataset = st.table(df.head())
    else:
        dataset = st.caption('none')

if __name__ == '__main__':
    intro()
    header()
    sidebar()