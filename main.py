import streamlit as st


def intro():
    return st.set_page_config(
        page_title="ML Playground",
        layout="centered",
        initial_sidebar_state="auto"
    )
dataset = None
def header():
    st.title('Bienvenue')
    

def sidebar():
    
    sellection=st.sidebar.selectbox('please select your dataset', options= ['Diabet inde','Vin', 'load csv file'], key ='data_sb', on_change=changeData)

def changeData():
    uploaded_file = None
    if st.session_state['data_sb'] == 'load csv file':
        uploaded_file = st.sidebar.file_uploader("Choose a file")
    if uploaded_file != None:
        st.caption('pass')
    dataset = st.caption(st.session_state['data_sb'])

if __name__ == '__main__':
    intro()
    header()
    sidebar()