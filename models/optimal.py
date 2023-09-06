from sklearn.model_selection import GridSearchCV
import streamlit as st
from sklearn.tree import DecisionTreeClassifier


def dt_optimalparams(X_train, y_train):
    params= {"criterion":["gini", "entropy"] ,"max_depth": list(range(1,51,10)),"min_samples_split": list(range(1,21,5)),"max_features": [None, "auto", "sqrt", "log2"]}
    m = GridSearchCV(DecisionTreeClassifier(random_state=42),params,cv=5)
    clf = m.fit(X_train, y_train)
    optimal = clf.best_params_
    st.session_state['criterion'] = ["gini", "entropy"].index(optimal['criterion'])
    #st.session_state['max_depth'] = optimal['max_depth']
    #st.session_state['min_samples_split'] = optimal['min_samples_split']
    #st.session_state['max_features'] = optimal['max_features']
