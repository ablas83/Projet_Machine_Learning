from models.decision_tree import dt_param_selector
from models.kneighbors_classifier import knn_param_selector
from models.linear_regression import lir_param_selector
from models.logistic_regression import lor_param_selector
from models.neural_network import nn_param_selector
from models.random_forest_classifier import rf_param_selector
from models.regression_ridge import rd_param_selector
from models.svc import svc_param_selector
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score
import streamlit as st
import preprocessing

def train_test(data, model, type_model):
    preprocessor = preprocessing.DataPreprocessor(data)
    X_train, X_test, y_train, y_test = preprocessor.preprocess_data()
    clf = model.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    if "SVC" in list(type_model):
        cm = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        f1_score = f1_score(y_test, y_pred)
    
    else : print(" pas encore ")

    return cm, accuracy, f1_score



