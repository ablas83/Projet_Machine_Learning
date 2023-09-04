from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
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
        f1score = f1_score(y_test, y_pred)
        return cm, accuracy, f1score
    else :
        mse = mean_squared_error(y_test,y_pred)
        r2score = r2_score(y_test,y_pred) 
        return mse, r2score

    



