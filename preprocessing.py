import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import Utils.bdd

df = Utils.bdd.get_data_to_df()


class DataPreprocessor:
    def __init__(self, df):
        self.df = df
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def missing_values(self, strategy='median', threshold=0.2):
        for column in self.df.columns:
            # Calcul du pourcentage de valeurs manquantes dans la colonne
            missing_percentage = self.df[column].isnull().mean()
            # Vérifiez si le pourcentage de valeurs manquantes dépasse le seuil
            if missing_percentage > threshold:
                # Supprimez la colonne si le seuil est dépassé
                self.df.drop(columns=[column], inplace=True)
            else:
                # Si le seuil n'est pas dépassé, vérifiez le type de données de la colonne
                if self.df[column].dtype == 'float':
                    # Imputez les valeurs manquantes avec la médiane pour les données numériques
                    imputer = SimpleImputer(strategy=strategy)
                    self.df[column] = imputer.fit_transform(self.df[[column]])
                elif self.df[column].dtype == 'int':
                    # Imputez les valeurs manquantes avec la médiane pour les données numériques
                    imputer = SimpleImputer(strategy=strategy)
                    self.df[column] = imputer.fit_transform(self.df[[column]])
                elif self.df[column].dtype == 'object':
                    # Imputez les valeurs manquantes avec une stratégie appropriée pour les données catégorielles (par exemple, classe majoritaire)
                    self.df[column].fillna(self.df[column].mode().iloc[0], inplace=True)
    
        return self.df

    def label_encoder(self):
        labelencoder = LabelEncoder()
        for column in self.df.columns:
            self.df[column] = labelencoder.fit_transform(self.df[column])
        return self.df

    def outliers(self, contamination=0.05):
        outlier_detector = IsolationForest(contamination=contamination)
        outlier_labels = outlier_detector.fit_predict(self.df)
        self.df = self.df[outlier_labels == 1]
        # Conservez les données originales
        self.original_df = self.df.copy()
        return self.df

    def standardisation(self):
        scaler = StandardScaler()
        self.df = scaler.fit_transform(self.df)
        return self.df
         
    def split_data(self):
        self.X = df.drop('target', axis = 1)
        self.y = df['target']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.3,random_state = 42)
        return (self.X_train, self.X_test, self.y_train, self.y_test)
    
    def preprocess_data(self):
        self.missing_values()
        self.label_encoder()
        self.outliers()
        self.standardisation()
        self.split_data()

# Créer une instance du préprocesseur de données
preprocessor = DataPreprocessor(df)
preprocessor.preprocess_data()
X_train, X_test, y_train, y_test = (
    preprocessor.X_train, preprocessor.X_test, preprocessor.y_train, preprocessor.y_test
)
print(y_test)