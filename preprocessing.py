import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

# Création de données aléatoires
np.random.seed(42)  # Pour obtenir des données aléatoires reproductibles
data = {
    'Colonne1': np.random.randint(1, 100, 200),
    'Colonne2': np.random.uniform(0.0, 1.0, 200),
    'target': np.random.choice(['A', 'B', 'C', 'D'], 200),
}

# Création du DataFrame
df = pd.DataFrame(data)

# Afficher les premières lignes du DataFrame
print(df.head())

class DataPreprocessor:
    def __init__(self, df):
        self.df = df
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def missing_values(self, strategy='median'):
        imputer = SimpleImputer(strategy=strategy)
        imputer = imputer.fit(self.df)
        self.df = imputer.transform(self.df)
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
        self.label_encoder()
        self.missing_values()
        self.outliers()
        self.standardisation()
        self.split_data()

# Créer une instance du préprocesseur de données
preprocessor = DataPreprocessor(df)

print(preprocessor.preprocess_data())