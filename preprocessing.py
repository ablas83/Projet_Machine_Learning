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
    'Target': np.random.choice(['A', 'B', 'C', 'D'], 200),
}

# Création du DataFrame
df = pd.DataFrame(data)

# Afficher les premières lignes du DataFrame
print(df.head())



X = df.iloc[:, :-1].values #prend toutes les colonnes à part la dernière
y = df.iloc[:,-1].values #prend les valeurs de la dernière colonne
print(X)
print(y)

# POO
class DataPreprocessor:
    def __init__(self, df, X, y):
        self.df = df
        self.X = X
        self.y = y
    
    def missing_values(self, strategy='mean'):
        imputer = SimpleImputer(strategy=strategy)
        imputer = imputer.fit(self.X)
        self.X = imputer.transform(self.X)

    def label_encoder_x(self):
        labelencoder_X = LabelEncoder()
        self.X[:, 0] = labelencoder_X.fit_transform(self.X[:, 0])

    def label_encoder_y(self):
        labelencoder_y = LabelEncoder()
        self.y = labelencoder_y.fit_transform(self.y)
    
    def one_hot_encoder(self):
        onehotencoder = OneHotEncoder(categorical_features=[0])
        self.X = onehotencoder.fit_transform(self.X).toarray()

    def outliers(self, contamination=0.05):
        outlier_detector = IsolationForest(contamination=contamination)
        outlier_labels = outlier_detector.fit_predict(self.X)
        self.df = self.df[outlier_labels == 1]

    def standardisation(self):
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)
         
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,random_state = 42)


# Créer une instance du préprocesseur de données
preprocessor = DataPreprocessor(df, X, y)
print(preprocessor)


# programmation non objet

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42) # split

essai = [X_train, X_test, y_train, y_test]
for i in essai:
    print(i)