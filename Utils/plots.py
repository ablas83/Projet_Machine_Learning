import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA
import scipy.stats as stats
import seaborn as sns


def digramme_dispersion(y_train, y_test):

    plt.scatter(y_train, y_test, color='blue', marker='o', label='Données réelles vs. Prédites')

    # Ajoutez une ligne de référence (y=x) pour montrer une prédiction parfaite
    plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='--', lw=2, label='Prédiction parfaite')

    # Ajoutez des étiquettes et une légende
    plt.xlabel('Valeurs réelles')
    plt.ylabel('Valeurs prédites')
    plt.legend(loc='best')

    # Affichez le diagramme de dispersion
    plt.title('Diagramme de dispersion entre les valeurs réelles et prédites')
    plt.grid(True)
    plt.show()


def courbe_regression(X, y, y_test):

    # Créez un graphique avec les points de données réelles
    plt.scatter(X, y, color='blue', label='Données réelles')

    # Tracez la ligne de régression (ligne de prédiction) en utilisant les valeurs prédites
    plt.plot(X, y_test, color='red', linestyle='-', linewidth=2, label='Ligne de régression')

    plt.xlabel('Caractéristiques')
    plt.ylabel('Valeurs réelles / prédites')
    plt.legend(loc='best')

    plt.title('Courbe de régression entre les données réelles et prédites')
    plt.grid(True)
    plt.show()


def histo_residu(y_train, y_test):

    # Calculez les résidus en soustrayant les valeurs réelles des valeurs prédites
    residus = y_train - y_test

    plt.hist(residus, bins=10, color='blue', alpha=0.7, edgecolor='black')

    plt.xlabel('Résidus')
    plt.ylabel('Fréquence')
    plt.title('Histogramme des résidus')
    plt.grid(True)
    plt.show()

    # Créez une fonction pour tracer la courbe d'apprentissage
def courbe_appr(model, X, y):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, train_sizes=np.linspace(0.1, 1.0, 50))

    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.title('Courbe d\'apprentissage')
    plt.xlabel('Taille de l\'ensemble d\'entraînement')
    plt.ylabel('Erreur quadratique moyenne')

    plt.plot(train_sizes, train_scores_mean, label='Score d\'entraînement', color='blue', marker='o')
    plt.plot(train_sizes, test_scores_mean, label='Score de validation', color='red', marker='o')

    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

def quant_quant(y_train, y_test):
    
    # Supposons que "residus" contienne les résidus de votre modèle.
    residus = y_train - y_test

    # Calculez les quantiles des résidus
    sorted_residus = np.sort(residus)
    normal_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residus)))

    # Créez un graphique QQ
    plt.figure(figsize=(6, 6))
    plt.scatter(normal_quantiles, sorted_residus, color='blue')
    plt.plot([min(normal_quantiles), max(normal_quantiles)], [min(normal_quantiles), max(normal_quantiles)], color='red', linestyle='--')
    plt.xlabel('Quantiles théoriques (distribution normale)')
    plt.ylabel('Quantiles observés (résidus)')
    plt.title('Graphique QQ')
    plt.grid(True)
    plt.show()


def conf_matrix(y_train, y_test):

    # Calculez la matrice de confusion
    conf_matrix = confusion_matrix(y_train, y_test)

    # Créez un graphique de la matrice de confusion
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, square=True)
    plt.xlabel('Valeurs Prédites')
    plt.ylabel('Valeurs Réelles')
    plt.title('Matrice de Confusion')
    plt.show()


def roc(y_train, y_prob):

    # Calculez la courbe ROC
    fpr, tpr, _ = roc_curve(y_train, y_prob)
    roc_auc = auc(fpr, tpr)

    # Tracez la courbe ROC
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Courbe ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de Faux Positifs')
    plt.ylabel('Taux de Vrais Positifs')
    plt.title('Courbe ROC')
    plt.legend(loc='lower right')
    plt.show()


def disp_classes(X, y_test):

    # Réduisez les dimensions à 2D en utilisant l'analyse en composantes principales (PCA)
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # Créez un diagramme de dispersion des classes
    plt.figure(figsize=(8, 6))
    colors = ['blue', 'red']
    for i in range(2):
        plt.scatter(X_2d[y_test == i, 0], X_2d[y_test == i, 1], color=colors[i], label=f'Classe {i}')
    plt.xlabel('Composante Principale 1')
    plt.ylabel('Composante Principale 2')
    plt.legend()
    plt.title('Diagramme de Dispersion des Classes (2D)')
    plt.show()
