# Importer les bibliothèques nécessaires
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# Fixer le random seed pour obtenir des résultats reproductibles
np.random.seed(42)

# Génération des données simulées
n = 100
surface = np.random.randint(50, 200, n)             # Surface en m2
nb_pieces = np.random.randint(1, 6, n)              # Nombre de pièces
age = np.random.randint(0, 50, n)                   # Âge de la maison en années
distance_centre = np.random.uniform(0, 20, n)       # Distance au centre en km

# Calcul du prix avec la formule donnée (en milliers d'€)
prix = 30 + (surface * 0.8) + (nb_pieces * 15) - (age * 0.5) - (distance_centre * 2) + np.random.normal(0, 10, n)

# Création du DataFrame
data = pd.DataFrame({
    'Surface': surface,
    'Nb_pieces': nb_pieces,
    'Age': age,
    'Distance_centre': distance_centre,
    'Prix': prix
})

# Afficher les premières lignes des données
print(data.head())

# Créer la variable cible binaire en fonction de la médiane des prix
data['Prix_categorie'] = (data['Prix'] >= data['Prix'].median()).astype(int)

# Séparer les données en variables explicatives et cible
X = data[['Surface', 'Nb_pieces', 'Age', 'Distance_centre']]
y = data['Prix_categorie']

# Diviser les données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardiser les données
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entraîner le modèle de régression logistique
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Prédire sur l'ensemble de test
y_pred = log_reg.predict(X_test)

# Évaluer le modèle : précision, rappel, et matrice de confusion
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Précision du modèle :", accuracy)
print("Rappel du modèle :", recall)
print("Matrice de confusion :\n", conf_matrix)

# Afficher la matrice de confusion
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Moins cher', 'Plus cher'])
disp.plot(cmap="Blues")
plt.title("Matrice de confusion - Régression logistique")
plt.show()

# Visualiser les distributions des prédictions
plt.figure(figsize=(10, 6))
sns.histplot(data[data['Prix_categorie'] == 1]['Prix'], color="blue", label="Plus cher", kde=True)
sns.histplot(data[data['Prix_categorie'] == 0]['Prix'], color="red", label="Moins cher", kde=True)
plt.xlabel("Prix (en milliers d'€)")
plt.ylabel("Fréquence")
plt.title("Distribution des prix des maisons par catégorie")
plt.legend()
plt.show()

# Prédiction d'une nouvelle maison
nouvelle_maison = np.array([[120, 3, 10, 5]])  # Exemple : Surface de 120m2, 3 pièces, 10 ans, 5 km du centre
nouvelle_maison = scaler.transform(nouvelle_maison)  # Standardiser les caractéristiques
prediction = log_reg.predict(nouvelle_maison)

print("La nouvelle maison est prédite comme :", "Plus cher" if prediction[0] == 1 else "Moins cher")
