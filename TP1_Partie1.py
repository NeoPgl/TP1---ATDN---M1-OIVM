#=======================================================

#PARTIE 1 


# Importer les bibliothèques nécessaires

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import scipy.stats as stats
from sklearn.preprocessing import PolynomialFeatures

# Fixer la graine aléatoire pour la reproductibilité
np.random.seed(0)

# Générer les données aléatoires
X = 2 * np.random.rand(100, 1)  # 100 échantillons pour X entre 0 et 2
bruit = np.random.randn(100, 1)  # Bruit gaussien
y = 7 + 4 * X + bruit  # Variable dépendante y

# Calcul du nombre d'échantillons
m = X.shape[0]  # Nombre total d'échantillons (100 dans ce cas)


# Ajuster le modèle de régression linéaire
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)  # Prédictions du modèle

# Calcul des résidus
residus = y - y_pred



# Coefficient de détermination R² pour la régression linéaire
r_squared_linear = 1 - (np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2))
print("Coefficient de détermination R² pour la régression linéaire :", r_squared_linear)

# MSE pour la régression linéaire
mse_linear = np.sum((y_pred - y)**2) / m
print("Erreur quadratique moyenne (MSE) pour la régression linéaire :", mse_linear)

# RMSE pour la régression linéaire
rmse_linear = np.sqrt(mse_linear)
print("Erreur quadratique moyenne (RMSE) pour la régression linéaire :", rmse_linear)


# Visualisation des données et du modèle de régression linéaire
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color="blue", alpha=0.5, label="Données avec bruit gaussien")
plt.plot(np.sort(X, axis=0).flatten(), y_pred[np.argsort(X, axis=0).flatten()], color="red", label="Régression linéaire") # Ajout de flatten pour corriger les erreurs --> Transforme X en tableau 1D
plt.xlabel("X")
plt.ylabel("y")
plt.title("Régression linéaire")
plt.legend()
plt.show()

# Visualisation de la distribution des résidus (histogramme)
plt.figure(figsize=(10, 6))
plt.hist(residus, bins=20, color="purple", edgecolor="black", alpha=0.7)
plt.xlabel("Résidus")
plt.ylabel("Fréquence")
plt.title("Distribution des résidus")
plt.show()

# Q-Q plot pour évaluer la normalité des résidus
plt.figure(figsize=(10, 6))
stats.probplot(residus.flatten(), dist="norm", plot=plt)
plt.title("Q-Q Plot des résidus")
plt.show()








# Transformation polynomiale de degré 2
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

# Régression polynomiale sur les caractéristiques polynomiales
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
y_poly_pred = poly_model.predict(X_poly)


# Calcul des résidus pour régression polynomiale
residus_polynomiale = y - y_poly_pred


# Coefficient de détermination R² pour la régression polynomiale
r_squared = 1 - (np.sum((y - y_poly_pred)**2) / np.sum((y - np.mean(y))**2))
# Afficher les résultats dans le terminal
print("Coefficient de détermination R² pour la régression polynomiale :", r_squared)




# MSE pour Mean Square Error (erreur quadratique moyenne)
mse = np.sum((y_poly_pred - y)**2) / m
print("Erreur quadratique moyenne (MSE) :", mse)


# RMSE pour Root Mean Square Error (racine carrée de l'erreur quadratique moyenne)
rmse = np.sqrt(mse)
print("Erreur quadratique moyenne (RMSE) :", rmse)



# Visualisation des données et du modèle polynomiale 
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color="blue", alpha=0.5, label="Données avec bruit")
plt.plot(np.sort(X, axis=0).flatten(), y_poly_pred[np.argsort(X, axis=0).flatten()], color="red", label="Régression polynomiale (degré 2)") # Ajout de flatten pour corriger les erreurs --> Transforme X en tableau 1D
plt.xlabel("X_poly")
plt.ylabel("y")
plt.title("Régression polynomiale de degré 2")
plt.legend()
plt.show()


# Visualisation de la distribution des résidus (histogramme) pour la régression polynomiale
plt.figure(figsize=(10, 6))
plt.hist(residus_polynomiale, bins=20, color="purple", edgecolor="black", alpha=0.7)  # Utiliser residus_polynomiale ici
plt.xlabel("Résidus")
plt.ylabel("Fréquence")
plt.title("Distribution des résidus (Régression polynomiale)")
plt.show()

# Q-Q plot pour évaluer la normalité des résidus de la régression polynomiale
plt.figure(figsize=(10, 6))
stats.probplot(residus_polynomiale.flatten(), dist="norm", plot=plt)  # Utiliser residus_polynomiale ici
plt.title("Q-Q Plot des résidus (Régression polynomiale)")
plt.show()

