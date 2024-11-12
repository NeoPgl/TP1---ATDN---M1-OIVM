#=======================================================
# PARTIE 2 



# Importer les bibliothèques nécessaires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
import seaborn as sns




# Importer les données
train_df = pd.read_csv('house-prices-advanced-regression-techniques/train.csv')
test_df = pd.read_csv('house-prices-advanced-regression-techniques/test.csv')
sample_submission_df = pd.read_csv('house-prices-advanced-regression-techniques/sample_submission.csv')

# Afficher les premières lignes des données
print(train_df.head())

# Lire le fichier de description des données (data_description.txt)
with open('house-prices-advanced-regression-techniques/data_description.txt', 'r') as f:
    description = f.read()

print(description)

# Visualiser la relation entre les variables explicatives et la variable cible
X = train_df[['GrLivArea']].values  # Surface habitable
y = train_df['SalePrice'].values  # Prix de vente

# Visualisation de la relation
plt.scatter(X, y, alpha=0.5)
plt.xlabel('Surface habitable (GrLivArea)')
plt.ylabel('Prix de vente (SalePrice)')
plt.title('Relation entre Surface habitable et Prix de vente')
plt.show()

# Régression linéaire
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_lin = lin_reg.predict(X)

# Calcul et affichage des métriques de la régression linéaire
r_squared_lin = r2_score(y, y_pred_lin)
rmse_lin = np.sqrt(mean_squared_error(y, y_pred_lin))
print("Régression linéaire - R²:", r_squared_lin)
print("Régression linéaire - RMSE:", rmse_lin)

# Visualisation des prédictions de la régression linéaire
plt.scatter(X, y, alpha=0.5, label='Données réelles')
plt.plot(X, y_pred_lin, color='red', label='Régression linéaire', linewidth=2)
plt.xlabel('Surface habitable (GrLivArea)')
plt.ylabel('Prix de vente (SalePrice)')
plt.title('Régression linéaire entre Surface habitable et Prix de vente')
plt.legend()
plt.show()

# Calcul des résidus et visualisation de la régression linéaire
residus_lin = y - y_pred_lin
plt.hist(residus_lin, bins=20, color="purple", edgecolor="black", alpha=0.7)
plt.xlabel('Résidus')
plt.ylabel('Fréquence')
plt.title('Distribution des résidus de la régression linéaire')
plt.show()

# Q-Q plot des résidus linéaires
stats.probplot(residus_lin, dist="norm", plot=plt)
plt.title('Q-Q Plot des résidus (Régression linéaire)')
plt.show()



# Régression polynomiale de degré 2
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_pred_poly = poly_reg.predict(X_poly)

# Calcul et affichage des métriques pour la régression polynomiale
r_squared_poly = r2_score(y, y_pred_poly)
rmse_poly = np.sqrt(mean_squared_error(y, y_pred_poly))
print("Régression polynomiale (degré 2) - R²:", r_squared_poly)
print("Régression polynomiale (degré 2) - RMSE:", rmse_poly)

# Visualisation des prédictions de la régression polynomiale
plt.scatter(X, y, alpha=0.5, label='Données réelles')
plt.plot(np.sort(X, axis=0), y_pred_poly[np.argsort(X, axis=0)], color='green', label='Régression polynomiale (degré 2)', linewidth=2)
plt.xlabel('Surface habitable (GrLivArea)')
plt.ylabel('Prix de vente (SalePrice)')
plt.title('Régression polynomiale de degré 2')
plt.legend()
plt.show()



# Régression polynomiale avec régularisation Ridge
ridge_reg = Ridge(alpha=1.0)  # Paramètre de régularisation alpha
ridge_reg.fit(X_poly, y)
y_pred_ridge = ridge_reg.predict(X_poly)

# Calcul et affichage des métriques pour la régression polynomiale avec régularisation Ridge
r_squared_ridge = r2_score(y, y_pred_ridge)
rmse_ridge = np.sqrt(mean_squared_error(y, y_pred_ridge))
print("Régression polynomiale (degré 2) avec Ridge - R²:", r_squared_ridge)
print("Régression polynomiale (degré 2) avec Ridge - RMSE:", rmse_ridge)

# Affichage des coefficients pour comparer les modèles
print("Coefficients de la régression polynomiale sans régularisation:", poly_reg.coef_)
print("Coefficients de la régression polynomiale avec régularisation Ridge:", ridge_reg.coef_)

# Visualisation des prédictions de la régression polynomiale avec Ridge
plt.scatter(X, y, alpha=0.5, label='Données réelles')
plt.plot(np.sort(X, axis=0), y_pred_ridge[np.argsort(X, axis=0)], color='blue', label='Régression polynomiale avec Ridge', linewidth=2)
plt.xlabel('Surface habitable (GrLivArea)')
plt.ylabel('Prix de vente (SalePrice)')
plt.title('Régression polynomiale avec régularisation Ridge (degré 2)')
plt.legend()
plt.show()


# Fonction pour calculer le RMSE (Root Mean Square Error)
rmse_scorer = make_scorer(mean_squared_error, squared=False)


# Paramètres de la validation croisée
k = 5  # Nombre de plis pour la validation croisée
kf = KFold(n_splits=k, shuffle=True, random_state=42)


# Liste pour stocker les scores de chaque pli
lin_reg_rmse_scores = []
poly_reg_rmse_scores = []


# Validation croisée pour la régression linéaire
lin_reg = LinearRegression()
lin_rmse_scores = cross_val_score(lin_reg, X, y, cv=kf, scoring=rmse_scorer)
lin_reg_rmse_scores.extend(lin_rmse_scores)

# Afficher les scores et la moyenne pour la régression linéaire
print("Scores de la régression linéaire (RMSE) pour chaque pli:", lin_rmse_scores)
print("Score moyen de la régression linéaire (RMSE):", np.mean(lin_rmse_scores))

# Validation croisée pour la régression polynomiale de degré 2
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)
poly_reg = LinearRegression()
poly_rmse_scores = cross_val_score(poly_reg, X_poly, y, cv=kf, scoring=rmse_scorer)
poly_reg_rmse_scores.extend(poly_rmse_scores)

# Afficher les scores et la moyenne pour la régression polynomiale
print("Scores de la régression polynomiale (RMSE) pour chaque pli:", poly_rmse_scores)
print("Score moyen de la régression polynomiale (RMSE):", np.mean(poly_rmse_scores))

# Tracer la dispersion des scores pour chaque modèle
plt.figure(figsize=(10, 6))
plt.boxplot([lin_reg_rmse_scores, poly_reg_rmse_scores], labels=['Régression linéaire', 'Régression polynomiale (degré 2)'])
plt.ylabel('RMSE')
plt.title("Dispersion des scores RMSE pour la régression linéaire et polynomiale")
plt.show()



# Définir les événements
# Événement 1 : Nombre de chambres > 3
event_bedrooms = train_df['BedroomAbvGr'] > 3  # Nombre de chambres au-dessus du rez-de-chaussée

# Événement 2 : Localisation dans la région "OldTown"
event_neighborhood = train_df['Neighborhood'] == 'OldTown'

# Calculer la probabilité conjointe avec un tableau croisé
joint_prob_table = pd.crosstab(event_bedrooms, event_neighborhood, normalize=True)

# Afficher le tableau des probabilités conjointes
print("Probabilité conjointe que la maison ait plus de 3 chambres et soit dans 'OldTown':")
print(joint_prob_table)


plt.figure(figsize=(8, 6))
sns.heatmap(joint_prob_table, annot=True, cmap="Blues", fmt=".2f")
plt.xlabel("Région (OldTown)")
plt.ylabel("Plus de 3 chambres")
plt.title("Probabilité conjointe : Plus de 3 chambres et localisation dans OldTown")
plt.show()


# Définition des probabilités données
P_A = 0.30  # Probabilité qu'une voiture soit économique
P_B_given_A = 0.85  # Probabilité d'une faible consommation si la voiture est économique
P_B_given_not_A = 0.20  # Probabilité d'une faible consommation si la voiture n'est pas économique
P_not_A = 1 - P_A  # Probabilité qu'une voiture ne soit pas économique

# Calcul de P(B) - la probabilité d'une faible consommation
P_B = (P_B_given_A * P_A) + (P_B_given_not_A * P_not_A)

# Calcul de la probabilité a posteriori P(A|B)
P_A_given_B = (P_B_given_A * P_A) / P_B

# Affichage des résultats
print("La probabilité qu'une voiture soit économique sachant qu'elle a une faible consommation (P(A|B)) est de : {:.2f}".format(P_A_given_B))
