import matplotlib.pyplot as plt
import numpy as np

# Exemple de données pour chaque entier (remplacez par vos propres données)
data5 = np.random.randn(50)
data6 = np.random.randn(60)
data7 = np.random.randn(40)
data8 = np.random.randn(50)
data9 = np.random.randn(50)
data10 = np.random.randn(50)

# Créer une liste de données et une liste de labels
data = [data5, data6, data7, data8, data9, data10]
labels = [5, 6, 7, 8, 9, 10]

# Créer les boxplots
plt.figure(figsize=(10, 6))
plt.boxplot(data, tick_labels=labels)
plt.title('Boxplots pour chaque entier de 5 à 10')
plt.xlabel('Entier')
plt.ylabel('Valeurs')
plt.grid(True)
plt.show()
