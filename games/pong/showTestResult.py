import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ast

# Wczytanie danych z pliku
data = []
with open('results.txt', 'r') as file:
    for line in file:
        data.append(ast.literal_eval(line.strip()))

# Rozpakowanie danych
methods = [item[0:1] for item in data]
values = [item[1:] for item in data]
record_ids = [i for i in range(len(data))]  # Dodane ID rekordu

# Konwersja na numpy array
methods = np.array(methods)
values = np.array(values)
record_ids = np.array(record_ids)  # Dodane ID rekordu

# Utworzenie figury i subplotu 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Kolorowanie punktów według wartości
norm = plt.Normalize(values.min(), values.max())
colors = plt.cm.viridis(norm(values))

# Wygenerowanie wykresu punktowego
scatter = ax.scatter(values[:, 0], values[:, 1], values[:, 2], s=100)

# Dodanie etykiet z ID rekordu
for i in range(len(data)):
    ax.text(values[i, 0], values[i, 1], values[i, 2], str(record_ids[i]), color='black')

# Ustawienie etykiet dla osi
ax.set_xlabel('accuracy')
ax.set_ylabel('loss')
ax.set_zlabel('classification_accuracy')

# Dodanie kolorowej skali
sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Wartość')

# Wyświetlenie wykresu
plt.title('Wizualizacja danych 3D')
plt.show()
