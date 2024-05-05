import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ast

# Wczytanie danych z pliku
data = []
with open('results.txt', 'r') as file:
    for line in file:
        data.append(ast.literal_eval(line.strip()))

# Przekształcenie danych wejściowych
architectures = np.arange(1, len(data) + 1)  # Oś X - kolejne architektury
lr_values = [np.log10(float(item[0].split(',')[1].strip())) for item in data]  # Oś Y - logarytm lr
accuracy_values = [item[1] for item in data]  # Ostatnia wartość - accuracy
loss_values = [item[2] for item in data]  # Ostatnia wartość - loss
classification_accuracy_values = [item[3] for item in data]  # Ostatnia wartość - classification accuracy

# Utworzenie oddzielnych wykresów dla accuracy, loss i classification accuracy
fig1 = plt.figure(figsize=(8, 6))
ax1 = fig1.add_subplot(111, projection='3d')
ax1.scatter(architectures, lr_values, accuracy_values)
ax1.set_xlabel('Architectures')
ax1.set_ylabel('Log(lr)')
ax1.set_zlabel('Accuracy')
ax1.set_title('Accuracy')

for i in range(len(architectures)):
    ax1.text(architectures[i], lr_values[i], accuracy_values[i], str(architectures[i]))

fig2 = plt.figure(figsize=(8, 6))
ax2 = fig2.add_subplot(111, projection='3d')
ax2.scatter(architectures, lr_values, loss_values)
ax2.set_xlabel('Architectures')
ax2.set_ylabel('Log(lr)')
ax2.set_zlabel('Loss')
ax2.set_title('Loss')

for i in range(len(architectures)):
    ax2.text(architectures[i], lr_values[i], loss_values[i], str(architectures[i]))

fig3 = plt.figure(figsize=(8, 6))
ax3 = fig3.add_subplot(111, projection='3d')
ax3.scatter(architectures, lr_values, classification_accuracy_values)
ax3.set_xlabel('Architectures')
ax3.set_ylabel('Log(lr)')
ax3.set_zlabel('Classification Accuracy')
ax3.set_title('Classification Accuracy')

for i in range(len(architectures)):
    ax3.text(architectures[i], lr_values[i], classification_accuracy_values[i], str(architectures[i]))

plt.show()

# Wykres 2D dla classification accuracy
plt.figure(figsize=(8, 6))
plt.plot(architectures, classification_accuracy_values, marker='o', linestyle='-')
plt.title('Poprawność klasyfikacji dla każdej z architektur')
plt.xlabel('Architectures')
plt.ylabel('Classification Accuracy')
plt.grid(True)
plt.xticks(architectures)
plt.tight_layout()
plt.show()


##########

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Wczytanie danych z pliku CSV
df_normalized = pd.read_csv("prepared_pong_pong.csv", header=None)
df_normalized.columns = ["x", "y", "vel_x", "vel_y", "posY", "move"]

# Normalizacja danych
for column in df_normalized.columns[:-1]:
    min_val = df_normalized[column].min()
    max_val = df_normalized[column].max()
    df_normalized[column] = (2 * (df_normalized[column] - min_val) / (max_val - min_val)) - 1


pd.set_option('display.max_rows', None)
basic_stats_normalized = df_normalized.describe()
print("Podstawowe statystyki po znormalizowaniu:")
print(basic_stats_normalized.to_string())

# Wykresy pudełkowe dla każdej kolumny po znormalizowaniu
df_normalized.boxplot()
plt.title("Wykresy pudełkowe dla każdej kolumny po znormalizowaniu")
plt.show()

# Wykresy liczby wystąpień wartości dla każdej kolumny po znormalizowaniu
for column in df_normalized.columns:
    plt.figure(figsize=(8, 6))
    df_normalized[column].hist(bins=20, alpha=0.7)
    plt.title(f"Rozkład wartości dla kolumny {column} po znormalizowaniu")
    plt.xlabel("Znormalizowane wartości")
    plt.ylabel("Liczba wystąpień")
    plt.show()
