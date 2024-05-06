import ast

import matplotlib.pyplot as plt
import numpy as np

# Wczytanie danych z pliku
data_min_max = []
data_min_max_nums = []
data_zero_one = []
data_zero_one_nums = []
i = 0
with open('results.txt', 'r') as file:
    for line in file:
        item = ast.literal_eval(line.strip())
        if "min_max" in item[0]:
            data_min_max.append(item)
            data_min_max_nums.append(i)
        elif "zero_one" in item[0]:
            data_zero_one.append(item)
            data_zero_one_nums.append(i)
        i += 1


# Przekształcenie danych wejściowych
def transform_data(data, nums):
    architectures = nums  # Oś X - kolejne architektury
    lr_values = [np.log10(float(item[0].split(',')[-2].strip())) for item in data]  # Oś Y - logarytm lr
    accuracy_values = [item[1] for item in data]  # Ostatnia wartość - accuracy
    loss_values = [item[2] for item in data]  # Ostatnia wartość - loss
    classification_accuracy_values = [item[3] for item in data]  # Ostatnia wartość - classification accuracy
    return architectures, lr_values, accuracy_values, loss_values, classification_accuracy_values


architectures_min_max, lr_values_min_max, accuracy_values_min_max, loss_values_min_max, classification_accuracy_values_min_max = transform_data(
    data_min_max, data_min_max_nums)
architectures_zero_one, lr_values_zero_one, accuracy_values_zero_one, loss_values_zero_one, classification_accuracy_values_zero_one = transform_data(
    data_zero_one, data_zero_one_nums)

# Utworzenie wykresów dla min_max normalization
fig1 = plt.figure(figsize=(8, 6))
ax1 = fig1.add_subplot(111, projection='3d')
scatter1 = ax1.scatter(architectures_min_max, lr_values_min_max, accuracy_values_min_max, c=accuracy_values_min_max,
                       cmap='inferno')
ax1.set_xlabel('Architecture')
ax1.set_ylabel('Log(lr)')
ax1.set_zlabel('Accuracy')
ax1.set_title('Accuracy - Min-Max Normalization')

for i in range(len(architectures_min_max)):
    ax1.text(architectures_min_max[i], lr_values_min_max[i], accuracy_values_min_max[i], f'{architectures_min_max[i]}',
             color='black')

plt.colorbar(scatter1, label='Accuracy')

fig2 = plt.figure(figsize=(8, 6))
ax2 = fig2.add_subplot(111, projection='3d')
scatter2 = ax2.scatter(architectures_min_max, lr_values_min_max, loss_values_min_max, c=loss_values_min_max,
                       cmap='inferno')
ax2.set_xlabel('Architecture')
ax2.set_ylabel('Log(lr)')
ax2.set_zlabel('Loss')
ax2.set_title('Loss - Min-Max Normalization')

for i in range(len(architectures_min_max)):
    ax2.text(architectures_min_max[i], lr_values_min_max[i], loss_values_min_max[i], f'{architectures_min_max[i]}',
             color='black')

plt.colorbar(scatter2, label='Loss')

fig3 = plt.figure(figsize=(8, 6))
ax3 = fig3.add_subplot(111, projection='3d')
scatter3 = ax3.scatter(architectures_min_max, lr_values_min_max, classification_accuracy_values_min_max,
                       c=classification_accuracy_values_min_max, cmap='inferno')
ax3.set_xlabel('Architecture')
ax3.set_ylabel('Log(lr)')
ax3.set_zlabel('Classification Accuracy')
ax3.set_title('Classification Accuracy - Min-Max Normalization')

for i in range(len(architectures_min_max)):
    ax3.text(architectures_min_max[i], lr_values_min_max[i], classification_accuracy_values_min_max[i],
             f'{architectures_min_max[i]}', color='black')

plt.colorbar(scatter3, label='Classification Accuracy')

# Utworzenie wykresów dla zero_one normalization
fig4 = plt.figure(figsize=(8, 6))
ax4 = fig4.add_subplot(111, projection='3d')
scatter4 = ax4.scatter(architectures_zero_one, lr_values_zero_one, accuracy_values_zero_one, c=accuracy_values_zero_one,
                       cmap='inferno')
ax4.set_xlabel('Architecture')
ax4.set_ylabel('Log(lr)')
ax4.set_zlabel('Accuracy')
ax4.set_title('Accuracy - Zero-One Normalization')

for i in range(len(architectures_zero_one)):
    ax4.text(architectures_zero_one[i], lr_values_zero_one[i], accuracy_values_zero_one[i],
             f'{architectures_zero_one[i]}', color='black')

plt.colorbar(scatter4, label='Accuracy')

fig5 = plt.figure(figsize=(8, 6))
ax5 = fig5.add_subplot(111, projection='3d')
scatter5 = ax5.scatter(architectures_zero_one, lr_values_zero_one, loss_values_zero_one, c=loss_values_zero_one,
                       cmap='inferno')
ax5.set_xlabel('Architecture')
ax5.set_ylabel('Log(lr)')
ax5.set_zlabel('Loss')
ax5.set_title('Loss - Zero-One Normalization')

for i in range(len(architectures_zero_one)):
    ax5.text(architectures_zero_one[i], lr_values_zero_one[i], loss_values_zero_one[i], f'{architectures_zero_one[i]}',
             color='black')

plt.colorbar(scatter5, label='Loss')

fig6 = plt.figure(figsize=(8, 6))
ax6 = fig6.add_subplot(111, projection='3d')
scatter6 = ax6.scatter(architectures_zero_one, lr_values_zero_one, classification_accuracy_values_zero_one,
                       c=classification_accuracy_values_zero_one, cmap='inferno')
ax6.set_xlabel('Architecture')
ax6.set_ylabel('Log(lr)')
ax6.set_zlabel('Classification Accuracy')
ax6.set_title('Classification Accuracy - Zero-One Normalization')

for i in range(len(architectures_zero_one)):
    ax6.text(architectures_zero_one[i], lr_values_zero_one[i], classification_accuracy_values_zero_one[i],
             f'{architectures_zero_one[i]}', color='black')

plt.colorbar(scatter6, label='Classification Accuracy')

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
