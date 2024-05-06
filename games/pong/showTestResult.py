import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Wczytanie danych z pliku CSV
data = pd.read_csv('results.csv')

# Mapowanie unikalnych etykiet architektury na liczby całkowite
architecture_labels = data['architecture'].unique()
architecture_map = {label: i for i, label in enumerate(architecture_labels)}
data['architecture_encoded'] = data['architecture'].map(architecture_map)

# Tworzenie wykresów dla każdej wartości normalizacji
for normalization in data['normalization'].unique():
    subset = data[data['normalization'] == normalization]

    # Wykres dla accuracy
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.scatter(subset['architecture_encoded'], subset['lr'], subset['accuracy'], c=subset['accuracy'], cmap="inferno")
    ax1.set_xticks(list(architecture_map.values()))
    ax1.set_xticklabels(list(architecture_map.keys()), rotation=45, ha='right')
    ax1.set_xlabel('Architecture')
    ax1.set_ylabel('Learning Rate')
    ax1.set_zlabel('Accuracy')
    ax1.set_title(f'Normalization: {normalization} - Accuracy')
    plt.show()

    # Wykres dla loss
    fig = plt.figure(figsize=(10, 8))
    ax2 = fig.add_subplot(111, projection='3d')
    ax2.scatter(subset['architecture_encoded'], subset['lr'], subset['loss'], c=subset['loss'], cmap="inferno")
    ax2.set_xticks(list(architecture_map.values()))
    ax2.set_xticklabels(list(architecture_map.keys()), rotation=45, ha='right')
    ax2.set_xlabel('Architecture')
    ax2.set_ylabel('Learning Rate')
    ax2.set_zlabel('Loss')
    ax2.set_title(f'Normalization: {normalization} - Loss')
    plt.show()

    # Wykres dla classification_accuracy
    fig = plt.figure(figsize=(10, 8))
    ax3 = fig.add_subplot(111, projection='3d')
    ax3.scatter(subset['architecture_encoded'], subset['lr'], subset['classification_accuracy'], c=subset['classification_accuracy'], cmap="inferno")
    ax3.set_xticks(list(architecture_map.values()))
    ax3.set_xticklabels(list(architecture_map.keys()), rotation=45, ha='right')
    ax3.set_xlabel('Architecture')
    ax3.set_ylabel('Learning Rate')
    ax3.set_zlabel('Classification accuracy')
    ax3.set_title(f'Normalization: {normalization} - Classification accuracy')
    plt.show()


