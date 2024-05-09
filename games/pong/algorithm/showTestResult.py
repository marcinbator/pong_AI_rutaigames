import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv('output/results_algorithm.csv')

architecture_labels = data['architecture'].unique()
architecture_map = {label: i for i, label in enumerate(architecture_labels)}
data['architecture_encoded'] = data['architecture'].map(architecture_map)

# Zastosowanie skali logarytmicznej do 'lr'
data['log_lr'] = np.log(data['lr'])

for normalization in data['normalization'].unique():
    subset = data[data['normalization'] == normalization]

    # 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.scatter(subset['architecture_encoded'], subset['log_lr'], subset['accuracy'], c=subset['accuracy'], cmap="inferno")
    ax1.set_xticks(list(architecture_map.values()))
    ax1.set_xticklabels(list(architecture_map.keys()), rotation=45, ha='right')
    ax1.set_xlabel('Architecture')
    ax1.set_ylabel('Learning Rate')
    ax1.set_yticklabels([f'log({lr})' for lr in subset['lr'].unique()])
    ax1.set_zlabel('Accuracy')
    ax1.set_title(f'Normalization: {normalization} - Accuracy')
    plt.show()

    # Heatmap
    heatmap_data = subset.pivot_table(index='architecture_encoded', columns='log_lr', values='accuracy', aggfunc=np.mean)
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, cmap="inferno", xticklabels=[f'log({lr})' for lr in subset['lr'].unique()], yticklabels=list(architecture_map.keys()))
    plt.xlabel('Learning Rate')
    plt.ylabel('Architecture')
    plt.title(f'Normalization: {normalization} - Accuracy')
    plt.show()

    fig = plt.figure(figsize=(10, 8))
    ax2 = fig.add_subplot(111, projection='3d')
    ax2.scatter(subset['architecture_encoded'], subset['log_lr'], subset['loss'], c=subset['loss'], cmap="inferno")
    ax2.set_xticks(list(architecture_map.values()))
    ax2.set_xticklabels(list(architecture_map.keys()), rotation=45, ha='right')
    ax2.set_xlabel('Architecture')
    ax2.set_ylabel('Learning Rate')
    ax2.set_yticklabels([f'log({lr})' for lr in subset['lr'].unique()])
    ax2.set_zlabel('Loss')
    ax2.set_title(f'Normalization: {normalization} - Loss')
    plt.show()

    heatmap_data = subset.pivot_table(index='architecture_encoded', columns='log_lr', values='loss',
                                      aggfunc=np.mean)
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, cmap="inferno", xticklabels=[f'log({lr})' for lr in subset['lr'].unique()],
                yticklabels=list(architecture_map.keys()))
    plt.xlabel('Learning Rate')
    plt.ylabel('Architecture')
    plt.title(f'Normalization: {normalization} - Loss')
    plt.show()

    fig = plt.figure(figsize=(10, 8))
    ax3 = fig.add_subplot(111, projection='3d')
    ax3.scatter(subset['architecture_encoded'], subset['log_lr'], subset['classification_accuracy'], c=subset['classification_accuracy'], cmap="inferno")
    ax3.set_xticks(list(architecture_map.values()))
    ax3.set_xticklabels(list(architecture_map.keys()), rotation=45, ha='right')
    ax3.set_xlabel('Architecture')
    ax3.set_ylabel('Learning Rate')
    ax3.set_yticklabels([f'log({lr})' for lr in subset['lr'].unique()])
    ax3.set_zlabel('Classification accuracy')
    ax3.set_title(f'Normalization: {normalization} - Classification accuracy')
    plt.show()

    heatmap_data = subset.pivot_table(index='architecture_encoded', columns='log_lr', values='classification_accuracy',
                                      aggfunc=np.mean)
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, cmap="inferno", xticklabels=[f'log({lr})' for lr in subset['lr'].unique()],
                yticklabels=list(architecture_map.keys()))
    plt.xlabel('Learning Rate')
    plt.ylabel('Architecture')
    plt.title(f'Normalization: {normalization} - Classification accuracy')
    plt.show()