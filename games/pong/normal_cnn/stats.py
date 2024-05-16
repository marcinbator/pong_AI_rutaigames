import matplotlib.pyplot as plt
import pandas as pd

# Wczytanie danych z pliku CSV
df_normalized = pd.read_csv("output/prepared_pong_pong_normal.csv", header=None)
df_normalized.columns = ["x", "y", "vel_x", "vel_y", "posY", "move"]


# Normalizacja danych
def normalize_data(df, normalization_type):
    for column in df.columns:
        if normalization_type == "min_max":
            min_val = df[column].min()
            max_val = df[column].max()
            df[column] = (2 * (df[column] - min_val) / (max_val - min_val)) - 1
        elif normalization_type == "zero_one":
            min_val = df[column].min()
            max_val = df[column].max()
            df[column] = (df[column] - min_val) / (max_val - min_val)
    return df


# Kopie danych do normalizacji dla różnych typów normalizacji
df_min_max = df_normalized.copy()
df_zero_one = df_normalized.copy()

# Normalizacja danych
df_min_max = normalize_data(df_min_max, "min_max")
df_zero_one = normalize_data(df_zero_one, "zero_one")

# Wykresy pudełkowe dla każdej kolumny po znormalizowaniu
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
df_min_max.boxplot()
plt.title("Wykresy pudełkowe po znormalizowaniu (min_max)")

plt.subplot(2, 2, 2)
df_zero_one.boxplot()
plt.title("Wykresy pudełkowe po znormalizowaniu (zero_one)")

plt.tight_layout()
plt.show()

# Wykresy liczby wystąpień wartości dla każdej kolumny po znormalizowaniu
for column in df_min_max.columns:
    plt.figure(figsize=(8, 6))
    df_min_max[column].hist(bins=20, alpha=0.7, color='blue')
    plt.title(f"Rozkład wartości dla kolumny {column} po znormalizowaniu (min_max)")
    plt.xlabel("Znormalizowane wartości")
    plt.ylabel("Liczba wystąpień")
    plt.show()

for column in df_zero_one.columns:
    plt.figure(figsize=(8, 6))
    df_zero_one[column].hist(bins=20, alpha=0.7, color='green')
    plt.title(f"Rozkład wartości dla kolumny {column} po znormalizowaniu (zero_one)")
    plt.xlabel("Znormalizowane wartości")
    plt.ylabel("Liczba wystąpień")
    plt.show()
