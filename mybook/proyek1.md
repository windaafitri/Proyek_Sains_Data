---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Laporan Proyek 1

## Pengembangan Model Prediksi Harga Chainlink (LINK-USD) Berbasis Machine Learning untuk Analisis Fluktuasi Pasar Kripto

## Pendahuluan

## Latar Belakang
Cryptocurrency adalah aset digital yang menggunakan blockchain untuk mencatat transaksi secara transparan dan aman. Chainlink (LINK) adalah platform oracle terdesentralisasi yang menghubungkan smart contract dengan data dunia nyata, menjadikannya penting dalam berbagai proyek berbasis blockchain.

Namun, harga Chainlink (LINK-USD) sangat fluktuatif, dipengaruhi oleh sentimen pasar, regulasi, dan kondisi ekonomi global, sehingga menyulitkan investor dalam mengambil keputusan. Teknologi seperti machine learning dapat digunakan untuk memprediksi harga berdasarkan data historis, membantu mengurangi risiko dan mendukung keputusan investasi yang lebih baik.

## Tujuan Proyek
Proyek ini bertujuan untuk membangun model prediksi harga cryptocurrency Chainlink (LINK-USD) dengan memanfaatkan data historis. Analisis ini diharapkan mampu:

- Mendukung investor dalam mengambil keputusan investasi yang lebih berbasis data.
- Menyediakan wawasan mengenai potensi fluktuasi harga Chainlink guna meningkatkan peluang keuntungan dan meminimalkan risiko secara lebih efektif.

## Rumusan Masalah
- Bagaimana membangun model yang mampu memprediksi harga Chainlink (LINK-USD) secara akurat menggunakan data historis?
- Bagaimana prediksi harga Chainlink dapat dimanfaatkan untuk meningkatkan kualitas keputusan investasi di pasar cryptocurrency?

## Metodologi

## Data Understanding
a. Sumber Data Data yang saya gunakan pada proyek ini diperoleh dari website https://finance.yahoo.com/quote/LINK-USD/history/?period1=1607212800&period2=1733443200, platform online yang menyediakan data keuangan pasar secara real-time.

Dalam proyek ini, data historis yang digunakan mencakup rentang waktu dari 06 Desember 2022 hingga 06 Desember 2024, yang diperoleh dalam format file CSV.

## Project

With MyST Markdown, you can define code cells with a directive like so:

```{code-cell}
pip install seaborn
```

```{code-cell}
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import seaborn as sns
import matplotlib.pyplot as plt

# Membaca data
df = pd.read_csv('https://raw.githubusercontent.com/windaafitri/dataset/refs/heads/main/chainlink.csv')

# mengubah kolom 'Date' dalam format datetime
df['Date'] = pd.to_datetime(df['Date'])

# Mengatur kolom 'Date' sebagai indeks
df.set_index('Date', inplace=True)

# Mensortir data berdasarkan kolom Date dari terkecil ke terbesar
df = df.sort_values(by='Date')
print(df)
```

b. Deskripsi Dataset

```{code-cell}
df.info()
```

```{code-cell}
df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].describe()
```

```{code-cell}
# Mencari Missing Value
df.isnull().sum()
```

## Data Understanding
```{code-cell}
import matplotlib.pyplot as plt
import seaborn as sns
for col in df:
    plt.figure(figsize=(7, 3))
    sns.lineplot(data=df, x='Date', y=col)
    plt.title(f'Trend of {col}')
    plt.xlabel('Date')
    plt.ylabel(col)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.show()
```

korelasi antar fitur
```{code-cell}
correlation_matrix = df.corr()

plt.figure(figsize=(7, 3))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Heatmap Korelasi Antar Fitur')
plt.show()
```

```{code-cell}
df = df.drop(columns=['Volume', 'Adj Close'])
df.head()
```

```{code-cell}
df['Close Target'] = df['Close'].shift(-1)

df = df[:-1]
df.head()
```

```{code-cell}
# Inisialisasi scaler untuk fitur (input) dan target (output)
scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()

# Normalisasi fitur (Open, High, Low,, 'Close' Close Target-4, Close Target-5)
df_features_normalized = pd.DataFrame(scaler_features.fit_transform(df[['Open', 'High', 'Low', 'Close']]),
                                      columns=['Open', 'High', 'Low', 'Close'],
                                      index=df.index)

# Normalisasi target (Close Target)
df_target_normalized = pd.DataFrame(scaler_target.fit_transform(df[['Close Target']]),
                                    columns=['Close Target'],
                                    index=df.index)

# Gabungkan kembali dataframe yang sudah dinormalisasi
df_normalized = pd.concat([df_target_normalized, df_features_normalized], axis=1)
df_normalized.head()
```

```{code-cell}
# Mengatur fitur (X) dan target (y)
X = df_normalized[['Open', 'High', 'Low', 'Close']]
y = df_normalized['Close Target']

# Membagi data menjadi training dan testing (60% training, 40% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=False)
```

```{code-cell}
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np
import matplotlib.pyplot as plt

# List model untuk ensemble Bagging
models = {
    "Neural Network (JST)": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=32),
    "SVR": SVR(),
    "KNN": KNeighborsRegressor(n_neighbors=5)
}

# Dictionary untuk menyimpan hasil evaluasi
results = {}

# Iterasi setiap model
for i, (name, base_model) in enumerate(models.items()):
    # Inisialisasi Bagging Regressor
    bagging_model = BaggingRegressor(
        estimator=base_model,
        n_estimators=10,
        max_samples=0.7,
        random_state=32,
        bootstrap=True
    )

    # Latih model
    bagging_model.fit(X_train, y_train)

    # Prediksi pada data uji
    y_pred = bagging_model.predict(X_test)

    # Evaluasi
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100  # Dalam persen

    # Simpan hasil evaluasi
    results[name] = {"RMSE": rmse, "MAPE": mape}

    # Kembalikan hasil prediksi ke skala asli
    y_pred_original = scaler_target.inverse_transform(y_pred.reshape(-1, 1))
    y_test_original = scaler_target.inverse_transform(y_test.values.reshape(-1, 1))

    # Plot hasil prediksi
    plt.figure(figsize=(15, 6))
    plt.plot(y_test.index, y_test_original, label="Actual", color="blue")
    plt.plot(y_test.index, y_pred_original, label=f"Predicted ({name})", color="red")

    # Tambahkan detail plot
    plt.title(f'Actual vs Predicted Values ({name})')
    plt.xlabel('Tanggal')
    plt.ylabel('Harga')
    plt.legend()
    plt.grid(True)

    # Tampilkan plot
    plt.show()

# Tampilkan hasil evaluasi
for model, metrics in results.items():
    print(f"{model}:\n  RMSE: {metrics['RMSE']:.2f}\n  MAPE: {metrics['MAPE']:.2f}%\n")
```

