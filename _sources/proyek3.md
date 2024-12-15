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

# Laporan Proyek 3

## Prediksi Saham Rigetti Computing (RGTI): Data Historis untuk Strategi Investasi Inovatif

## Pendahuluan

## Latar Belakang
Quantum Computing adalah bidang yang merevolusi dunia teknologi dengan prinsip mekanika kuantum untuk pemrosesan data yang lebih cepat dan efisien. Rigetti Computing (RGTI) adalah salah satu pemain utama dalam industri ini, mengembangkan perangkat keras dan perangkat lunak berbasis teknologi hybrid quantum-classical. Sebagai perusahaan inovatif di sektor teknologi tinggi, fluktuasi harga saham Rigetti sering mencerminkan kemajuan teknologi, dinamika pasar, serta faktor global lainnya.  

Fluktuasi harga saham Rigetti yang signifikan sering menjadi tantangan bagi investor dalam membuat keputusan yang tepat. Untuk menghadapi tantangan ini, model prediksi harga berbasis data historis dan kecerdasan buatan dapat membantu memberikan wawasan yang lebih mendalam. Model ini dirancang untuk memprediksi pergerakan harga saham Rigetti di masa depan, membantu investor merencanakan strategi yang lebih efektif, mengurangi risiko, dan memaksimalkan peluang keuntungan.

Pengembangan model prediksi saham Rigetti menjadi penting untuk mendukung strategi investasi yang lebih terinformasi dan berbasis data. Dengan memanfaatkan data historis, proyek ini bertujuan membantu investor memahami dinamika pasar Rigetti secara lebih mendalam dan memberikan solusi praktis bagi tantangan di pasar saham teknologi tinggi, khususnya dalam sektor komputasi kuantum.

## Tujuan Proyek
Penelitian ini bertujuan untuk mengeksplorasi pemanfaatan data historis dalam memprediksi pergerakan harga saham Rigetti Computing (RGTI) guna memberikan wawasan yang lebih mendalam kepada investor dalam merancang strategi investasi yang lebih optimal. Selain itu, penelitian ini juga bertujuan untuk mengembangkan model prediksi berbasis data historis yang andal, sehingga dapat membantu investor dalam mengelola risiko serta membuat keputusan investasi yang lebih terarah dan strategis. Dengan demikian, penelitian ini diharapkan mampu memberikan solusi praktis untuk mendukung pengambilan keputusan yang lebih baik di pasar saham teknologi tinggi, khususnya di sektor komputasi kuantum.

## Rumusan Masalah
Bagaimana data historis dapat dimanfaatkan untuk memprediksi pergerakan harga saham Rigetti Computing (RGTI) dan memberikan wawasan yang lebih mendalam kepada investor dalam merancang strategi investasi yang lebih optimal? Selain itu, bagaimana analisis data historis dapat menghasilkan model prediktif yang akurat untuk membantu investor mengatasi volatilitas harga saham serta membuat keputusan investasi yang lebih berbasis data dan terinformasi?

## Metodologi

## Data Understanding
a. Sumber Data 
Data yang digunakan dalam proyek ini berasal dari platform Yahoo Finance, yang menyediakan informasi historis terkait harga berbagai aset keuangan, termasuk saham teknologi tinggi. Yahoo Finance merupakan sumber data terpercaya yang sering digunakan oleh investor dan analis untuk mengakses data harga, volume perdagangan, dan indikator pasar lainnya. Dalam proyek ini, digunakan data historis harga saham Rigetti Computing (RGTI) dari tahun 2022 hingga 2024 dengan frekuensi harian. Data ini mencakup informasi seperti harga pembukaan (open), harga tertinggi (high), harga terendah (low), harga penutupan (close), serta volume perdagangan. Data ini dimanfaatkan untuk mendukung analisis dan pengembangan model prediksi harga saham Rigetti yang akurat dan berbasis data.

```{code-cell}
# import library
import pandas as pd
```

With MyST Markdown, you can define code cells with a directive like so:

```{code-cell}
# Membaca data CSV
# Membaca data
df = pd.read_csv('https://raw.githubusercontent.com/windaafitri/dataset/refs/heads/main/Rigetti.csv')

# Mengubah kolom 'Tanggal' menjadi format datetime
df['Date'] = pd.to_datetime(df['Date'])

# Mengatur kolom 'Date' sebagai indeks
df.set_index('Date', inplace=True)

# Mensortir data berdasarkan tanggal
df = df.sort_values(by='Date')
print(df.head())
```

b. Deskripsi Dataset
Dataset ini terdiri dari 6 fitur atau kolom dan 755 record atau baris. Atribut dalam dataset ini meliputi:  

1. Date: Kolom ini mencatat tanggal setiap data harga. Formatnya adalah YYYY-MM-DD (tahun-bulan-hari), yang mencakup data harian dari 1 Januari 2022 hingga 31 Desember 2024.  
   
2. Open: Harga pembukaan saham Rigetti Computing (RGTI) pada awal hari perdagangan. Nilai ini menunjukkan harga pertama yang tercatat ketika pasar mulai aktif pada hari tersebut.  
   
3. High: Harga tertinggi yang dicapai oleh saham Rigetti Computing (RGTI) selama hari perdagangan. Nilai ini menunjukkan level maksimum yang dicapai oleh saham pada hari tersebut.  
   
4. Low: Harga terendah yang dicapai oleh saham Rigetti Computing (RGTI) selama hari perdagangan. Nilai ini menunjukkan level minimum yang dicapai oleh saham pada hari tersebut.  
   
5. Close: Harga penutupan saham Rigetti Computing (RGTI) pada akhir hari perdagangan. Nilai ini adalah harga terakhir yang tercatat sebelum pasar tutup.  
   
6. Adj Close: Harga penutupan yang telah disesuaikan untuk faktor-faktor tertentu, seperti aksi korporasi atau pembagian dividen. Dalam kasus saham Rigetti Computing, nilai ini biasanya sama dengan harga penutupan kecuali ada penyesuaian khusus.  
   
7. Volume: Jumlah total saham Rigetti Computing (RGTI) yang diperdagangkan selama hari tersebut. Volume menunjukkan tingkat aktivitas perdagangan dan dapat digunakan untuk mengukur minat pasar pada saham tersebut.  

Data ini diambil untuk mendukung analisis dan pengembangan model prediksi harga saham Rigetti yang berbasis data dan akurat. Dataset ini mencakup informasi penting yang relevan untuk memahami dinamika harga saham dalam jangka waktu 2022 hingga 2024.


```{code-cell}
df.info()
print('Ukuran data ', df.shape)
```

```{code-cell}
df.dtypes
```

Jenis Data

1. Open: Data numerik (kontinu), menunjukkan harga pembukaan saham Rigetti Computing (RGTI) pada awal hari perdagangan. Nilainya berupa pecahan desimal untuk mencerminkan perubahan harga dengan presisi tinggi.  

2. High: Data numerik (kontinu), mencatat harga tertinggi yang dicapai saham Rigetti Computing (RGTI) selama hari perdagangan. Nilai ini kontinu karena dapat memiliki pecahan desimal.  

3. Low: Data numerik (kontinu), menunjukkan harga terendah yang dicapai saham Rigetti Computing (RGTI) selama hari perdagangan. Data ini bersifat kontinu karena dapat memiliki pecahan desimal.  

4. Close: Data numerik (kontinu), menunjukkan harga penutupan saham Rigetti Computing (RGTI) pada akhir hari perdagangan. Nilainya kontinu dan penting untuk analisis pergerakan harga harian.  

5. Adj Close: Data numerik (kontinu), menunjukkan harga penutupan yang telah disesuaikan untuk mencerminkan peristiwa seperti pembagian dividen atau aksi korporasi lainnya. Data ini penting untuk analisis historis yang lebih akurat.  

6. Volume: Data numerik (diskrit), menunjukkan jumlah unit saham Rigetti Computing (RGTI) yang diperdagangkan selama hari tersebut. Karena volume dihitung dalam bilangan bulat (jumlah unit saham), data ini bersifat diskrit.  

Fitur-fitur ini mencakup informasi penting untuk memahami dinamika harga saham Rigetti Computing dari data historis, dengan karakteristik yang dapat mendukung analisis dan pengembangan model prediksi berbasis data yang akurat.

## Eksplorasi 

```{code-cell}
# Mencari Missing Value
df.isnull().sum()
```

```{code-cell}
# Deskripsi Statistik
df.describe()
```

- count: Menghitung jumlah entri valid (tidak kosong) dalam kolom.  
- mean: Menghitung rata-rata dari seluruh nilai dalam kolom.  
- std: Mengukur standar deviasi, yang menunjukkan seberapa jauh nilai-nilai dalam kolom menyebar dari rata-rata.  
- min: Menampilkan nilai terkecil dalam kolom.  
- 25%: Kuartil pertama, menunjukkan bahwa 25% data memiliki nilai lebih kecil atau sama dengan nilai ini.  
- 50% (Median): Kuartil kedua, menunjukkan nilai tengah—setengah dari data berada di bawah dan setengah lainnya di atas nilai ini.  
- 75%: Kuartil ketiga, menunjukkan bahwa 75% data memiliki nilai lebih kecil atau sama dengan nilai ini.  
- max: Menampilkan nilai terbesar dalam kolom. 

## Tren Setiap Fitur
Berikutnya, untuk menganalisis dinamika harga saham Rigetti Computing (RGTI), akan dilakukan visualisasi tren dari setiap fitur dalam data historis. Visualisasi ini mencakup pola pergerakan harga pembukaan (Open), harga tertinggi (High), harga terendah (Low), harga penutupan (Close), serta volume perdagangan (Volume) sepanjang waktu. Analisis ini bertujuan untuk mengidentifikasi pola pergerakan harga saham Rigetti selama periode tertentu dan memberikan wawasan yang lebih mendalam untuk mendukung pengambilan keputusan investasi.

```{code-cell}
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np  # Tambahkan ini untuk memperbaiki error

# Membuat subplot otomatis berdasarkan jumlah kolom dalam dataframe
plt.figure(figsize=(9, int(np.ceil(len(df.columns) / 3))*3))

for i, col in enumerate(df.columns):
    plt.subplot(int(np.ceil(len(df.columns) / 3)), 3, i + 1)
    sns.lineplot(data=df, x='Date', y=col)
    plt.title(f'Trend of {col}')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
```

## Korelasi Antar Fitur
Korelasi antar fitur dalam data historis harga saham Rigetti Computing (RGTI) mengacu pada hubungan atau keterkaitan antara dua atau lebih variabel dalam dataset. Dalam konteks ini, fitur-fitur seperti harga pembukaan (Open), harga tertinggi (High), harga terendah (Low), harga penutupan (Close), dan volume perdagangan (Volume) dapat memiliki pengaruh satu sama lain. Melakukan analisis korelasi memungkinkan kita untuk memahami sejauh mana perubahan pada satu fitur terkait dengan perubahan pada fitur lainnya.

```{code-cell}
correlation_matrix = df.corr()

plt.figure(figsize=(7, 3))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Heatmap Korelasi Antar Fitur')
plt.show()
```

## Preprocessing Data

## Menghapus Fitur yang Tidak Relevan
Pada tahap perhitungan matriks korelasi, terdeteksi bahwa fitur ‘volume’ tidak memiliki relevansi atau pengaruh terhadap fitur lainnya, sehingga fitur tersebut akan dihapus. Selain itu, fitur ‘Adj Close’ juga dihilangkan karena nilainya identik dengan fitur ‘Close’.

```{code-cell}
# Menghapus kolom yang tidak digunakan
df = df.drop(columns=['Volume', 'Adj Close'])
df.head()
```

## Rekayasa Fitur
Penelitian ini bertujuan untuk memprediksi harga saham Rigetti Computing pada hari berikutnya. Untuk itu, diperlukan variabel baru yang akan menjadi target prediksi. Fitur ini memberikan indikasi tentang potensi penurunan harga, yang dapat dimanfaatkan oleh investor untuk membeli saham pada harga yang lebih rendah, sehingga meningkatkan kemungkinan memperoleh keuntungan ketika harga saham Rigetti mengalami kenaikan kembali.

```{code-cell}
df['Close Target'] = df['Close'].shift(-1)

df = df[:-1]
df.head()
```

```{code-cell}
# Membuat target untuk n langkah ke depan
for i in range(1, FORECAST_STEPS + 1):
    df[f'Close Target+{i}'] = df['Close'].shift(-i)

# Menghapus baris yang memiliki nilai NaN pada target
df = df[:-FORECAST_STEPS]
```

## Forecasting
Pada tahap ini, parameter FORECAST_STEPS digunakan untuk menentukan jumlah periode atau langkah waktu yang akan diprediksi dalam metode peramalan multi-langkah. Nilainya diatur menjadi 4, yang berarti model akan memproyeksikan nilai target untuk 4 hari atau periode mendatang. Parameter ini dapat disesuaikan sesuai dengan kebutuhan analisis atau tujuan prediksi. Dalam pendekatan Iterative Multi-Step Forecasting, setiap prediksi yang dihasilkan akan digunakan sebagai input untuk prediksi berikutnya. Proses dimulai dengan memprediksi langkah pertama, lalu hasilnya menjadi input untuk langkah kedua, dan seterusnya hingga jumlah langkah yang telah ditentukan oleh FORECAST_STEPS tercapai. Pendekatan ini memungkinkan prediksi bertahap untuk beberapa periode ke depan.

```{code-cell}
# Parameter untuk Multi-Step Forecasting
FORECAST_STEPS = 5  # Jumlah langkah ke depan yang ingin diprediksi
```

```{code-cell}
# Membuat target untuk n langkah ke depan
for i in range(1, FORECAST_STEPS + 1):
    df[f'Close Target+{i}'] = df['Close'].shift(-i)

# Menghapus baris yang memiliki nilai NaN pada target
df = df[:-FORECAST_STEPS]
```

## Normalisasi Data
```{code-cell}
# Import library yang dibutuhkan
# Import library yang dibutuhkan
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Inisialisasi scaler untuk fitur dan target
scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()

# Normalisasi fitur
features = ['Open', 'High', 'Low', 'Close']
df_features_normalized = pd.DataFrame(
    scaler_features.fit_transform(df[features]),
    columns=features,
    index=df.index
)

# Normalisasi target
target_columns = [f'Close Target+{i}' for i in range(1, FORECAST_STEPS + 1)]
df_target_normalized = pd.DataFrame(
    scaler_target.fit_transform(df[target_columns]),
    columns=target_columns,
    index=df.index
)

# Menggabungkan kembali dataframe yang sudah dinormalisasi
df_normalized = pd.concat([df_features_normalized, df_target_normalized], axis=1)
```

## Modelling
Menjelaskan proses apa saja yang akan diproses berdasarkan data yang sudah ada.

a. Pembagian Data
Selanjutnya, data dibagi menjadi data training dan data testing menggunakan fungsi train_test_split, dengan 80% data untuk training dan 20% untuk testing. Proses ini dilakukan dengan opsi shuffle=False agar urutan data tetap terjaga sesuai dengan urutan aslinya. Setelah pembagian, data training (X_train dan y_train) digunakan untuk melatih model, sementara data testing (X_test dan y_test) digunakan untuk mengevaluasi kinerja model yang telah dilatih.

```{code-cell}
from sklearn.model_selection import train_test_split  # Tambahkan ini

# Mengatur fitur (X) dan target (y)
X = df_normalized[features]
y = df_normalized[target_columns]

# Membagi data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=False)
```

b. Pembangunan Model
Pada tahap ini, dilakukan eksperimen dengan menggunakan tiga model utama, yaitu Jaringan Saraf Tiruan (JST), Decision Tree, dan kombinasi SVR dengan Decision Tree. Selain itu, untuk meningkatkan akurasi dan kinerja model, diterapkan teknik ensemble melalui metode bagging.

```{code-cell}
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np
import matplotlib.pyplot as plt

# Model regresi
models = {
    "Jaringan Syaraf Tiruan": MultiOutputRegressor(MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=32)),
    "Decision Tree": DecisionTreeRegressor(random_state=32),
    "SVR": MultiOutputRegressor(SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1))
}

# Dictionary untuk menyimpan hasil evaluasi
results = {}

# Iterasi setiap model
for name, model in models.items():
    # Latih model
    model.fit(X_train, y_train)

    # Prediksi pada data uji
    y_pred = model.predict(X_test)

    # Evaluasi untuk setiap target hari ke depan
    mse_list = []
    mape_list = []
    for i in range(FORECAST_STEPS):
        mse = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
        mape = mean_absolute_percentage_error(y_test.iloc[:, i], y_pred[:, i]) * 100
        mse_list.append(mse)
        mape_list.append(mape)

    # Simpan hasil evaluasi rata-rata
    results[name] = {
        "Average RMSE": np.sqrt(np.mean(mse_list)),
        "Average MAPE": np.mean(mape_list)
    }

    # Kembalikan hasil prediksi ke skala asli
    y_pred_original = scaler_target.inverse_transform(y_pred)
    y_test_original = scaler_target.inverse_transform(y_test)

    # Plot hasil prediksi untuk setiap hari
    plt.figure(figsize=(15, 6))
    for i in range(FORECAST_STEPS):
        plt.plot(
            y_test.index, y_test_original[:, i], label=f"Actual Target+{i+1}", linestyle="dashed"
        )
        plt.plot(
            y_test.index, y_pred_original[:, i], label=f"Predicted Target+{i+1}", alpha=0.7
        )

    # Tambahkan detail plot
    plt.title(f'Actual vs Predicted Values ({name})')
    plt.xlabel('Tanggal')
    plt.ylabel('Kurs')
    plt.legend()
    plt.grid(True)

    # Tampilkan plot
    plt.show()

# Tampilkan hasil evaluasi
print("HASIL EVALUASI MODEL")
for model, metrics in results.items():
    print(f"{model}:")
    print(f"  Average RMSE: {metrics['Average RMSE']:.2f}")
    print(f"  Average MAPE: {metrics['Average MAPE']:.2f}%")

# Cari model dengan Average MAPE terbaik (nilai terkecil)
best_model_name = min(results, key=lambda x: results[x]["Average MAPE"])
best_model = models[best_model_name]

print(f" Model terbaik ({best_model_name})")
```

## Evaluasi
Berdasarkan hasil evaluasi, Jaringan Syaraf Tiruan (RMSE: 0.05, MAPE: 12.74%) adalah model terbaik, dengan akurasi prediksi yang paling tinggi. SVR (RMSE: 0.05, MAPE: 14.96%) memiliki performa serupa tetapi dengan kesalahan relatif lebih besar. Decision Tree (RMSE: 0.09, MAPE: 20.70%) menunjukkan performa terburuk, dengan kesalahan yang lebih tinggi baik dalam RMSE maupun MAPE.

## Deploy Huggingface
link: https://huggingface.co/spaces/windaafitri/Rigetti 





 