# %% [markdown]
# # Recommendation System

# %% [markdown]
# # 1. Data Loading

# %% [markdown]
# ## 1.1. Kaggle Credentials

# %% [markdown]
# Kaggle Username dan Kaggle Key diperlukan untuk mengakses dataset pada Kaggle. Kedua variabel tersebut kemudian disimpan dalam environment variable dengan bantuan library `os`.

# %%
import os
os.environ['KAGGLE_USERNAME'] = 'fahmial'
os.environ['KAGGLE_KEY'] = 'b34007481a89d1b149bccd090f285846'

# %% [markdown]
# ## 1.2. Download the Dataset

# %% [markdown]
# Dataset yang digunakan adalah [Indonesia Tourism Destination](https://www.kaggle.com/datasets/aprabowo/indonesia-tourism-destination) dengan `tourism_with_id.csv` dan `tourism_rating.csv` sebagai dataset.

# %%
# Download tourism_with_id.csv dan tourism_rating.csv ke local directory
!kaggle datasets download -d aprabowo/indonesia-tourism-destination -f tourism_with_id.csv -p .
!kaggle datasets download -d aprabowo/indonesia-tourism-destination -f tourism_rating.csv -p .

# %% [markdown]
# # 2. Data Understanding

# %% [markdown]
# ## 2.1. Jumlah Data

# %% [markdown]
# Menampilkan masing-masing dataset yaitu `tourism_with_id.csv` dan `tourism_rating.csv` menggunakan library pandas dari format .csv menjadi dataframe.

# %%
import pandas as pd

places = pd.read_csv('tourism_with_id.csv')
ratings = pd.read_csv('tourism_rating.csv')

print('Jumlah places: ', len(places.Place_Id.unique()))
print('Jumlah ratings: ', len(ratings.Place_Ratings))


# %% [markdown]
# ## 2.2. Univariate Exploratory Data Analysis (EDA)

# %% [markdown]
# Di sini akan dilakukan proses investigasi awal pada data untuk menganalisis karakteristik, menemukan pola, anomali, dan memeriksa asumsi pada data.

# %% [markdown]
# ### 2.2.1. Dataset Tourism Destinations (Places)

# %% [markdown]
# Pengecekan informasi variabel dari dataset places berupa jumlah kolom, nama kolom, jumlah data per kolom dan tipe datanya.

# %%
places.info()

# %% [markdown]
# File ini terdiri dari 10 kolom sebagai berikut:
# 
# - Place_Id: kolom yang menunjukkan id dari setiap tempat wisata.
# - Place_Name: kolom yang menunjukkan nama dari setiap tempat wisata.
# - Description: kolom yang menunjukkan deskripsi dari setiap tempat wisata.
# - Category: kolom yang menunjukkan kategori dari setiap tempat wisata.
# - City: kolom yang menunjukkan kota dimana tempat wisata tersebut berada.
# - Price: kolom yang menunjukkan harga tiket masuk ke tempat wisata tersebut.
# - Rating: kolom yang menunjukkan rating dari setiap tempat wisata.
# - Time_Minutes: kolom yang menunjukkan waktu yang diperlukan untuk mengunjungi tempat wisata tersebut.
# - Coordinate: kolom yang menunjukkan koordinat dari setiap tempat wisata.
# - Lat: kolom yang menunjukkan latitude dari setiap tempat wisata.
# - Long: kolom yang menunjukkan longitude dari setiap tempat wisata.

# %% [markdown]
# Menampilkan sample dataset places.

# %%
places.head()


# %% [markdown]
# Melakukan pengecekan deskripsi statistik dataset places dengan fitur describe().

# %%
places.describe()


# %% [markdown]
# Berdasarkan output diatas, didapatkan deskripsi statistik yaitu:
# 1. count: Jumlah sampel data
# 2. mean: Nilai rata-rata
# 3. std: Standar deviasi
# 4. min: Nilai minimum
# 5. 25%: Kuartil bawah/Q1
# 6. 50%: Kuartil tengah/Q2/median
# 7. 75%: Kuartil atas/Q3
# 8. max: Nilai maksimum

# %% [markdown]
# ### 2.2.2. Dataset Ratings

# %% [markdown]
# Pengecekan informasi variabel dari dataset ratings berupa jumlah kolom, nama kolom, jumlah data per kolom dan tipe datanya.

# %%
ratings.info()


# %% [markdown]
# File ini terdiri dari 3 kolom sebagai berikut:
# 
# - User_Id: identitas unik dari setiap pengguna.
# - Place_Id: identitas unik dari setiap tempat wisata.
# - Place_Ratings: penilaian atau rating yang diberikan oleh pengguna terhadap tempat wisata tertentu.

# %% [markdown]
# Menampilkan sample dataset ratings.

# %%
ratings.head()


# %% [markdown]
# Melakukan pengecekan deskripsi statistik dataset ratings dengan fitur describe().

# %%
ratings.describe()


# %% [markdown]
# # **3. Data Preparation**

# %% [markdown]
# Tahap data preparation merupakan proses transformasi data menjadi bentuk yang dapat diterima oleh model machine learning nanti. Proses data preparation yang dilakukan, yaitu membersihkan data missing value, dan melakukan pengecekan data duplikat.

# %% [markdown]
# ## 3.1. Menghapus Kolom yang Tidak Diperlukan

# %% [markdown]
# Data yang diperlukan hanya ada pada kolom `Place_Id`, `Place_Name`, dan `Category`, jadi hapus yang lain.

# %%
places = places.drop(['Description', 'City', 'Price', 'Rating', 'Time_Minutes', 'Coordinate', 'Lat', 'Long', 'Unnamed: 11', 'Unnamed: 12'], axis=1)

# %% [markdown]
# ## 3.2. Pengecekan Missing Value

# %%
places.isnull().sum()


# %%
ratings.isnull().sum()


# %% [markdown]
# ## 3.3. Pengecekan Data Duplikat

# %%
print(f'Jumlah data places yang duplikat: {places.duplicated().sum()}')
print(f'Jumlah data rating yang duplikat: {ratings.duplicated().sum()}')


# %% [markdown]
# Menghapus duplicate

# %%
ratings.drop_duplicates(inplace = True)

# %% [markdown]
# # **4. Modeling**

# %% [markdown]
# Tahap pengembangan modeling sistem rekomendasi dilakukan untuk membangun model sistem rekomendasi yang dapat menyarankan destinasi wisata terbaik bagi pengguna tertentu berdasarkan rating atau penilaian mereka terhadap destinasi wisata. Teknik yang digunakan untuk membangun model ini adalah _content-based filtering recommendation_ dan _collaborative filtering recommendation_.

# %% [markdown]
# ## 4.1. Model Development dengan Content-based

# %% [markdown]
# Penggunaan teknik _content-based filtering_ dalam sistem rekomendasi bertujuan untuk menyarankan item yang mirip dengan item yang telah disukai pengguna di masa lalu. Teknik ini mempelajari profil minat pengguna baru berdasarkan data dari objek yang telah dinilai oleh pengguna. Dengan menyarankan item yang serupa dengan yang pernah disukai atau sedang dilihat di masa kini, algoritma ini berusaha memberikan rekomendasi yang akurat kepada pengguna. Semakin banyak informasi yang diberikan pengguna, semakin baik akurasi sistem rekomendasi.

# %% [markdown]
# ### 4.1.1. TF-IDF Vectorizer

# %% [markdown]
# TF-IDF Vectorizer digunakan untuk menemukan representasi fitur yang penting dari setiap kategori destinasi wisata. Alat ini dari library scikit-learn akan mengubah nilai-nilai tersebut menjadi vektor dengan menggunakan metode fit_transform dan transform, serta melakukan pemecahan data menjadi bagian-bagian yang lebih kecil secara langsung.

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

tf = TfidfVectorizer()

tf.fit(places['Category'])

tf.get_feature_names_out()


# %% [markdown]
# Transformasi data tempat pada kolom category menjadi bentuk verktor matriks.

# %%
tfidf_matrix = tf.fit_transform(places['Category'])
tfidf_matrix.shape


# %% [markdown]
# Mengubah bentuk vectorizer yaitu vektor menjadi bentuk matriks.

# %%
tfidf_matrix.todense()


# %%
pd.DataFrame(
    tfidf_matrix.todense(),
    columns=tf.get_feature_names_out(),
    index=places.Place_Name
).sample(10, axis=0)


# %% [markdown]
# ### 4.1.2. Cosine Similarity

# %% [markdown]
# Melakukan perhitungan derajat kesamaan atau similatiry degree antar nama tempat wisata dengan teknik cosine similarity menggunakan library scikit-learn.

# %%
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(tfidf_matrix)
cosine_sim


# %% [markdown]
# Mengubah matriks cosine similarity menjadi bentuk dataframe antar nama tempat (destinasi wisata).

# %%
cosine_sim_df = pd.DataFrame(
    cosine_sim, index=places.Place_Name, columns=places.Place_Name)
print('Shape:', cosine_sim_df.shape)

cosine_sim_df.sample(10, axis=0)


# %% [markdown]
# ### 4.1.3. Recommendation Testing

# %% [markdown]
# Melakukan pendefinisian fungsi place_recommendations untuk menampilkan hasil rekomendasi tempat berdasarkan kesamaan kategori dari sebuah tempat.

# %%
def place_recommendations(place_name, similarity_data=cosine_sim_df, items=places[['Place_Name', 'Category']], k=5):
    index = similarity_data.loc[:,place_name].to_numpy().argpartition(range(-1, -k, -1))
    closest = similarity_data.columns[index[-1:-(k+2):-1]]
    closest = closest.drop(place_name, errors='ignore')
    return pd.DataFrame(closest).merge(items).head(k)

# %%
place_name = 'Monumen Nasional'
places[places.Place_Name.eq(place_name)]


# %%
place_recommendations(place_name)

# %% [markdown]
# Berdasarkan hasil rekomendasi di atas, dapat dilihat bahwa sistem yang dibuat berhasil memberikan rekomendasi tempat berdasarkan sebuah tempat, yaitu 'Monumen Nasional' dan dihasilkan rekomendasi tempat dengan kategori yang sama, yaitu budaya.

# %% [markdown]
# ## 4.2. Model Development dengan Collaborative Filtering

# %% [markdown]
# Collaborative Filtering adalah teknik merekomendasikan item yang mirip dengan preferensi pengguna yang sama di masa lalu, misalnya berdasarkan penilaian tempat yang telah diberikan oleh seorang pengguna. Sistem akan merekomendasikan tempat berdasarkan riwayat penilaian pengguna tersebut terhadap tempat dan kategorinya.

# %% [markdown]
# ### 4.2.1. Data Preparation

# %%
import pandas as pd
import numpy as np
from zipfile import ZipFile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import matplotlib.pyplot as plt


# %% [markdown]
# Proses encoding fitur User_Id pada dataset ratings menjadi array.

# %%
user_ids = ratings['User_Id'].unique().tolist()
print('list User_Id: ', user_ids)

user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
print('encoded User_Id : ', user_to_user_encoded)

user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}
print('encoded angka ke User_Id: ', user_encoded_to_user)


# %% [markdown]
# Proses encoding fitur Place_Id pada dataset ratings menjadi array.

# %%
place_ids = ratings['Place_Id'].unique().tolist()
place_to_place_encoded = {x: i for i, x in enumerate(place_ids)}
place_encoded_to_place = {i: x for i, x in enumerate(place_ids)}

# %% [markdown]
# Melakukan mapping atau pemetaan kolom user dan place ke dataset ratings yang berkaitan.

# %%
ratings['user'] = ratings['User_Id'].map(user_to_user_encoded)
ratings['place'] = ratings['Place_Id'].map(place_to_place_encoded)


# %% [markdown]
# Melakukan pengecekan jumlah user, jumlah tempat, penilaian minimal, dan penilaian maksimal.

# %%
users_count = len(user_to_user_encoded)
place_count = len(place_encoded_to_place)

ratings['rating'] = ratings['Place_Ratings'].values.astype(np.float32)

min_rating = min(ratings['rating'])
max_rating = max(ratings['rating'])

print(f'Users Count: {users_count}')
print(f'Places Count: {place_count}')
print(f'Min rating: {min_rating}')
print(f'Max rating: {max_rating}')


# %% [markdown]
# ### 4.2.2. Split Data Latih dan Data Validasi

# %% [markdown]
# Mengacak dataset ratings.

# %%
ratings = ratings.sample(frac=1, random_state=42)
ratings


# %% [markdown]
# Membagi dataset menjadi data latih (train) dan data uji (test), yaitu sebesar 20% data uji dan 80% data latih.

# %%
x = ratings[['user', 'place']].values
y = ratings['rating'].apply(lambda x: (
    x - min_rating) / (max_rating - min_rating)).values

train_indices = int(0.8 * ratings.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:]
)

print(x, y)


# %% [markdown]
# ### 4.2.3. Model Development

# %% [markdown]
# Melakukan pendefinisian kelas RecommenderNet untuk membangun model klasifikasi teks tersebut. Model ini akan memberikan rekomendasi kepada pengguna berdasarkan preferensi atau kecenderungan pengguna di masa lalu. Model ini dapat digunakan dalam berbagai bidang, seperti rekomendasi film, musik, produk, dan lain-lain. RecommenderNet menggunakan algoritma pembelajaran mesin seperti collaborative filtering atau content-based filtering untuk menentukan rekomendasi yang tepat untuk pengguna.
# 
# Parameter yang digunakan dalam model ini adalah:
# - users_count: jumlah user yang akan jadi input dimension pada user embedding, tepatnya sebagai jumlah elemen dari vocabulary atau kata-kata yang digunakan dalam input data
# - place_count: jumlah tempat yang akan jadi input dimension pada tempat embedding, tepatnya sebagai jumlah elemen dari vocabulary atau kata-kata yang digunakan dalam input data
# - embedding_size: ukuran embedding akan jadi output dimension pada user embedding dan tempat embedding, yaitu jumlah fitur yang dihasilkan oleh Embedding layer, yang merupakan hasil pengurangan dimensi dari input data.
# 
# Embedding layer ini akan mengubah representasi numerik dari input data menjadi representasi vektor yang lebih bermakna dan dapat dipahami oleh model machine learning.

# %%
class RecommenderNet(tf.keras.Model):
  def __init__(self, users_count, place_count, embedding_size, **kwargs):
    super(RecommenderNet, self).__init__(**kwargs)
    self.users_count = users_count
    self.place_count = place_count
    self.embedding_size = embedding_size
    self.user_embedding = layers.Embedding(
        users_count,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-8)
    )
    self.user_bias = layers.Embedding(users_count, 1)
    self.place_embedding = layers.Embedding(
        place_count,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-8)
    )
    self.place_bias = layers.Embedding(place_count, 1)
    
  def call(self, inputs):
    user_vector = self.user_embedding(inputs[:,0])
    user_bias = self.user_bias(inputs[:, 0])
    place_vector = self.place_embedding(inputs[:, 1])
    place_bias = self.place_bias(inputs[:, 1])
    
    dot_user_place = tf.tensordot(user_vector, place_vector, 2) 
    
    x = dot_user_place + user_bias + place_bias
    
    return tf.nn.sigmoid(x)

# %% [markdown]
# Proses kompilasi atau compile dengan:
# - binary crossentropy loss function: loss function untuk menghitung loss pada model klasifikasi biner.
# - adam optimizer: algoritma optimisasi yang digunakan untuk mengupdate bobot pada model machine learning secara efisien.
# - metrik RMSE (Root Mean Square Error): metrik yang digunakan untuk mengukur seberapa jauh hasil prediksi dari model dari nilai aktual. RMSE dihitung dengan mencari rata-rata dari kuadrat error yang diakumulasikan dari seluruh data.

# %%
model = RecommenderNet(users_count, place_count, 50)

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)


# %% [markdown]
# Menambahkan callback EarlyStopping yang akan menghentikan training jika tidak ada peningkatan selama 5 epochs.

# %%
from keras.callbacks import  EarlyStopping

callbacks = EarlyStopping(
    min_delta=0.0001,
    patience=5,
    restore_best_weights=True,
)


# %% [markdown]
# Melatih model.

# %%
history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=8,
    epochs=100,
    validation_data=(x_val, y_val),
    callbacks=[callbacks]
)


# %% [markdown]
# Visualisasi grafik data training dan testing untuk masing-masing metrik Root Mean Square Error dan loss function.

# %%
plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('model error')
plt.ylabel('root_mean_squared_error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# %%
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# %% [markdown]
# ### 4.2.4. Tes Rekomendasi

# %% [markdown]
# Melakukan uji coba atau tes rekomendasi tempat yang diberikan. Namun perlu dikertahui terlebih dahulu untuk variabel khusus orang yang belum pernah mengunjungi tempat tersebut (belum memberikan rating) dengan place_not_rated.

# %%
place_df = places
ratings_df = ratings

# %%
user_id = ratings_df.User_Id.sample(1).iloc[0]
place_rated = ratings_df[ratings_df.User_Id == user_id]

place_not_rated = place_df[~place_df['Place_Id'].isin(
    place_rated.Place_Id.values)]['Place_Id']
place_not_rated = list(
    set(place_not_rated).intersection(set(place_to_place_encoded.keys()))
)

place_not_rated = [
    [place_to_place_encoded.get(x)] for x in place_not_rated]
user_encoder = user_to_user_encoded.get(user_id)
user_place_array = np.hstack(
    ([[user_encoder]] * len(place_not_rated), place_not_rated)
)


# %% [markdown]
# Melakukan pengujian prediksi hasil rekomendasi tempat berdasarkan nama tempat dan kategori.

# %%
ratings = model.predict(user_place_array).flatten()

top_ratings_indices = ratings.argsort()[-10:][::-1]
recommended_place_ids = [
    place_encoded_to_place.get(place_not_rated[x][0]) for x in top_ratings_indices
]

print('Showing recommendations for users: {}'.format(user_id))
print('=====' * 8)
print('Place with high ratings from user')
print('-----' * 8)

top_place_user = (
    place_rated.sort_values(
        by = 'rating',
        ascending=False
    )
    .head(5)
    .Place_Id.values
)

place_df_rows = place_df[place_df['Place_Id'].isin(top_place_user)]
for row in place_df_rows.itertuples():
    print(row.Place_Name + ':', row.Category)

print('-----' * 8)
print('Top 10 place recommendation')
print('-----' * 8)

recommended_place = place_df[place_df['Place_Id'].isin(recommended_place_ids)]
for row in recommended_place.itertuples():
    print(row.Place_Name + ':', row.Category)


# %% [markdown]
# Berdasarkan hasil rekomendasi tempat di atas, dapat dilihat bahwa sistem rekomendasi mengambil pengguna acak (14), lalu dilakukan pencarian tempat dengan rating terbaik dari user tersebut.
# 
# - Margasatwa Muara Angke: **Cagar Alam**
# - Situs Warungboto: **Taman Hiburan**
# - Stone Garden Citatah: **Taman Hiburan**
# - Gua Pawon: **Cagar Alam**
# - Semarang Chinatown: **Budaya**
# 
# Selanjutnya, sistem akan menampilkan 10 daftar tempat yang direkomendasikan berdasarkan kategori yang dimiliki terhadap data pengguna acak tadi. Dapat dilihat bahwa sistem merekomendasikan beberapa tempat dengan kategori yang sama, seperti
# 
# - Pantai Goa Cemara: **Bahari**
# - Desa Wisata Kelor: **Taman Hiburan**
# - Pantai Kukup: **Bahari**
# - Pantai Pok Tunggal: **Bahari**
# - Balai Kota Surabaya: **Budaya**

# %% [markdown]
# # **5. Kesimpulan**

# %% [markdown]
# Dengan begitu, dapat disimpulkan bahwa sistem berhasil melakukan rekomendasi baik dengan pendekatan _content-based filtering_ maupun _collaborative filtering_. _Collaborative filtering_ membutuhkan data penilaian tempat dari pengguna, sedangkan pada _content-based filtering_, data rating tidak dibutuhkan karena sistem akan merekomendasikan berdasarkan konten tempat tersebut, yaitu kategori.

# %%



