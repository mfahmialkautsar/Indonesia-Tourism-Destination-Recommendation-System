# Laporan Proyek Machine Learning - Muhamad Fahmi Al Kautsar

## Daftar Isi

- [Project Overview](#project-overview)
- [Business Understanding](#business-understanding)
- [Data Understanding](#data-understanding)
- [Data Preprocessing](#data-preprocessing)
- [Data Preparation](#data-preparation)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Kesimpulan](#kesimpulan)
- [Referensi](#referensi)

## Project Overview

Proyek ini membahas tentang sistem rekomendasi tempat wisata di Indonesia berdasarkan kategori, preferensi pengguna, dan penilaian pengguna lainnya.

Selain memiliki banyak bahasa dan budaya, Indonesia juga kaya akan beragam destinasi wisata yang tersebat di seluruh wilayahnya. Destinasi wisata Indonesia merupakan salah satu daya tarik utama bagi wisatawan domestik maupun asing. Indonesia memiliki kekayaan alam yang luar biasa, mulai dari pantai, pegunungan, hingga pulau-pulau kecil yang indah. Selain itu, Indonesia juga memiliki budaya dan tradisi yang unik yang dapat dinikmati oleh wisatawan yang tentunya memiliki peran untuk Indonesia sendiri seperti: peranan ekonomi, yaitu sebagai sumber devisa negara; peranan sosial, yaitu sebagai penciptaan lapangan pekerjaan; dan peranan kebudayaan yaitu, memperkenalkan kebudayaan dan kesenian. [[1]](https://journal.stp-bandung.ac.id/index.php/barista/article/download/153/79)

Banyaknya destinasi wisata yang dapat dikunjungi seringkali membuat para wisatawan kebingungan tentang destinasi yang paling sesuai dengan keinginan mereka. Oleh karena itu, proyek ini dilakukan untuk membantu para wisatawan menemukan destinasi wisata yang bisa menyesuaikan dengan preferensi mereka dengan sistem rekomendasi destinasi wisata. Sistem rekomendasi ini menggunakan data yang diperoleh dari data tempat wisata itu sendiri, maupun data yang diperoleh dari user. Misalnya ketika user memberi mengunjungi tempat wisata dan memberi rating, maka sistem dapat memberikan rekomendasi tempat wisata lain dengan kategori sejenis. Sistem rekomendasi semacam ini dapat membantu user dalam menentukan pilihannya. [[2]](https://jursistekni.nusaputra.ac.id/article/download/63/38/)

## Business Understanding

### Problem Statements

Pada proyek ini, masalah yang akan dipecahkan adalah:

1. Bagaimana melakukan tahap pra-pemrosesan data sebelum data tersebut dimasukkan ke dalam model _machine learning_?
2. Bagaimana menyiapkan data _places_ dan _ratings_ untuk digunakan dalam melatih model _machine learning_ sistem rekomendasi?
3. Bagaimana membuat model _machine learning_ yang dapat memberikan rekomendasi destinasi wisata yang sesuai dengan rating atau penilaian pengguna?

### Goals

Berdasarkan masalah yang diidentifikasi di atas, maka tujuan dari proyek ini adalah:

1. Untuk melakukan tahap pra-pemrosesan data sebelum data tersebut dimasukkan ke dalam model _machine learning_.
2. Untuk melakukan tahap persiapan data sehingga data siap digunakan melatih model _machine learning_ sistem rekomendasi.
3. Untuk membuat model _machine learning_ dalam memberikan rekomendasi _place_ atau destinasi wisata terbaik sesuai dengan _ratings_ dan pengguna tersebut.

### Solution Statements

Dari rumusan masalah dan tujuan di atas, maka disimpulkan beberapa solusi yang dapat dilakukan untuk mencapai tujuan dari proyek ini, yaitu:

1. Tahap persiapan data atau _data preparation_ dilakukan dengan menggunakan beberapa teknik persiapan data, yaitu:

   - Melakukan proses pengecekan data yang hilang atau _missing value_ pada data _places_ dan _ratings_.
   - Melakukan proses pengecekan data duplikat pada data _place_ dan _ratings_.

2. Tahap membuat model _machine learning_ yang dapat memberikan rekomendasi tempat wisata kepada pengguna berdasarkan _ratings_ atau penilaian pengguna terhadap tempat wisata tertentu. Tahap pembuatan model _machine learning_ untuk sistem rekomendasi menggunakan pendekatan _content-based filtering recommendation_ dan _collaborative filtering recommendation_.

   - **Content-based Filtering Recommendation**

     _Content-based filtering_ adalah metode rekomendasi yang menggunakan informasi tentang item yang akan direkomendasikan untuk memberikan rekomendasi kepada pengguna. Hal ini dilakukan dengan mengidentifikasi fitur atau karakteristik dari item tersebut, seperti judul, deskripsi, atau kategori, kemudian mencocokkannya dengan preferensi atau kebutuhan pengguna yang teridentifikasi melalui data historis atau input pengguna. Content-based filtering biasanya digunakan pada sistem rekomendasi yang memiliki informasi yang cukup tentang item yang direkomendasikan, seperti sistem rekomendasi film atau buku.

     - TF-IDF Vectorizer

       TF-IDF Vectorizer adalah sebuah alat yang digunakan untuk mengubah teks menjadi vektor fitur dengan menghitung bobot teks menggunakan metrik term frequency-inverse document frequency (TF-IDF). Metrik ini mengukur seberapa sering sebuah kata muncul dalam sebuah dokumen dan seberapa jarang kata tersebut muncul di seluruh dokumen yang ada dalam kumpulan data. Vektor fitur yang dihasilkan dari TF-IDF Vectorizer dapat digunakan dalam model machine learning untuk mengelompokkan dokumen atau untuk melakukan klasifikasi.

       TF-IDF menggunakan rumus untuk menghitung bobot masing-masing dokumen terhadap kata kunci, yaitu, [[3]](https://www.researchgate.net/publication/336982602_IMPLEMENTASI_TERM_FREQUENCY_-INVERSE_DOCUMENT_FREQUENCY_TF-IDF_DAN_VECTOR_SPACE_MODEL_VSM_UNTUK_PENCARIAN_BERITA_BAHASA_INDONESIA)

       $$W_{dt} = tf_{dt} \times IDF_{t}$$

       Di mana:
       $d =$ dokumen ke-d
       $t =$ kata ke-t dari kata kunci
       $W_{dt} =$ bobot dokumen ke-d terhadap kata ke-t
       $tf_{dt} =$ banyaknya kata yang dicari pada sebuah dokumen
       $IDF =$ Inversed Document Frequency

     - Cosine Similarity

     Cosine Similarity adalah sebuah metrik similarity yang mengukur kemiripan antara dua vektor. Metrik ini mengukur sudut yang terdapat antara kedua vektor tersebut, dengan menggunakan rumus sebagai berikut:

     $$Similarity = cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|} = \frac{\displaystyle\sum^{n}_{i=1} (wA_i \times wB_i)} {\sqrt{\displaystyle\sum^{n}_{i=1} (wA_{i})^{2} } \sqrt{\displaystyle\sum^{n}_{i=1} (wB_{i})^{2}} }$$

     Di mana:
     $A \cdot B =$ dot product antara vektor A dan vektor B
     $\|A\| =$ panjang vektor A
     $\|B\| =$ panjang vektor B
     $\|A\| \|B\| =$ cross product antara $\|A\|$ dan $\|B\|$
     $wA_i =$ bobot term pada query ke-i
     $wB_i =$ bobot term pada dokumen ke-i
     $i =$ jumlah term dalam kalimat
     $n =$ jumlah vektor

   - **Collaborative Filtering Recommendation**

     Collaborative Filtering Recommendation adalah metode rekomendasi yang menggunakan informasi dari banyak orang untuk memberikan rekomendasi kepada seorang pengguna. Ini menggunakan interaksi pengguna yang terdahulu dengan item-item yang diberikan untuk memberikan rekomendasi kepada pengguna tentang item-item yang mungkin disukainya. Collaborative Filtering Recommendation biasanya dilakukan dengan menggunakan algoritma yang mengklasifikasikan pengguna berdasarkan preferensi dan kemiripan dengan pengguna lain, kemudian menggunakan informasi tersebut untuk memberikan rekomendasi yang cocok untuk pengguna tersebut.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah dataset [Indonesia Tourism Destination](https://www.kaggle.com/datasets/aprabowo/indonesia-tourism-destination) yang diambil dari Kaggle dengan `tourism_with_id.csv` dan `tourism_rating.csv` sebagai dataset yang digunakan.

Berikut adalah _Exploratory Data Analysis_ (EDA) yang merupakan proses investigasi awal pada data untuk menganalisis karakteristik, menemukan pola, anomali, dan memeriksa asumsi pada data.

- **Place**

  **Tabel 1. Informasi Dataset tourism_with_id**

  | #   | Column       | Non-Null Count | Dtype   |
  | --- | ------------ | -------------- | ------- |
  | 0   | Place_Id     | 437 non-null   | int64   |
  | 1   | Place_Name   | 437 non-null   | object  |
  | 2   | Description  | 437 non-null   | object  |
  | 3   | Category     | 437 non-null   | object  |
  | 4   | City         | 437 non-null   | object  |
  | 5   | Price        | 437 non-null   | int64   |
  | 6   | Rating       | 437 non-null   | float64 |
  | 7   | Time_Minutes | 205 non-null   | float64 |
  | 8   | Coordinate   | 437 non-null   | object  |
  | 9   | Lat          | 437 non-null   | float64 |
  | 10  | Long         | 437 non-null   | float64 |

  tourism_with_id terdiri dari 437 baris dan 10 kolom sebagai berikut:

  - Place_Id: kolom yang menunjukkan id dari setiap tempat wisata.
  - Place_Name: kolom yang menunjukkan nama dari setiap tempat wisata.
  - Description: kolom yang menunjukkan deskripsi dari setiap tempat wisata.
  - Category: kolom yang menunjukkan kategori dari setiap tempat wisata.
  - City: kolom yang menunjukkan kota dimana tempat wisata tersebut berada.
  - Price: kolom yang menunjukkan harga tiket masuk ke tempat wisata tersebut.
  - Rating: kolom yang menunjukkan rating dari setiap tempat wisata.
  - Time_Minutes: kolom yang menunjukkan waktu yang diperlukan untuk mengunjungi tempat wisata tersebut.
  - Coordinate: kolom yang menunjukkan koordinat dari setiap tempat wisata.
  - Lat: kolom yang menunjukkan latitude dari setiap tempat wisata.
  - Long: kolom yang menunjukkan longitude dari setiap tempat wisata.

  Berikut adalah visualisasi dari dataset tourism_with_id:

  **Tabel 2. Sampel Dataset tourism_with_id**

  | #   | Place_Id | Place_Name                        | Category      |
  | --- | -------- | --------------------------------- | ------------- |
  | 0   | 1        | Monumen Nasional                  | Budaya        |
  | 1   | 2        | Kota Tua                          | Budaya        |
  | 2   | 3        | Dunia Fantasi                     | Taman Hiburan |
  | 3   | 4        | Taman Mini Indonesia Indah (TMII) | Taman Hiburan |
  | 4   | 5        | Atlantis Water Adventure          | Taman Hiburan |

- **Rating**

  **Tabel 3. Informasi Dataset tourism_rating**

  | #   | Column        | Non-Null Count | Dtype |
  | --- | ------------- | -------------- | ----- |
  | 0   | User_Id       | 10000 non-null | int64 |
  | 1   | Place_Id      | 10000 non-null | int64 |
  | 2   | Place_Ratings | 10000 non-null | int64 |

  tourism_rating terdiri dari 10000 baris dan 3 kolom sebagai berikut:

  - User_Id: identitas unik dari setiap pengguna.
  - Place_Id: identitas unik dari setiap tempat wisata.
  - Place_Ratings: penilaian atau rating yang diberikan oleh pengguna terhadap tempat wisata tertentu.

  Berikut adalah visualisasi dari dataset tourism_rating:

  **Tabel 4. Sampel Dataset tourism_rating**

  | #   | User_Id | Place_Id | Place_Ratings |
  | --- | ------- | -------- | ------------- |
  | 0   | 1       | 179      | 3             |
  | 1   | 1       | 344      | 2             |
  | 2   | 1       | 5        | 5             |
  | 3   | 1       | 373      | 3             |
  | 4   | 1       | 101      | 4             |

## Data Preparation

Tahap ini bertujuan untuk mempersiapkan data yang akan digunakan untuk proses training model. Di sini dilakukan penghapusan kolom yang tidak diperlukan, pembersihkan data _missing value_, dan melakukan pengecekan dan penghapusan data duplikat.

1. **Penghapusan Kolom yang Tidak Diperlukan**

   Pada dataset tourism_with_id, data yang diperlukan hanya ada pada kolom `Place_Id`, `Place_Name`, dan `Category`, jadi hapus yang lain.

   Pada dataset tourism_rating, semua kolom diperlukan, jadi tidak ada kolom yang dihapus.

2. **Pengecekan Missing Values**

   Proses pengecekan data yang hilang atau _missing value_ dilakukan pada masing-masing dataset tourism_with_id dan tourism_rating. Berdasarkan hasil pengecekan, ternyata tidak ada data yang hilang dari kedua dataset tersebut.

## Modeling

Pada tahap pengembangan model _machine learning_ sistem rekomendasi, teknik _content-based filtering recommendation_ dan _collaborative filtering recommendation_ digunakan untuk memberikan rekomendasi tempat terbaik kepada pengguna berdasarkan rating atau penilaian yang telah mereka berikan pada tempat tersebut. Tujuannya adalah untuk memberikan hasil rekomendasi yang tepat sesuai dengan keinginan pengguna.

1. **Content-based Filtering Recommendation**

Beberapa tahap yang dilakukan untuk membuat sistem rekomendasi dengan pendekatan _content-based filtering_ adalah TF-IDF Vectorizer, _cosine similarity_, dan pengujian sistem rekomendasi.

- TF-IDF Vectorizer

  TF-IDF Vectorizer akan melakukan transformasi teks nama tempat menjadi bentuk angka berupa matriks.

- Cosine Similarity

  Cosine similarity digunakan untuk menghitung tingkat kesamaan antara dua data place dengan mengukur sudut antara kedua data tersebut. Teknik ini menghitung tingkat kesamaan dengan menggunakan sudut antara data place yang dianalisis. Hasil perhitungan ini akan memberikan nilai yang menunjukkan tingkat kesamaan antara dua data place, dimana nilai yang mendekati 1 menunjukkan tingkat kesamaan yang tinggi, dan nilai yang mendekati 0 menunjukkan tingkat kesamaan yang rendah.

- Hasil _Top-N Recommendation_

  Setelah data tempat wisata dikonversi menjadi matriks dengan menggunakan TF-IDF Vectorizer, dan tingkat kesamaan antar nama tempat ditentukan dengan menggunakan cosine similarity, selanjutnya dilakukan pengujian terhadap sistem rekomendasi yang menggunakan pendekatan content-based filtering recommendation. Hasil pengujian tersebut dapat dilihat sebagai berikut:

  Diambil sebuah nama tempat yang dipilih oleh pengguna.

  **Tabel 5. Nama Tempat yang Dipilih Pengguna**
  | Place_Id | Place_Name | Category |
  | -------- | ---------- | ----------- |
  | 1 | Monumen Nasional | Budaya |

  Berikut adalah hasil rekomendasi nama tempat berdasarkan kategori yang sama.

  **Tabel 6. Hasil Rekomendasi _Content-based Filtering_**

  | Place_Name                          | Category |
  | ----------------------------------- | -------- |
  | Candi Sewu                          | Budaya   |
  | Museum Benteng Vredeburg Yogyakarta | Budaya   |
  | Museum Satria Mandala               | Budaya   |
  | Kyotoku Floating Market             | Budaya   |
  | Bandros City Tour                   | Budaya   |

  Berdasarkan hasil rekomendasi di atas, dapat dilihat bahwa sistem yang dibuat berhasil memberikan rekomendasi tempat berdasarkan sebuah tempat, yaitu 'Monumen Nasional' dan dihasilkan rekomendasi tempat dengan kategori yang sama, yaitu budaya.

2. **Collaborative Filtering Recommendation**

Tahap-tahap yang dilakukan untuk membuat sistem rekomendasi dengan pendekatan _collaborative filtering_ meliputi _data preparation_, pembagian data menjadi data latih dan data validasi, serta pembangunan model dan pengujian sistem rekomendasi.

- Data Preparation
  Tahap _data preparation_ dilakukan dengan proses encoding fitur User_Id pada dataset ratings dan fitur Place_Id pada dataset ratings menjadi sebuah array. Lalu hasil encoding tersebut akan dilakukan pemetaan atau mapping fitur yang telah dilakukan encoding tersebut ke dalam dataset ratings.
  Berdasarkan hasil encoding dan mapping tersebut, diperoleh jumlah user sebesar 300, jumlah tempat sebesar 437, nilai rating minimal sebesar 1.0, dan nilai rating maksimal yaitu 5.0.
- Membagi Data Latih dan Data Validasi

  Tahap pembagian dataset diawali dengan mengacak dataset ratings, kemudian melakukan pembagian menjadi data latih dan data validasi, yaitu dengan rasio data latih banding data validasi sebesar 80:10.

- Model Development dan Hasil Rekomendasi

  Dari model _machine learning_ yang telah dibangun menggunakan layer embedding dan _regularizer_, serta _adam optimizer_, _binary crossentropy loss function_, dan metrik RMSE (_Root Mean Squared Error_), diperoleh hasil pengujian sistem rekomendasi tempat wisata dengan pendekatan _collaborative filtering_.

  Berdasarkan hasil rekomendasi tempat di atas, dapat dilihat bahwa sistem rekomendasi mengambil pengguna acak (14), lalu dilakukan pencarian tempat dengan rating terbaik dari user tersebut.

  - Margasatwa Muara Angke: **Cagar Alam**
  - Situs Warungboto: **Taman Hiburan**
  - Stone Garden Citatah: **Taman Hiburan**
  - Gua Pawon: **Cagar Alam**
  - Semarang Chinatown: **Budaya**

  Selanjutnya, sistem akan menampilkan 10 daftar tempat yang direkomendasikan berdasarkan kategori yang dimiliki terhadap data pengguna acak tadi. Dapat dilihat bahwa sistem merekomendasikan beberapa tempat dengan kategori yang sama, seperti

  - Pantai Goa Cemara: **Bahari**
  - Desa Wisata Kelor: **Taman Hiburan**
  - Pantai Kukup: **Bahari**
  - Pantai Pok Tunggal: **Bahari**
  - Balai Kota Surabaya: **Budaya**

## Evaluation

1. **Content-based Filtering Recommendation**

   Tahap evaluasi untuk sistem rekomendasi dengan _content-based filtering_ dapat menggunakan metrik _precision_.

   $$precision = \frac{TP}{TP + FP}$$

   Di mana:
   $TP =$ _True Positive_; rekomendasi yang sesuai
   $FP =$ _False Positive_; rekomendasi yang tidak sesuai

   Berdasarkan hasil rekomendasi tempat wisata dengan pendekatan _content-based filtering_ dapat dilihat bahwa hasil yang diberikan oleh sistem rekomendasi berdasarkan tempat wisata **Monumen Nasional** dengan kategori **Budaya**, menghasilkan 5 rekomendasi judul tempat wisata yang tepat. Tetapi secara keseluruhan sistem merekomendasikan tempat wisata dengan tepat.

   $$precision = \frac{5}{5 + 0} = 100\%$$

   Dengan begitu, diperoleh nilai _precision_ sebesar **100%**.

2. **Collaborative Filtering Recommendation**

   Tahap evaluasi untuk sistem rekomendasi dengan _collaborative filtering_ menggunakan metrik RMSE (Root Mean Squared Error). Rumus untuk mencari nilai RMSE sebagai berikut,

   $$RMSE=\sqrt{\sum^{n}_{i=1} \frac{y_i - y\\_pred_i}{n}}$$

   Di mana:
   $n =$ jumlah _dataset_
   $i =$ urutan data dalam _dataset_
   $y_i =$ nilai yang sebenarnya
   $y_{pred} =$ nilai prediksi terhadap $i$

   Nilai RMSE dari sistem rekomendasi dengan pendekatan _collaborative filtering_ adalah 0.3384 pada _Training RMSE_, dan 0.3512 pada _Validation RMSE_. Sedangkan untuk nilai _training loss_ sebesar 0.6849, dan _validation loss_ sebesar 0.6958.

## Kesimpulan

Dengan begitu, dapat disimpulkan bahwa sistem berhasil melakukan rekomendasi baik dengan pendekatan _content-based filtering_ maupun _collaborative filtering_. _Collaborative filtering_ membutuhkan data penilaian tempat dari pengguna, sedangkan pada _content-based filtering_, data rating tidak dibutuhkan karena sistem akan merekomendasikan berdasarkan konten tempat tersebut, yaitu kategori.

## Referensi

[1] Sari, D.P., 2018. Apakah Ada Peranan Aktivitas Wisata Dalam Peningkatan Ekonomi Daerah Di Kota Bogor?. Barista: Jurnal Kajian Bahasa dan Pariwisata, 5(1), pp.12-22.

[2] Nugroho, F. and Rahayu, M.I., 2020. SISTEM REKOMENDASI PRODUK UKM DI KOTA BANDUNG MENGGUNAKAN ALGORITMA COLLABORATIVE FILTERING. Jurnal Riset Sistem Informasi dan Teknologi Informasi (JURSISTEKNI), 2(3), pp.23-31.

[3] Adli, H.R., 2020. IMPLEMENTASI TERM FREQUENCY-INVERS DOCUMENT FREQUENCY (TF-IDF) DAN COSINE SIMILARITY DALAM PENENTUAN REVIEWER PENELITIAN DOSEN UNIVERSITAS PENDIDIKAN INDONESIA (Doctoral dissertation, Universitas Pendidikan Indonesia).
