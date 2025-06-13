# Laporan Proyek Machine Learning – Aura Rachmawaty

## Project Overview

Sistem rekomendasi buku merupakan salah satu aplikasi penting dalam dunia digital saat ini, khususnya untuk platform seperti Goodreads, Amazon, dan e-commerce lainnya. Dengan meningkatnya volume buku yang tersedia, pengguna kesulitan menemukan buku yang relevan dengan preferensi mereka. Oleh karena itu, dibutuhkan sistem rekomendasi yang mampu menyajikan saran buku secara personal dan otomatis.

Menurut laporan oleh Gomez-Uribe & Hunt (2016), sistem rekomendasi memiliki kontribusi besar terhadap kepuasan pengguna, retensi, dan peningkatan konversi dalam platform online. Dengan pendekatan machine learning, khususnya Collaborative Filtering, kita dapat memodelkan preferensi pengguna berdasarkan pola interaksi mereka terhadap item sebelumnya.

Referensi: 

Gomez-Uribe, C. A., & Hunt, N. (2016). The Netflix Recommender System: Algorithms, Business Value, and Innovation. ACM Transactions on Management Information Systems (TMIS), 6(4), 13.

## Business Understanding

### Problem Statements

- Bagaimana cara memberikan rekomendasi buku yang relevan bagi pengguna berdasarkan riwayat rating mereka?

- Bagaimana memanfaatkan data interaksi pengguna dengan buku (rating) untuk memodelkan preferensi mereka secara akurat?

### Goals

- Menghasilkan daftar buku yang direkomendasikan secara personal untuk pengguna tertentu.

- Meningkatkan akurasi rekomendasi dengan menggunakan model SVD berbasis Collaborative Filtering.

### Solution Approach

- Solution 1: Collaborative Filtering (CF) adalah pendekatan yang merekomendasikan buku berdasarkan pola interaksi pengguna, dengan algoritma SVD yang mampu mempelajari preferensi pengguna meskipun tanpa informasi konten buku.

- Solution 2 : Content-Based Filtering (CBF) adalah pendekatan yang merekomendasikan buku berdasarkan kemiripan fitur konten seperti judul atau pengarang dengan buku-buku yang pernah disukai oleh pengguna tersebut.


## Data Understanding

Dataset yang digunakan berasal dari Kaggle:
[Book Recommendation Dataset] https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset

#### Struktur Dataset:

    | Nama File   | Jumlah Baris | Jumlah Kolom | Deskripsi Singkat                               |
    |-------------|--------------|--------------|-------------------------------------------------|
    | Books.csv   | 271,360      | 8            | Informasi metadata dari buku                    |
    | Users.csv   | 278,858      | 3            | Informasi pengguna                              |
    | Ratings.csv | 1,149,780    | 3            | Interaksi pengguna terhadap buku (rating)       |

#### **Struktur dan Deskripsi Fitur Dataset**

#### 1. Books.csv

Berisi metadata buku dengan atribut sebagai berikut:

- ISBN: Nomor identifikasi unik buku (International Standard Book Number).

- Book-Title: Judul buku.

- Book-Author: Nama penulis buku.

- Year-Of-Publication: Tahun terbit buku.

- Publisher: Nama penerbit buku.

- Image-URL-S: URL gambar kecil dari sampul buku.

- Image-URL-M: URL gambar ukuran sedang dari sampul buku.

- Image-URL-L: URL gambar ukuran besar dari sampul buku.

#### 2. Users.csv

Berisi informasi pengguna dengan fitur:

- User-ID: ID unik pengguna.

- Location: Lokasi pengguna (biasanya dalam format city, state, country).

- Age: Usia pengguna dalam satuan tahun (ada nilai ekstrim seperti <5 dan >100).

#### 3. Ratings.csv

Mewakili interaksi antara pengguna dan buku, dengan fitur:

- User-ID: ID pengguna yang memberikan rating.

- ISBN: ID buku yang diberi rating.

- Book-Rating: Nilai rating dari pengguna untuk buku tersebut, dalam skala 0–10 (0 berarti implicit feedback, bukan penilaian eksplisit).

#### **Kondisi Data Awal**

#### 1. Jumlah Missing Values:

Output .isnull().sum() menunjukkan jumlah missing values berikut:

- Missing values in Books:

        | Kolom          | Jumlah Missing Values |
        |----------------|-----------------------|
        | Book-Author    | 2                     |
        | Publisher      | 2                     |
        | Image-URL-L    | 3                     |

- Missing values in Users:

        | Kolom | Jumlah Missing Values |
        |--------|----------------------|
        | Age    | 110,762              |

- Missing values in Ratings:

    Tidak ditemukan missing values.

#### 2. Jumlah Duplikat:

- Duplicate rows:

        | File        | Jumlah Duplikat |
        |-------------|-----------------|
        | Books.csv   | 0               |
        | Users.csv   | 0               |
        | Ratings.csv | 0               |

#### 3. Distribusi Nilai Rating:

Sebagian besar nilai rating adalah 0, yang berarti rating tidak eksplisit dan perlu difilter. Hanya rating > 0 yang digunakan dalam pemodelan karena dianggap mencerminkan preferensi nyata pengguna.

    ratings_filtered = ratings[ratings['Book-Rating'] > 0] 
    print("Jumlah rating eksplisit:", len(ratings_filtered))

#### 4. Visualisasi dan EDA

- Distribusi usia pengguna menunjukkan adanya outlier, yaitu nilai ekstrem di bawah 5 dan di atas 100.

- Jumlah rating eksplisit setelah filtering sekitar 433.671


## Data Preparation

1. Import Dataset

    File yang digunakan:

    - Books.csv — berisi metadata buku

    - Users.csv — berisi informasi pengguna

    - Ratings.csv — berisi data rating buku dari pengguna

    **Alasan** : Dataset ini digunakan untuk membangun sistem rekomendasi berdasarkan data eksplisit (rating dari pengguna).

2. Filtering Rating Eksplisit

        ratings_filtered = ratings[ratings['Book-Rating'] > 0]

    Tujuan:

    - Menghapus rating dengan nilai 0, karena dianggap sebagai rating implisit (tidak memberikan informasi preferensi nyata).

    **Alasan** : Model collaborative filtering (CF) berbasis SVD memerlukan data rating eksplisit sebagai input pelatihan.

3. Deteksi dan Penanganan Missing Values

        print(books.isnull().sum())   
        print(users.isnull().sum())   
        print(ratings.isnull().sum()) 

    Langkah lanjutan:

    - Hapus baris di Books.csv yang memiliki nilai kosong pada kolom penting.

        books_cleaned = books.dropna(subset=['Book-Author', 'Publisher', 'Image-URL-L'])

    - Tangani nilai ekstrem dan missing value pada kolom Age di Users.csv

        users['Age'] = users['Age'].apply(lambda x: None if x < 5 or x > 100 else x)
        users['Age'].fillna(users['Age'].median(), inplace=True)

    **Alasan** :

    - Data yang tidak lengkap atau tidak valid (seperti usia di bawah 5 atau lebih dari 100) bisa menyebabkan bias atau kesalahan dalam model.

    - Kolom seperti Book-Author, Publisher, dan Image-URL-L penting untuk rekomendasi dan visualisasi, sehingga wajib ada.

4. Cek dan Hapus Data Duplikat

        print("Books:", books.duplicated().sum())
        print("Users:", users.duplicated().sum())
        print("Ratings:", ratings.duplicated().sum())

    **Alasan** : Data duplikat bisa menyebabkan pembelajaran model menjadi berat sebelah dan memperbesar ukuran data tanpa informasi tambahan.

5. Gabungkan Data Rating dengan Metadata Buku

        ratings_merged = pd.merge(ratings_filtered, books_cleaned, on='ISBN')

    **Alasan**: Untuk menghasilkan rekomendasi yang menyertakan judul dan penulis buku, perlu menggabungkan data rating dengan metadata buku.

6. Siapkan Data untuk Model SVD

        reader = Reader(rating_scale=(1, 10))
        data = Dataset.load_from_df(ratings_merged[['User-ID', 'ISBN', 'Book-Rating']], reader)

    **Alasan**: Library surprise membutuhkan format User-ID, Item-ID, dan Rating, sehingga data harus dikonversi terlebih dahulu sesuai format tersebut.

7. Split Data ke Train dan Test

        trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    **Alasan**: Untuk mengevaluasi performa model, diperlukan pemisahan antara data pelatihan dan data pengujian.

## Modeling and Results

- Permasalahan yang Diselesaikan

    Sistem rekomendasi dikembangkan untuk membantu pengguna menemukan buku yang sesuai dengan preferensi mereka berdasarkan data rating historis. Permasalahan ini relevan karena pengguna sering kesulitan menemukan buku baru yang menarik dari jutaan pilihan yang tersedia.

1. Pendekatan Collaborative Filtering (CF) – Matrix Factorization (SVD)

    Algoritma: Singular Value Decomposition (SVD)

    Menggunakan algoritma SVD dari library surprise untuk memperkirakan rating yang belum diberikan oleh pengguna, berdasarkan pola rating pengguna lain yang serupa.

    Langkah:

    - Menggunakan dataset User-ID, ISBN, dan Book-Rating.

    - Model dilatih menggunakan train_test_split.

            model = SVD()
            model.fit(trainset)
            predictions = model.test(testset)
            rmse = accuracy.rmse(predictions)

    - Hasil prediksi diuji pada data test menggunakan metrik RMSE.

    **Top-N Recommendation Output**

    Pada tahap ini, kita memilih **User-ID** dari dataset `ratings_filtered` yang telah difilter sebelumnya, kemudian menampilkan rekomendasi buku untuk pengguna tersebut menggunakan metode **Collaborative Filtering**.

            user_sample = ratings_filtered['User-ID'].sample(1).values[0]

    Fungsi recommend_books_cf(user_sample) dibuat untuk menyajikan Top-N rekomendasi buku bagi pengguna berdasarkan hasil prediksi rating tertinggi.

            recommend_books_cf(user_sample)

    **contoh output :**

    ### Rekomendasi Buku untuk User-ID: 84038

        | No | Book-Title                                                 | Book-Author        | Predicted Rating |
        |----|------------------------------------------------------------|--------------------|------------------|
        | 1  | My Sister's Keeper : A Novel (Picoult, Jodi)               | Jodi Picoult       | 9.201056         |
        | 2  | Harry Potter and the Chamber of Secrets Postca...          | J. K. Rowling      | 9.127904         |
        | 3  | Dilbert: A Book of Postcards                               | Scott Adams	    | 9.114887         |
        | 4  | Wolves of the Calla (The Dark Tower, Book 5)	           | Stephen King       | 9.092526         |
        | 5  | Lonesome Dove                                              | Larry McMurtry     | 9.087445         |

2. Pendekatan Content-Based Filtering (CBF) (Solusi Tambahan)

    Algoritma: Cosine Similarity Based on Book Metadata

    Merekomendasikan buku yang mirip secara konten (judul, penulis, atau genre) dengan buku-buku yang disukai oleh pengguna.

**Collaborative Filtering (SVD)**

Kelebihan:

    1. Personalisasi Tinggi

        - SVD dapat mempelajari pola preferensi pengguna berdasarkan interaksi historis, sehingga mampu memberikan rekomendasi yang sangat personal.

    2. Tidak Memerlukan Informasi Konten Buku

        - SVD hanya membutuhkan data interaksi pengguna (rating), tanpa perlu data tambahan seperti genre, deskripsi, atau penulis buku.

    3. Mampu Menangkap Hubungan Tersembunyi

        - SVD dapat mengidentifikasi hubungan tersembunyi antara pengguna dan item melalui dekomposisi matriks, sehingga bisa merekomendasikan item yang mungkin tidak tampak relevan secara langsung.

    4. Efektif pada Skala Besar

        - SVD termasuk metode yang efisien dalam menangani data dalam skala besar jika dioptimasi dengan baik.

Kekurangan:

    1. Masalah Cold Start

        - Model tidak dapat memberikan rekomendasi yang baik untuk pengguna baru (yang belum pernah memberi rating) atau item baru (yang belum pernah dirating), karena tidak ada data historis untuk digunakan.

    2. Rentan terhadap Data Sparsity (Kekosongan Data)

        - Dataset rating biasanya sangat jarang (sparse), artinya banyak kombinasi pengguna dan buku yang belum pernah terjadi. Hal ini dapat mengurangi akurasi model.

    3. Tidak Bisa Memberi Alasan yang Jelas

        - Rekomendasi berbasis SVD sulit dijelaskan kepada pengguna, karena hasil prediksi berasal dari dimensi laten (faktor tersembunyi) yang tidak memiliki arti langsung.

    4. Bergantung pada Kualitas Data Rating

        - Jika data rating tidak konsisten atau mengandung banyak noise (misalnya rating asal-asalan), maka performa model bisa sangat terpengaruh.

## Evaluation

- Metrik Evaluasi yang Digunakan

    Pada proyek ini, metrik yang digunakan untuk mengukur kinerja model Collaborative Filtering (SVD) adalah:

    **Root Mean Squared Error (RMSE)**

        from surprise import accuracy
        rmse = accuracy.rmse(predictions)

    ### Definisi:

    RMSE adalah metrik yang digunakan untuk mengukur seberapa besar rata-rata kesalahan (error) antara rating yang diprediksi oleh model dan rating aktual yang diberikan oleh pengguna.

    RMSE menghitung akar dari rata-rata kuadrat selisih antara prediksi dan nilai aktual.

    ### Formula RMSE : 
    $$
    RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (r_i - \hat{r}_i)^2}
    $$

    **Keterangan:**
    - r_i: rating aktual dari pengguna ke-i  
    - r^_i: rating prediksi dari model untuk item ke-i  
    - n: jumlah data pada test set

    ### Cara Kerja RMSE:
    1. Model memprediksi rating untuk user–item pair di test set.
    2. Hitung selisih antara rating aktual dan rating prediksi.
    3. Kuadratkan setiap selisih agar tidak ada nilai negatif dan memberi penalti lebih besar pada kesalahan besar.
    4. Hitung rata-rata dari seluruh kuadrat selisih.
    5. Ambil akar dari nilai rata-rata tersebut.


- Hasil Evaluasi 

        RMSE: 1.6343

    **Interpretasi RMSE : 1.6343**

    - Nilai Root Mean Squared Error (RMSE) sebesar 1.6343 berarti bahwa rata-rata kesalahan prediksi rating model terhadap data test adalah sekitar 1.63 poin dari skala rating yang digunakan (1–10).
    
    - Dengan kata lain, jika seorang pengguna sebenarnya memberi rating 8 pada sebuah buku, model bisa memprediksikannya sekitar 6.4 atau 9.6 secara rata-rata.