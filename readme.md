# Hotel Booking Cancellation Prediction

## 1. Latar Belakang
Di industri perhotelan yang sangat kompetitif, pembatalan reservasi menjadi tantangan besar dalam mengelola pendapatan dan operasional. Hotel sering kali mengalami ketidakpastian dalam peramalan hunian kamar akibat pembatalan mendadak yang dapat mengakibatkan hilangnya pendapatan dan inefisiensi operasional. Ketika reservasi dibatalkan, terutama pada saat-saat terakhir, hotel menghadapi kesulitan untuk mengisi kembali kamar yang kosong, sehingga memengaruhi tingkat okupansi dan pendapatan secara keseluruhan. Selain itu, ketidakpastian dalam tingkat hunian juga berdampak pada perencanaan staf dan sumber daya, yang dapat menyebabkan biaya operasional yang tidak efisien.

Untuk mengatasi masalah ini, hotel memerlukan kemampuan untuk memprediksi kemungkinan pembatalan reservasi dengan akurat. Dengan adanya model prediktif yang efektif, hotel dapat mengoptimalkan strategi overbooking, menyesuaikan harga kamar secara dinamis, dan merencanakan kebutuhan operasional dengan lebih baik. Model ini dapat mempertimbangkan berbagai faktor seperti segmen pasar, tipe pelanggan, jenis deposit, riwayat pembatalan, dan permintaan khusus pelanggan untuk menghasilkan prediksi yang lebih akurat.

Implementasi solusi ini tidak hanya membantu hotel dalam meminimalkan dampak negatif dari pembatalan reservasi, tetapi juga meningkatkan efisiensi operasional dan kepuasan pelanggan. Dengan strategi yang lebih proaktif dalam mengelola inventaris kamar dan menentukan harga secara optimal, hotel dapat mempertahankan daya saingnya di pasar yang kompetitif.

---

## 2. Stakeholders and Their Needs
- **Manajer Pendapatan Hotel (Revenue Managers):** 
  - **Kebutuhan:** Prediksi pembatalan reservasi untuk mengoptimalkan strategi overbooking dan penetapan harga guna memaksimalkan pendapatan.
  - **Manfaat:** Membantu dalam perencanaan penjualan kamar dan strategi pendapatan yang lebih efektif.
# 
- **Tim Operasional:** 
  - **Kebutuhan:** Informasi yang akurat tentang tingkat hunian untuk perencanaan staf dan pengalokasian sumber daya yang efisien.
  - **Manfaat:** Meningkatkan efisiensi operasional dan mengurangi biaya yang tidak perlu.

- **Tim Pemasaran:** 
  - **Kebutuhan:** Memahami pola pembatalan untuk menyusun kampanye pemasaran yang tepat sasaran dan meningkatkan retensi pelanggan.
  - **Manfaat:** Mengurangi tingkat pembatalan dengan strategi pemasaran yang lebih efektif dan personalisasi penawaran.

- **Pelanggan (Tamu Hotel):**
  - **Kebutuhan:** Kepastian akan ketersediaan kamar dan pengalaman pemesanan yang andal.
  - **Manfaat:** Meningkatkan kepuasan dan loyalitas pelanggan dengan mengurangi overbooking dan memastikan ketersediaan kamar.

---
# 
## 3. Permasalahan yang Dihadapi
Pembatalan reservasi hotel menyebabkan gangguan pada prediksi pendapatan, pengelolaan inventaris kamar, dan perencanaan operasional, yang mengakibatkan:
- **Kehilangan Pendapatan:** Pembatalan reservasi mengakibatkan hilangnya peluang pendapatan, terutama jika kamar tidak terisi kembali.
- **Inefisiensi Operasional:** Kekurangan atau kelebihan staf akibat prediksi tingkat hunian yang tidak akurat.
- **Ketidakpuasan Pelanggan:** Ketidakmampuan untuk memenuhi permintaan kamar akibat pembatalan mendadak atau no-show.
# 
---
# 
## 4. Pentingnya Penyelesaian Masalah
- **Optimasi Pendapatan:** Prediksi pembatalan memungkinkan hotel untuk menerapkan strategi overbooking dan mengoptimalkan harga kamar, sehingga pendapatan dapat lebih dimaksimalkan.
- **Efisiensi Operasional:** Peramalan yang akurat memungkinkan perencanaan staf dan inventaris yang lebih baik, sehingga mengurangi biaya operasional.
- **Peningkatan Kepuasan Pelanggan:** Dengan meminimalkan overbooking atau kekurangan kamar mendadak, kepuasan dan loyalitas pelanggan dapat meningkat.
# 
---
# 
## 5. Tujuan dan Target
Tujuan dari proyek ini adalah membangun model machine learning untuk memprediksi pembatalan reservasi hotel secara akurat, sehingga hotel dapat:
- **Meminimalkan Kehilangan Pendapatan:** Dengan menerapkan strategi overbooking berdasarkan prediksi pembatalan.
- **Mengoptimalkan Efisiensi Operasional:** Melalui prediksi permintaan dan pengalokasian sumber daya yang akurat.
- **Meningkatkan Kepuasan Pelanggan:** Dengan mengurangi skenario overbooking dan memastikan ketersediaan kamar.
# 
---

Dataset ini berisi informasi pemesanan untuk sebuah hotel yang berlokasi di Portugal dan mencakup data terkait reservasi kamar untuk masing-masing pelanggan. Semua informasi yang dapat mengidentifikasi pribadi telah dihapus dari data ini.

Dataset ini memiliki 83.573 entri dan 11 kolom. Berikut adalah gambaran singkat mengenai kolom-kolom yang terdapat dalam dataset:

Kolom dan Tipe Data:
- country (object) - Negara asal pelanggan (terdapat nilai kosong: 351)
- market_segment (object) - Segmen pasar dari pelanggan
- previous_cancellations (int64) - Jumlah pembatalan sebelumnya yang pernah dilakukan oleh pelanggan
- booking_changes (int64) - Jumlah perubahan yang dilakukan pada pemesanan setelah - reservasi dilakukan hingga waktu check-in atau pembatalan
- deposit_type (object) - Jenis deposit yang dilakukan untuk menjamin pemesanan
- days_in_waiting_list (int64) - Jumlah hari pemesanan berada di daftar tunggu sebelum dikonfirmasi
- customer_type (object) - Tipe pelanggan berdasarkan jenis pemesanan
- reserved_room_type (object) - Tipe kamar yang dipesan (dikodekan untuk menjaga anonimitas)
- required_car_parking_spaces (int64) - Jumlah tempat parkir mobil yang dibutuhkan oleh pelanggan
- total_of_special_requests (int64) - Jumlah permintaan khusus yang diajukan oleh pelanggan (misalnya: twin bed atau lantai tinggi)
- is_canceled (int64) - Variabel target yang menunjukkan apakah pemesanan dibatalkan (1) atau tidak (0)

## Feature Selection and Feature Engineering

### Feature Engineering
#### Alasan Menggunakan OneHotEncoder

OneHotEncoder digunakan untuk mengonversi fitur kategorikal menjadi bentuk numerik biner agar dapat digunakan dalam model machine learning.
Dalam dataset ini, terdapat beberapa kolom kategorikal seperti `market_segment`, `deposit_type`, dan `customer_type`.
Model machine learning umumnya tidak dapat memahami data dalam bentuk teks, sehingga diperlukan proses encoding.

**Mengapa Menggunakan OneHotEncoder?**
- OneHotEncoder mengubah setiap kategori unik menjadi kolom biner (0 atau 1).
- Cara ini **menghindari masalah urutan** yang mungkin muncul jika menggunakan Label Encoding.
  - Contoh: `market_segment` memiliki kategori seperti `Online TA`, `Direct`, `Groups`.
    - Jika menggunakan Label Encoding, kategori ini akan menjadi 0, 1, dan 2 yang secara numerik memberi kesan bahwa `Groups` lebih besar dari `Direct`, padahal tidak demikian.
    - OneHotEncoder mengubahnya menjadi kolom terpisah seperti: `Online_TA`, `Direct`, dan `Groups` dengan nilai 0 atau 1, sehingga tidak ada urutan yang salah.

- Menggunakan `drop='first'` untuk **menghindari multikolinearitas**, yaitu ketika informasi kategori bisa didapatkan dari kombinasi kategori lainnya.
  - Contoh: Jika ada tiga kategori (`A`, `B`, `C`), maka cukup dua kolom (`A` dan `B`) karena jika keduanya bernilai 0, artinya pasti `C`.

- OneHotEncoder cocok digunakan ketika:
  - Fitur kategorikal **tidak memiliki urutan** (nominal).
  - Jumlah kategori relatif **sedikit**, sehingga tidak terlalu banyak menambah jumlah kolom.

Dengan menggunakan OneHotEncoder, data kategorikal dapat diubah menjadi bentuk numerik tanpa memberikan makna urutan atau hubungan antar kategori.
Ini membuat model machine learning dapat **menginterpretasikan data secara akurat** dan **menghindari kesalahan logika**.

### Feature Selection
Pada tahap Feature Selection, kita perlu memilih fitur (kolom) yang paling relevan dan memiliki pengaruh kuat terhadap target variabel.
Proses ini bertujuan untuk meningkatkan akurasi dan efisiensi model dengan menghilangkan fitur yang kurang informatif.

Dalam kasus ini, kita memilih `is_canceled` sebagai target variabel, karena kita ingin memprediksi apakah pemesanan akan dibatalkan atau tidak.
Nilai 1 pada `is_canceled` menunjukkan bahwa pemesanan dibatalkan, sedangkan nilai 0 menunjukkan bahwa pemesanan tidak dibatalkan.

Semua kolom lainnya digunakan sebagai fitur input (X) karena berdasarkan analisis sebelumnya, kolom-kolom ini menunjukkan hubungan dengan target variabel.

Dengan melakukan Feature Selection yang tepat, kita dapat meningkatkan performa model dan mengurangi overfitting.

## Modeling



Pada tahap ini, kita akan membangun model machine learning untuk memprediksi kemungkinan pembatalan pemesanan hotel (`is_canceled`).
Tujuan utama dari modeling ini adalah untuk **membandingkan performa beberapa algoritma klasifikasi** dan memilih model yang memberikan hasil paling akurat.

### Model Klasifikasi yang Akan Dibandingkan:
1. **Logistic Regression:**
   - Merupakan model klasifikasi yang sederhana dan interpretatif.
   - Digunakan sebagai baseline untuk membandingkan performa model lainnya.

2. **Random Forest:**
   - Model berbasis tree ensemble yang kuat dalam menangani data non-linear dan outlier.
   - Cocok untuk dataset ini yang memiliki fitur numerik dan kategorikal hasil dari OneHotEncoder.

3. **XGBoost:**
   - Model boosting yang sangat populer dan sering digunakan dalam kompetisi data science.
   - Memiliki kemampuan generalisasi yang baik dan dapat menangani **class imbalance** dengan baik.

4. **K-Nearest Neighbors (KNN):**
   - Model berbasis instance yang mengklasifikasikan data berdasarkan kedekatan dengan data lain di sekitarnya.
   - Berguna untuk mendeteksi pola lokal dalam data, namun sensitif terhadap scaling data.

### Alasan Memilih Model Ini:
- Model-model ini dipilih untuk **membandingkan pendekatan yang berbeda**:
  - **Logistic Regression**: Pendekatan linear yang sederhana dan interpretatif.
  - **Random Forest dan XGBoost**: Pendekatan tree-based yang kuat dalam menangani non-linearitas.
  - **KNN**: Pendekatan instance-based yang mempertimbangkan kedekatan data.
- Dengan membandingkan model-model ini, kita dapat **memilih model dengan performa terbaik** berdasarkan evaluasi metrik yang sesuai.

### Evaluasi Model:
- **Accuracy**: Untuk melihat proporsi prediksi yang benar.
- **Recall**: Untuk mengevaluasi kemampuan model dalam mendeteksi pembatalan (kelas 1).
- **Precision**: Untuk mengevaluasi akurasi dari prediksi pembatalan.
- **F1 Score**: Untuk menyeimbangkan Precision dan Recall.
- **ROC AUC**: Untuk mengevaluasi performa model pada berbagai threshold.

Tahap ini akan meliputi:
- Training dan evaluasi pada setiap model.
- **Hyperparameter Tuning** pada model dengan performa terbaik.
- Membuat **confusion matrix** dan **classification report** untuk melihat performa secara mendalam.

Dengan pendekatan ini, kita diharapkan dapat **memilih model yang paling akurat dan andal** dalam memprediksi pembatalan pemesanan hotel.

#### Standard Scaler 
StandardScaler digunakan untuk **menstandarisasi skala data numerik** sehingga memiliki:
- **Mean = 0** dan **Standard Deviation = 1**.
- Tujuannya agar semua fitur berada pada skala yang sama dan **tidak mendominasi** model.

## Kesimpulan Pemilihan Model Machine Learning

Setelah membandingkan empat model klasifikasi: 
- Logistic Regression
- XGBoost
- Random Forest
- K-Nearest Neighbors (KNN)

Berdasarkan hasil evaluasi, **XGBoost menunjukkan kinerja terbaik di semua metrik**:
- **Accuracy = 81.12%** (tertinggi di antara semua model)
- **ROC AUC = 0.888** (kemampuan terbaik dalam membedakan kelas)
- **F1 Score = 0.729** (harmoni terbaik antara Precision dan Recall)
- **Precision = 0.773** (tingkat False Positive yang rendah)
- **Recall = 0.689** (deteksi pembatalan yang lebih baik)

---

### Model yang Dipilih: XGBoost
- **Mengapa Memilih XGBoost?**
  - **Performa Terbaik di Semua Metrik**: Secara konsisten memiliki nilai tertinggi pada **Accuracy, ROC AUC, F1 Score, Precision, dan Recall**.
  - **Kemampuan Generalisasi yang Baik**: Dengan **ROC AUC yang tinggi (0.888)**, model ini **mampu membedakan kelas secara efektif**.
  - **Recall yang Lebih Tinggi**: Recall yang lebih tinggi (0.689) berarti model **lebih baik dalam mendeteksi pembatalan**, sehingga **mengurangi risiko kekosongan kamar yang tidak terduga**.
  - **Stabil dan Andal**: XGBoost dikenal dengan **stabilitas dan kemampuan generalisasi yang kuat** pada berbagai dataset.

## Hyperparameter Tuning

Setelah memilih XGBoost sebagai model terbaik berdasarkan evaluasi kinerja pada metrik Accuracy, ROC AUC, F1 Score, Precision, dan Recall langkah selanjutnya adalah mengoptimalkan performa model melalui Hyperparameter Tuning

Hyperparameter Tuning bertujuan untuk:
- Mencari kombinasi parameter terbaik yang dapat meningkatkan akurasi dan kemampuan generalisasi model
- Mengoptimalkan trade-off antara Precision dan Recall terutama untuk mendeteksi pembatalan dengan lebih akurat
- Mencegah overfitting dengan menemukan nilai parameter yang tepat

Pada tahap ini, kita akan menggunakan RandomizedSearchCV untuk:
- Menjelajahi ruang hyperparameter yang luas dengan efisien.
- Memilih kombinasi parameter terbaik berdasarkan kinerja model pada data validasi

%% [markdown]
## Hyperparameter Tuning dengan RandomizedSearchCV

Pada tahap ini, dilakukan **Hyperparameter Tuning** pada **XGBoost** menggunakan **RandomizedSearchCV**. 
Tujuan dari langkah ini adalah untuk **menemukan kombinasi parameter terbaik** yang dapat **meningkatkan performa model** dalam **memprediksi pembatalan booking**.

---

### Apa yang Dilakukan?
- **RandomizedSearchCV** digunakan untuk **menguji berbagai kombinasi hyperparameter** secara acak dalam ruang parameter yang ditentukan.
- **ROC AUC digunakan sebagai metrik evaluasi utama**, karena fokusnya adalah pada **kemampuan model dalam membedakan kelas yang dibatalkan dan tidak dibatalkan**.
- **Cross-validation dengan 5-fold** digunakan untuk **memastikan generalisasi model** dan mengurangi risiko overfitting.
- Dilakukan **50 iterasi** untuk mencari kombinasi parameter yang optimal.

---

### Hasil Hyperparameter Tuning:
```
Best Parameters: {'subsample': 1.0, 'reg_lambda': 1, 'reg_alpha': 0, 
                 'n_estimators': 400, 'max_depth': 9, 'learning_rate': 0.05, 
                 'gamma': 0.1, 'colsample_bytree': 0.8}
Best ROC AUC Score during CV: 0.889
```
- **ROC AUC Score tertinggi selama Cross-Validation adalah 0.889**, yang menunjukkan **kemampuan model yang sangat baik dalam membedakan kelas yang dibatalkan dan tidak dibatalkan**.
- **Parameter terbaik ditemukan dengan kombinasi di atas**, yang akan digunakan pada model final untuk **evaluasi pada data testing**.

---

### Kesimpulan:
- **Hyperparameter Tuning berhasil meningkatkan performa model**, terlihat dari **ROC AUC Score sebesar 0.889**.
- Model dengan parameter terbaik ini **diharapkan mampu memprediksi pembatalan dengan lebih akurat**, sehingga **dapat membantu hotel dalam mengoptimalkan manajemen kamar dan meningkatkan pendapatan**.
- Langkah selanjutnya adalah **menggunakan model dengan parameter terbaik ini pada data testing untuk evaluasi final**.

---

## Evaluasi Hasil Setelah Hyperparameter Tuning

Setelah melakukan **Hyperparameter Tuning** pada model **XGBoost**, didapatkan **peningkatan performa** meskipun hanya sedikit. Berikut adalah perbandingan sebelum dan sesudah tuning:

---

### Sebelum Tuning:
```
Accuracy: 81.12%
ROC AUC: 0.888
F1 Score: 0.729
Precision: 0.773
Recall: 0.689
```

### Setelah Tuning:
```
Accuracy: 81.17% (+0.05%)
ROC AUC: 0.889 (+0.05%)
F1 Score: 0.730 (+0.001)
Precision: 0.774 (+0.0008)
Recall: 0.690 (+0.001)
```

---

### Analisis dan Kesimpulan:
- **Peningkatan performa** terlihat pada **semua metrik** meski **hanya sedikit**.
- **ROC AUC meningkat dari 0.888 menjadi 0.889**, menunjukkan **sedikit perbaikan dalam kemampuan model membedakan kelas**.
- **F1 Score, Precision, dan Recall juga meningkat**, yang menunjukkan **peningkatan kemampuan model dalam mendeteksi pembatalan**.
- Meskipun perbedaannya kecil, **hyperparameter tuning tetap efektif dalam mengoptimalkan performa model**.

---

### Implikasi:
- **Model yang sudah dituning** dapat digunakan untuk **prediksi pembatalan yang lebih akurat**.
- Dengan **ROC AUC yang lebih tinggi**, model ini **lebih andal dalam mengklasifikasikan pembatalan dan tidak pembatalan**.
- **Informasi ini dapat membantu hotel dalam pengambilan keputusan operasional dan strategi marketing**.

---

## Kesimpulan

### 1. Penyelesaian Masalah Bisnis
Model yang dikembangkan berhasil menyelesaikan masalah bisnis yang didefinisikan di awal, yaitu memprediksi pembatalan booking hotel. Dengan prediksi yang lebih akurat, pihak hotel dapat mengoptimalkan inventaris kamar, meminimalkan kehilangan pendapatan, dan meningkatkan kepuasan pelanggan melalui perencanaan yang lebih efektif.

### 2. Pencapaian yang Terukur
- Setelah dilakukan Hyperparameter Tuning, model XGBoost menunjukkan performa yang lebih baik dengan metrik:
  - Accuracy: 81.16%
  - ROC AUC: 88.89%
  - F1 Score: 72.97%
- Peningkatan ini menunjukkan bahwa model memiliki kemampuan yang lebih akurat dalam mengidentifikasi kemungkinan pembatalan, sehingga keputusan bisnis dapat lebih efektif.

### 3. Kapan Model Baik Digunakan
- Model ini baik digunakan ketika akurasi prediksi menjadi prioritas untuk mengoptimalkan pemesanan kamar dan meminimalkan dampak pembatalan.
- Cocok digunakan pada:
  - Peak season atau periode permintaan tinggi, di mana pembatalan dapat berdampak signifikan terhadap pendapatan hotel.
  - Saat merencanakan strategi overbooking untuk memaksimalkan okupansi kamar.

### 4. Kapan Model Kurang Dapat Dipercaya
- Model ini kurang dapat diandalkan pada kondisi:
  - Terjadi perubahan tren pasar atau perilaku pelanggan yang berbeda dari pola historis dalam data training.
  - Kondisi eksternal yang tidak terprediksi, seperti pandemi atau krisis ekonomi global, yang dapat mengubah pola pembatalan secara drastis.
  - Saat terdapat segmen pelanggan baru atau perubahan dalam strategi pemasaran yang belum tercermin dalam data historis.

### 5. Dampak Implementasi pada Proses Bisnis
- **Optimisasi Pendapatan**: Dengan memprediksi pembatalan secara akurat, hotel dapat dengan cepat mengisi kamar yang dibatalkan sehingga mengurangi potensi kerugian pendapatan.
- **Efisiensi Operasional**: Model ini membantu dalam perencanaan sumber daya seperti persiapan kamar dan alokasi staf, sehingga operasional hotel menjadi lebih efisien.
- **Strategi Harga dan Promosi yang Lebih Tepat**: Dengan informasi pelanggan yang memiliki kemungkinan tinggi untuk membatalkan, hotel dapat melakukan strategi harga atau memberikan penawaran khusus untuk mengurangi pembatalan.
- **Peningkatan Kepuasan Pelanggan**: Dengan mengurangi risiko overbooking dan meningkatkan ketersediaan kamar, pengalaman pelanggan dapat lebih ditingkatkan.

### 6. Rekomendasi Bisnis
- **Implementasi Model Machine Learning Secara Menyeluruh**: Menerapkan model prediksi ini pada sistem operasional hotel secara real-time untuk mendukung keputusan bisnis yang lebih akurat.
- **Review Kebijakan Pembatalan Hotel**: Melakukan peninjauan ulang terhadap kebijakan pembatalan dan menerapkan pendekatan yang lebih ketat untuk mengurangi jumlah pembatalan.
- **Menghubungi Pelanggan di Waiting List Secara Proaktif**: Menghubungi pelanggan dalam daftar tunggu segera setelah terjadi pembatalan untuk memastikan kamar tidak kosong.
- **Meningkatkan Efisiensi Operasional**: Memastikan jumlah staf yang memadai dan dilatih dengan baik untuk menghadapi situasi hotel yang selalu penuh.
- **Strategi Overbooking yang Lebih Efektif**: Menggunakan prediksi pembatalan untuk mengatur strategi overbooking secara lebih akurat tanpa merugikan pelanggan.
- **Promosi yang Tepat Sasaran**: Memberikan penawaran khusus kepada pelanggan yang memiliki kemungkinan tinggi untuk membatalkan, sehingga dapat mengurangi tingkat pembatalan.
- **Dynamic Pricing**: Mengoptimalkan harga kamar berdasarkan kemungkinan pembatalan untuk memaksimalkan pendapatan.
- **Pengelolaan Stok Kamar yang Lebih Baik**: Mempersiapkan kamar untuk pelanggan yang diprediksi tidak akan membatalkan sehingga meningkatkan efisiensi okupansi.
- **Strategi Komunikasi yang Lebih Baik**: Mengirimkan pengingat atau penawaran fleksibilitas kepada pelanggan yang diprediksi berpotensi membatalkan untuk mempertahankan reservasi mereka.

### 7. Batasan dan Keterbatasan Model
- **Batasan Data**:
  - Model menggunakan data historis yang mungkin tidak mencerminkan tren masa depan yang dinamis.
  - Data imbalance antara pembatalan dan non-pembatalan dapat mempengaruhi performa model.
- **Batasan Model**:
  - Model tidak mempertimbangkan faktor eksternal seperti event lokal, cuaca, atau tren ekonomi yang dapat mempengaruhi keputusan pelanggan.
  - Model hanya melakukan prediksi tanpa menjelaskan alasan di balik keputusan pelanggan.
- **Batasan Waktu dan Implementasi**:
  - Proses Hyperparameter Tuning memerlukan waktu komputasi yang cukup lama.
  - Implementasi model secara real-time membutuhkan integrasi yang matang dengan sistem reservasi hotel.

### 8. Rekomendasi Pengembangan Selanjutnya
- **Integrasi Data Eksternal**: Menggunakan data eksternal seperti cuaca, event lokal, atau tren perjalanan global untuk meningkatkan akurasi prediksi.
- **Pengembangan Dashboard untuk Decision-Making**: Mengintegrasikan model ke dalam dashboard analitik yang dapat digunakan oleh tim manajemen hotel untuk membuat keputusan strategis secara real-time.
- **Pengembangan Fitur Personalisasi**: Memberikan rekomendasi yang lebih personal kepada pelanggan berdasarkan riwayat perilaku dan prediksi pembatalan.
- **Pengujian Model Secara Real-Time**: Melakukan deployment model ke dalam sistem operasional hotel untuk menguji performa dan akurasi prediksi secara real-time.
- **Evaluasi dan Pembaruan Berkala**: Mengikuti tren terbaru dalam perilaku pelanggan dan memperbarui model secara berkala agar tetap relevan.

Dengan kesimpulan dan rekomendasi ini, solusi yang dihasilkan tidak hanya menjawab permasalahan bisnis yang didefinisikan di awal, tetapi juga memberikan dampak nyata pada optimalisasi pendapatan, efisiensi operasional, dan peningkatan pengalaman pelanggan.
