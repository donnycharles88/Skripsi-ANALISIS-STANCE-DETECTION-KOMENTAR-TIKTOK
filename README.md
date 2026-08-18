# 🏛️ Stance Analysis System - IndoBERT & SMOTE

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://stanceapp.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Model](https://img.shields.io/badge/model-IndoBERT%20%2B%20MLP-green.svg)](https://huggingface.co/)

Sistem Analisis **Stance Detection** (Deteksi Keberpihakan) Komentar TikTok berbasis **IndoBERT Transformer** dan **MLP Classifier** dengan penanganan *Class Imbalance* menggunakan **SMOTE** (Synthetic Minority Over-sampling Technique).

🌐 **Demo Aplikasi Live:** [https://stanceapp.streamlit.app/](https://stanceapp.streamlit.app/)

---

## 📌 Ringkasan Proyek

Aplikasi ini dikembangkan untuk mengklasifikasikan keberpihakan publik (stance) pada komentar media sosial TikTok terkait isu/kinerja kepolisian ke dalam 2 kelas utama:

- **Class 0: `Support / Kritis Polisi`** – Komentar yang berisi dukungan atau kritik membangun terhadap kinerja dan integritas kepolisian.
- **Class 1: `Oppose / Neutral`** – Komentar netral, di luar konteks evaluasi institusi, atau tidak menunjukkan penentangan/dukungan langsung.

Proyek ini mengimplementasikan alur ekstraksi fitur konteksual menggunakan vektor embedding `[CLS]` dari **IndoBERT** (`indobert-base-uncased`, 768 dimensi), yang kemudian diklasifikasikan menggunakan **MLP (Multi-Layer Perceptron) Classifier**. Untuk mengatasi ketidakseimbangan data (class imbalance), teknik **SMOTE** diterapkan dan dibandingkan performanya terhadap model *baseline* (tanpa SMOTE).

---

## ✨ Fitur Utama Aplikasi

Aplikasi web didesain interaktif menggunakan **Streamlit** dengan 4 menu utama:

1. **🏠 Beranda**
   - Penjelasan metodologi penelitian, arsitektur integrasi IndoBERT + MLP, serta teknik resolusi *imbalance data* dengan SMOTE.
   - Diagram alur inferensi model dari input teks hingga klasifikasi *stance*.

2. **📊 Eksplorasi Data & Pra-proses**
   - Visualisasi dataset komentar TikTok riil (`.csv`) interaktif.
   - Statistik data: total komentar, total *likes*, komentar terpopuler, dan pencarian kata kunci.
   - Demo alur pra-proses teks step-by-step: *Cleansing*, *Slang Normalization*, *Stopword Removal*, dan *Stemming Sastrawi*.

3. **⚖️ Perbandingan Evaluasi Model**
   - Analisis komparatif antara **Model Baseline (Tanpa SMOTE)** vs **Model Optimized (Dengan SMOTE)**.
   - Visualisasi Matriks Konfusi (Confusion Matrix) 2x2.
   - Perbandingan metrik evaluasi: *Accuracy*, *Precision*, *Recall*, dan *F1-Score*.

4. **🔮 Demo Prediksi Interaktif**
   - Pengujian klasifikasi teks komentar baru secara *real-time*.
   - Pilihan penggunaan teks asli (direkomendasikan untuk arsitektur BERT) atau teks hasil pra-proses Sastrawi.
   - Fitur **"Muat Contoh Data"** secara acak dari dataset TikTok.
   - Perbandingan hasil prediksi *side-by-side* antara Model Baseline dan Model SMOTE lengkap dengan *confidence score* (%) dan latensi inferensi [CLS] (ms).

---

## 📂 Struktur Repositori

```
Integrasi Aplikasi/
├── Dataset/                             # Dataset komentar TikTok (.csv)
│   ├── tiktok_comments_results (1).csv
│   └── tiktok_comments_results_kompascom2.csv
├── Model/                               # Bobot Model IndoBERT & Classifier
│   ├── Dengan SMOTE/
│   │   ├── indobert_model_deploy/       # Tokenizer & Model IndoBERT fine-tuned (SMOTE)
│   │   ├── mlp_model_stance.pkl         # Classifier MLP (Dengan SMOTE)
│   │   └── model_metadata.pkl
│   └── Tanpa SMOTE/
│       ├── indobert_tokenizer/          # Tokenizer IndoBERT (Tanpa SMOTE)
│       └── mlp_model_stance.pkl         # Classifier MLP (Tanpa SMOTE)
├── code/                                # Komponen template visualisasi HTML
│   ├── beranda_stance_detection.html
│   ├── demo_prediksi_interaktif.html
│   ├── eksplorasi_data_praproses.html
│   └── perbandingan_evaluasi_model.html
├── app.py                               # Skrip utama aplikasi Streamlit
├── requirements.txt                     # Daftar dependensi Python
└── README.md                            # Dokumentasi proyek
```

---

## 🚀 Panduan Instalasi & Penggunaan Lokal

### 1. Prasyarat
- **Python**: Versi `3.10` atau lebih baru
- **Git** (Opsional untuk cloning repositori)

### 2. Kloning / Unduh Repositori
```bash
git clone https://github.com/donnycharles88/Skripsi-ANALISIS-STANCE-DETECTION-KOMENTAR-TIKTOK.git
cd "Integrasi Aplikasi"
```

### 3. Buat Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Instal Dependensi
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Jalankan Aplikasi Streamlit
```bash
streamlit run app.py
```
Aplikasi akan secara otomatis terbuka di browser pada alamat `http://localhost:8501`.

---

## 🛠️ Teknologi & Library yang Digunakan

- **Language & Framework**: Python 3.10+, [Streamlit](https://streamlit.io/)
- **NLP & Deep Learning**: PyTorch, Hugging Face Transformers (`AutoTokenizer`, `BertModel`), PySastrawi
- **Machine Learning**: Scikit-Learn (MLPClassifier, Metrics), Imbalanced-Learn (SMOTE)
- **Data Manipulation & Visualization**: Pandas, NumPy, Plotly, Seaborn, Matplotlib

---

## 🌐 Deploy di Streamlit Cloud

Aplikasi telah berhasil di-deploy dan dapat diakses publik melalui tautan berikut:
👉 **[https://stanceapp.streamlit.app/](https://stanceapp.streamlit.app/)**

---

## 👨‍💻 Tim Pengembang

- **Peneliti / Pengembang**: Tim Kelompok Doni & Adinda
- **Fakultas / Program Studi**: Skripsi Analisis Stance Detection Komentar TikTok

---

© 2024 Tim Kelompok Doni & Adinda. All Rights Reserved.
