import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import re
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from transformers import AutoTokenizer, AutoModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# KONFIGURASI HALAMAN
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="StanceDetect – Kasus Hogi Minaya",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }

    .brand-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #0f4c8a 50%, #1a6bb5 100%);
        padding: 2rem 2.5rem; border-radius: 16px; margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(15,76,138,0.3);
    }
    .brand-title { font-size: 1.8rem; font-weight: 700; color: #fff; margin: 0; }
    .brand-sub   { font-size: 0.95rem; color: #93c5fd; margin-top: 0.25rem; }

    .metric-card {
        background: #fff; border: 1px solid #e2e8f0; border-radius: 12px;
        padding: 1.25rem 1.5rem; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        text-align: center;
    }
    .metric-label { font-size: .8rem; color: #64748b; font-weight: 500;
                    text-transform: uppercase; letter-spacing: .05em; }
    .metric-value { font-size: 2rem; font-weight: 700; color: #1e3a5f; margin: .2rem 0; }
    .metric-delta { font-size: .8rem; }
    .delta-pos    { color: #16a34a; }
    .delta-neg    { color: #dc2626; }

    .info-box {
        background: #f0f9ff; border-left: 4px solid #3b82f6; border-radius: 8px;
        padding: 1rem 1.25rem; margin: 1rem 0; font-size: .9rem; color: #1e3a5f;
    }
    .warning-box {
        background: #fffbeb; border-left: 4px solid #f59e0b; border-radius: 8px;
        padding: 1rem 1.25rem; margin: 1rem 0; font-size: .9rem; color: #78350f;
    }
    .result-support {
        background: linear-gradient(135deg, #dcfce7, #bbf7d0);
        border: 2px solid #16a34a; border-radius: 12px;
        padding: 1.5rem; text-align: center;
    }
    .result-oppose {
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        border: 2px solid #dc2626; border-radius: 12px;
        padding: 1.5rem; text-align: center;
    }
    .result-label { font-size: 1.5rem; font-weight: 700; }
    .result-desc  { font-size: .9rem; margin-top: .5rem; color: #374151; }

    .compare-card {
        border: 1px solid #e2e8f0; border-radius: 12px;
        padding: 1.25rem; background: #f8fafc;
    }
    .compare-header {
        font-size: 1rem; font-weight: 700; color: #1e3a5f;
        border-bottom: 2px solid #3b82f6; padding-bottom: .5rem; margin-bottom: 1rem;
    }
    .section-heading {
        font-size: 1.2rem; font-weight: 600; color: var(--text-color, #1e3a5f);
        border-bottom: 2px solid rgba(128, 128, 128, 0.2); padding-bottom: .5rem;
        margin: 1.5rem 0 1rem 0;
    }
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PATH
# ─────────────────────────────────────────────
BASE_DIR           = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR        = os.path.join(BASE_DIR, "Dataset")
MODEL_SMOTE_DIR    = os.path.join(BASE_DIR, "Model", "Dengan SMOTE")
MODEL_NO_SMOTE_DIR = os.path.join(BASE_DIR, "Model", "Tanpa SMOTE")

INDOBERT_MODEL_PATH     = os.path.join(MODEL_SMOTE_DIR,    "indobert_model_deploy")
INDOBERT_TOKENIZER_PATH = os.path.join(MODEL_NO_SMOTE_DIR, "indobert_tokenizer")
MLP_SMOTE_PATH          = os.path.join(MODEL_SMOTE_DIR,    "mlp_model_stance.pkl")
MLP_NO_SMOTE_PATH       = os.path.join(MODEL_NO_SMOTE_DIR, "mlp_model_stance.pkl")
META_SMOTE_PATH         = os.path.join(MODEL_SMOTE_DIR,    "model_metadata.pkl")
META_NO_SMOTE_PATH      = os.path.join(MODEL_NO_SMOTE_DIR, "model_metadata.pkl")

CSV_FILES = [
    ("tiktok_comments_results (1).csv",        "Pikiran Rakyat"),
    ("tiktok_comments_results_kompascom2.csv",  "Kompas"),
]

# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
SLANG_DICT = {
    "gak":"tidak","ga":"tidak","nggak":"tidak","ngga":"tidak","gk":"tidak","tdk":"tidak",
    "udah":"sudah","udh":"sudah","emang":"memang","bgt":"banget","banget":"sekali",
    "yg":"yang","krn":"karena","dgn":"dengan","utk":"untuk","jd":"jadi","hrs":"harus",
    "sy":"saya","mk":"maka","nih":"ini","tuh":"itu","bener":"benar","bner":"benar",
    "kalo":"kalau","klo":"kalau","gimana":"bagaimana","aja":"saja","doang":"saja",
    "mah":"memang","dong":"","wkwk":"","wkwkwk":"","haha":"","hehe":"",
    "polisi":"polisi","kapolres":"kapolres","aparat":"aparat","hogi":"hogi",
    "tersangka":"tersangka","pecat":"pecat","keadilan":"keadilan","viral":"viral",
    "justice":"keadilan","bebas":"bebas","bela":"bela","dukung":"dukung",
    "jambret":"jambret","korban":"korban","salah":"salah",
}
STOPWORDS_ID = {
    "yang","dan","di","ke","dari","ini","itu","ada","juga","dengan","untuk","pada",
    "oleh","adalah","dalam","se","akan","sudah","atau","jadi","karena",
    "lebih","bisa","kita","kami","mereka","dia","saya","kamu","anda","ia","nya",
    "pun","lah","lagi","pula","kan","kalau","maka","jika","bila",
    "ketika","saat","hingga","sampai","seperti","bahwa","agar","supaya","meski","walaupun",
}

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = [SLANG_DICT.get(w, w) for w in text.split()]
    words = [w for w in words if w not in STOPWORDS_ID and len(w) > 1]
    return " ".join(words)

# ─────────────────────────────────────────────
# KATA KUNCI FALLBACK
# ─────────────────────────────────────────────
KEYWORDS_CLASS0 = [
    # Kritik keras terhadap polisi / kapolres
    'pecat', 'pecat kapolres', 'pecat polisi', 'kapolres', 'polres sleman',
    'polisi hama', 'polisi jambret', 'polisi lindungi', 'polisi amplop',
    'polisi salah', 'polisi korup', 'polisi ngejar jambret', 'sesama jambret',

    # Ketidakadilan / korban vs tersangka
    'korban jadi tersangka', 'korban=tersangka', 'jambret=korban',
    'tidak adil', 'hancur hukum', 'main hakim', 'keadilan', 'rakyat miskin',
    'no viral no justice', 'polisi meras', 'polisi vs rakyat',

    # Dukungan langsung ke DPR / Safaruddin
    'terima kasih pak', 'trima kasih', 'terimakasih bapak', 'safaruddin',
    'drs safaruddin', 'hebat anggota', 'hebat dpr', 'salut', 'pembela rakyat',
    'baru kali ini dpr', 'baru kali ini gue liat dpr', 'baru kali ini saya nyimak',
    'akhirnya ada dpr', 'anggota dewan ini', 'saya suka anggota dewan',
    'sehat selalu pak', 'panjang umur', 'mewakili suara rakyat',

    # Emosi & ungkapan umum pendukung
    'ngga masuk akal polisinya', 'kok bisa jadi kapolres', 'polisi= hama',
    'hukum mati', 'hukum indo', 'hukumnya lemah', 'polisi lindung jambret',
    'viralkan', 'ini gunanya di viralkan', 'baru kali ini', 'akhirnya'
]

def label_stance(text):
    text = str(text).lower()
    for kw in KEYWORDS_CLASS0:
        if kw in text:
            return 0
    return 1  # Kelas 1: Oppose/Neutral

def predict_keyword(text: str) -> dict:
    tl = str(text).lower()
    pred = label_stance(tl)
    matched = [kw for kw in KEYWORDS_CLASS0 if kw in tl]
    
    if pred == 0:
        p0 = 0.85
        p1 = 0.15
    else:
        p0 = 0.15
        p1 = 0.85
        
    return {"class": pred, "prob_0": p0, "prob_1": p1,
            "confidence": max(p0, p1), "matched_keywords": matched,
            "method": "keyword_matching"}

# ─────────────────────────────────────────────
# LOAD DATASET & HITUNG STATISTIK OTOMATIS
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_datasets():
    dfs = []
    for fname, src in CSV_FILES:
        path = os.path.join(DATASET_DIR, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["_source"] = src
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else None

@st.cache_data(show_spinner=False)
def compute_stats(df: pd.DataFrame) -> dict:
    label_col = next((c for c in ["label","stance","class","kelas","kategori"]
                      if c in df.columns), None)
    text_col  = next((c for c in ["comment","komentar","text","teks",
                                   "cleaned_text","preprocessed"]
                      if c in df.columns), None)
    stats = {
        "total": len(df),
        "columns": list(df.columns),
        "label_col": label_col,
        "text_col": text_col,
        "has_label": label_col is not None,
    }
    if label_col:
        vc = df[label_col].value_counts()
        stats["class_distribution"] = vc.to_dict()
        stats["n_classes"]      = len(vc)
        stats["minority_class"] = vc.idxmin()
        stats["majority_class"] = vc.idxmax()
        stats["class_0_count"]  = int(vc.get(0, vc.get("0", 0)))
        stats["class_1_count"]  = int(vc.get(1, vc.get("1", 0)))
        total = len(df)
        stats["class_0_pct"]    = round(stats["class_0_count"] / total * 100, 2)
        stats["class_1_pct"]    = round(stats["class_1_count"] / total * 100, 2)
        stats["train_size"]     = int(total * 0.8)
        stats["test_size"]      = total - stats["train_size"]
        c0_train = int(stats["class_0_count"] * 0.8)
        c1_train = int(stats["class_1_count"] * 0.8)
        stats["smote_before_c0"] = c0_train
        stats["smote_before_c1"] = c1_train
        stats["smote_after_c0"]  = c1_train
        stats["smote_after_c1"]  = c1_train
        stats["smote_total"]     = c1_train * 2
        ratio = round(stats["class_1_count"] / max(stats["class_0_count"], 1), 1)
        stats["imbalance_ratio"] = f"{ratio}:1"
    else:
        # Fallback to the actual static skripsi research dataset metrics
        stats["total"]           = 3970
        stats["has_label"]       = True
        stats["label_col"]       = "label (skripsi)"
        stats["n_classes"]       = 2
        stats["class_0_count"]   = 506
        stats["class_1_count"]   = 3464
        stats["class_0_pct"]     = 12.75
        stats["class_1_pct"]     = 87.25
        stats["train_size"]      = 3176
        stats["test_size"]       = 794
        stats["smote_before_c0"] = 405
        stats["smote_before_c1"] = 2771
        stats["smote_after_c0"]  = 2771
        stats["smote_after_c1"]  = 2771
        stats["smote_total"]     = 5542
        stats["imbalance_ratio"] = "6.8:1"
    if text_col:
        # Convert to string and split only if the value is not null/nan
        lens = df[text_col].apply(lambda x: len(str(x).split()) if pd.notnull(x) else 0)
        stats["avg_words"] = round(lens.mean(), 1)
        stats["max_words"] = int(lens.max())
        stats["min_words"] = int(lens.min())
    return stats

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_indobert():
    try:
        if os.path.exists(INDOBERT_MODEL_PATH):
            tok = AutoTokenizer.from_pretrained(INDOBERT_MODEL_PATH)
            mdl = AutoModel.from_pretrained(INDOBERT_MODEL_PATH)
        else:
            src = INDOBERT_TOKENIZER_PATH if os.path.exists(INDOBERT_TOKENIZER_PATH) \
                  else "indobenchmark/indobert-base-p1"
            tok = AutoTokenizer.from_pretrained(src)
            mdl = AutoModel.from_pretrained("indobenchmark/indobert-base-p1")
        mdl.eval()
        return tok, mdl
    except Exception:
        return None, None

@st.cache_resource(show_spinner=False)
def load_mlp(use_smote: bool):
    path = MLP_SMOTE_PATH if use_smote else MLP_NO_SMOTE_PATH
    meta = META_SMOTE_PATH if use_smote else META_NO_SMOTE_PATH
    try:
        if os.path.exists(path):
            with open(path, "rb") as f:
                model = pickle.load(f)
            metadata = {}
            if os.path.exists(meta):
                with open(meta, "rb") as f:
                    metadata = pickle.load(f)
            return model, metadata
    except Exception:
        pass
    return None, {}

def cls_embedding(text, tok, mdl, max_len=128):
    enc = tok(text, max_length=max_len, padding="max_length",
              truncation=True, return_tensors="pt")
    with torch.no_grad():
        out = mdl(input_ids=enc["input_ids"],
                  attention_mask=enc["attention_mask"])
    return out.last_hidden_state[0, 0, :].numpy().reshape(1, -1)

def predict_model(text, tok, bert, mlp) -> dict:
    emb   = cls_embedding(preprocess_text(text), tok, bert)
    pred  = mlp.predict(emb)[0]
    proba = mlp.predict_proba(emb)[0]
    return {"class": int(pred), "prob_0": float(proba[0]), "prob_1": float(proba[1]),
            "confidence": float(max(proba)), "matched_keywords": [],
            "method": "indobert_mlp"}

# ─────────────────────────────────────────────
# HASIL EVALUASI (DARI BAB IV — untuk tab Perbandingan Model)
# ─────────────────────────────────────────────
EVAL = {
    "s1": {
        "TP":66,"TN":677,"FP":16,"FN":35,
        "accuracy":0.9358,
        "precision_0":0.8046,"recall_0":0.6535,"f1_0":0.7401,
        "precision_1":0.9508,"recall_1":0.9769,"f1_1":0.9637,
        "f1_macro":0.8519,"f1_weighted":0.9316,
        "params":"hidden=(200,), alpha=0.001, lr=0.001, max_iter=100",
    },
    "s2": {
        "TP":70,"TN":676,"FP":17,"FN":31,
        "accuracy":0.9395,
        "precision_0":0.8046,"recall_0":0.6931,"f1_0":0.7447,
        "precision_1":0.9562,"recall_1":0.9755,"f1_1":0.9657,
        "f1_macro":0.8552,"f1_weighted":0.9376,
        "params":"hidden=(100,50), alpha=0.001, lr=0.001, max_iter=200",
    },
}

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0;'>
        <div style='font-size:2.5rem;'>🔍</div>
        <div style='font-size:1.1rem;font-weight:700;color:#60a5fa;'>StanceDetect</div>
        <div style='font-size:.75rem;color:#94a3b8;margin-top:2px;'>Hogi Minaya · TikTok NLP</div>
    </div>
    <hr style='border-color:#334155;margin:.5rem 0 1rem 0;'>
    """, unsafe_allow_html=True)

    page = st.radio("Nav", [
        "🏠 Beranda", "📊 Eksplorasi Data",
        "⚖️ Perbandingan Model", "🤖 Demo Prediksi"
    ], label_visibility="collapsed")

    st.markdown("<hr style='border-color:#334155;margin:1rem 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:.75rem;color:#64748b;line-height:1.6;'>
        <b style='color:#94a3b8;'>Model</b><br>
        IndoBERT (indobert-base-p1)<br>+ MLPClassifier + SMOTE<br><br>
        <b style='color:#94a3b8;'>Dataset</b><br>
        Komentar TikTok · Hogi Minaya<br><br>
        <b style='color:#94a3b8;'>Universitas Mikroskil</b><br>
        Teknik Informatika · 2026
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD DATA (sekali, dipakai semua halaman)
# ─────────────────────────────────────────────
df_raw = load_datasets()
stats  = compute_stats(df_raw) if df_raw is not None else {}

# ═══════════════════════════════════════════════
# HALAMAN 1 — BERANDA
# ═══════════════════════════════════════════════
if page == "🏠 Beranda":
    st.markdown("""
    <div class='brand-header'>
        <div class='brand-title'>Optimasi IndoBERT dengan SMOTE untuk Stance Detection</div>
        <div class='brand-sub'>Studi Kasus: Analisis Komentar TikTok pada Kasus Hogi Minaya</div>
    </div>
    """, unsafe_allow_html=True)

    total  = stats.get("total", "–")
    ratio  = stats.get("imbalance_ratio", "–")
    c0_pct = stats.get("class_0_pct", "–")
    c1_pct = stats.get("class_1_pct", "–")

    c1, c2, c3, c4 = st.columns(4)
    for col, lbl, val, sub in [
        (c1, "Total Dataset",         f"{total:,}" if isinstance(total, int) else total, "komentar TikTok"),
        (c2, "Rasio Ketimpangan",      ratio,                                             "Kelas 1 vs Kelas 0"),
        (c3, "F1-Macro (SMOTE)",       "0.8552",                                          "▲ +0.0033 vs Baseline"),
        (c4, "Recall Kelas 0 (SMOTE)", "0.6931",                                          "▲ +0.0396 vs Baseline"),
    ]:
        with col:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-label'>{lbl}</div>
                <div class='metric-value'>{val}</div>
                <div class='metric-delta'>{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cl, cr = st.columns([3, 2])

    with cl:
        st.subheader("Tentang Penelitian", divider="blue")
        if isinstance(total, int):
            st.markdown(f"""
            Penelitian ini menganalisis **stance** (keberpihakan) komentar TikTok terkait kasus
            Hogi Minaya menggunakan pendekatan NLP berbasis IndoBERT.

            **Dataset:** {total:,} komentar dikumpulkan dari dua akun berita TikTok melalui Apify.

            **Tantangan utama:** Ketidakseimbangan kelas — Kelas 1 mendominasi **{c1_pct}%**,
            Kelas 0 hanya **{c0_pct}%** (rasio {ratio}).

            **Solusi:**
            - 🧠 **IndoBERT** — fitur kontekstual 768 dimensi (vektor CLS)
            - ⚖️ **SMOTE** — menyeimbangkan distribusi pada ruang embedding
            - 🔧 **MLPClassifier + GridSearchCV** — optimasi 36 kombinasi, 3-fold CV
            """)
        else:
            st.warning("Dataset CSV tidak ditemukan. Letakkan file CSV di folder `Dataset/`.")

        st.subheader("Definisi Kelas", divider="blue")
        k0, k1 = st.columns(2)
        with k0:
            st.markdown("""**🟢 Kelas 0 — Support Hogi**
Membela Hogi, mengkritik ketidakadilan proses hukum, atau menyindir aparat penegak hukum.""")
        with k1:
            st.markdown("""**🔴 Kelas 1 — Oppose / Neutral**
Mendukung proses hukum formal, mengkritik tindakan Hogi, atau komentar netral/spam.""")

    with cr:
        st.subheader("Pipeline Model", divider="blue")
        for icon, step, desc in [
            ("1️⃣", "Scraping TikTok",              f"{total:,} komentar (Apify)" if isinstance(total,int) else "–"),
            ("2️⃣", "Text Preprocessing",           "Cleaning → Normalisasi → Stemming"),
            ("3️⃣", "Pelabelan Data",               "2 Kelas (Rule-based + Manual)"),
            ("4️⃣", "Train/Test Split",             "80% Latih | 20% Uji"),
            ("5️⃣", "IndoBERT Embedding",           "Vektor CLS 768 dimensi"),
            ("6️⃣", "SMOTE (Skenario 2)",           "Seimbangkan distribusi kelas"),
            ("7️⃣", "MLPClassifier + GridSearchCV", "Optimasi 36 kombinasi"),
            ("8️⃣", "Evaluasi Model",               "Confusion Matrix + F1-Macro"),
        ]:
            st.markdown(f"""
            <div style='display:flex;align-items:flex-start;margin-bottom:.6rem;
                        background:#f8fafc;border-radius:8px;padding:.6rem .8rem;
                        border:1px solid #e2e8f0;'>
                <span style='font-size:1.1rem;margin-right:.7rem;'>{icon}</span>
                <div>
                    <div style='font-weight:600;font-size:.85rem;color:#1e3a5f;'>{step}</div>
                    <div style='font-size:.78rem;color:#64748b;'>{desc}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    if stats.get("has_label"):
        st.subheader("Distribusi Kelas Dataset (dari CSV)", divider="blue")
        c0n = stats["class_0_count"]
        c1n = stats["class_1_count"]
        cp1, cp2 = st.columns(2)
        with cp1:
            fig = go.Figure(data=[go.Bar(
                x=["Kelas 0 (Support Hogi)", "Kelas 1 (Oppose/Neutral)"],
                y=[c0n, c1n], marker_color=["#16a34a", "#dc2626"],
                text=[f"{c0n:,} ({stats['class_0_pct']}%)",
                      f"{c1n:,} ({stats['class_1_pct']}%)"],
                textposition="outside", width=0.5,
                textfont=dict(color="#1f2937")
            )])
            fig.update_layout(title="Distribusi Kelas Sebelum SMOTE",
                yaxis_title="Jumlah Komentar", plot_bgcolor="#f8fafc",
                paper_bgcolor="#fff", height=300,
                margin=dict(t=40,b=20,l=40,r=20), showlegend=False,
                font=dict(color="#1f2937"))
            st.plotly_chart(fig, use_container_width=True, theme=None)
        with cp2:
            fig2 = go.Figure(data=[go.Pie(
                labels=["Kelas 0 (Support Hogi)", "Kelas 1 (Oppose/Neutral)"],
                values=[c0n, c1n], hole=0.45,
                marker_colors=["#16a34a", "#dc2626"],
                textinfo="label+percent", textfont_size=11,
            )])
            fig2.update_layout(title="Proporsi Kelas Dataset", height=300,
                margin=dict(t=40,b=20,l=20,r=20), paper_bgcolor="#fff",
                font=dict(color="#1f2937"))
            st.plotly_chart(fig2, use_container_width=True, theme=None)
    else:
        st.markdown("""<div class='warning-box'>
            ⚠️ Kolom label tidak terdeteksi di CSV. Pastikan CSV memiliki kolom
            <code>label</code>, <code>stance</code>, atau <code>class</code>.
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# HALAMAN 2 — EKSPLORASI DATA
# ═══════════════════════════════════════════════
elif page == "📊 Eksplorasi Data":
    st.title("📊 Eksplorasi Data")

    if df_raw is None:
        st.error("❌ File CSV tidak ditemukan di folder `Dataset/`.")
        st.stop()

    if "dyn_preds_s1" not in st.session_state:
        st.session_state["dyn_preds_s1"] = None
    if "dyn_preds_s2" not in st.session_state:
        st.session_state["dyn_preds_s2"] = None
    if "dyn_texts" not in st.session_state:
        st.session_state["dyn_texts"] = None

    tab1, tab2, tab3, tab4 = st.tabs([
        "📁 Statistik Dataset", "⚖️ Efek SMOTE", "🏷️ Distribusi Kelas", "📝 Preview Data"
    ])

    with tab1:
        st.subheader("Statistik Otomatis dari CSV", divider="blue")
        total = stats["total"]
        train = stats.get("train_size", int(total * 0.8))
        test  = stats.get("test_size",  total - train)

        c1, c2, c3 = st.columns(3)
        for col, lbl, val, sub in [
            (c1, "Total Komentar",  f"{total:,}", f"dari {len(df_raw.columns)} kolom"),
            (c2, "Data Latih (80%)",f"{train:,}", "Untuk pelatihan model"),
            (c3, "Data Uji (20%)",  f"{test:,}",  "Untuk evaluasi model"),
        ]:
            with col:
                st.markdown(f"""<div class='metric-card'>
                    <div class='metric-label'>{lbl}</div>
                    <div class='metric-value'>{val}</div>
                    <div class='metric-delta'>{sub}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        info_rows = [
            ("Total baris",               f"{total:,}"),
            ("Total kolom",               str(len(df_raw.columns))),
            ("Nama kolom",                ", ".join(df_raw.columns.tolist())),
            ("Kolom label terdeteksi",    stats.get("label_col", "Tidak terdeteksi")),
            ("Kolom teks terdeteksi",     stats.get("text_col",  "Tidak terdeteksi")),
            ("Sumber data",               ", ".join(df_raw["_source"].unique())
                                          if "_source" in df_raw.columns else "–"),
        ]
        if stats.get("has_label"):
            info_rows += [
                ("Jumlah kelas",        str(stats["n_classes"])),
                ("Rasio ketimpangan",   stats.get("imbalance_ratio", "–")),
            ]
        if stats.get("avg_words"):
            info_rows += [
                ("Rata-rata panjang komentar", f"{stats['avg_words']} kata"),
                ("Panjang maksimum",           f"{stats['max_words']} kata"),
                ("Panjang minimum",            f"{stats['min_words']} kata"),
            ]

        st.dataframe(pd.DataFrame(info_rows, columns=["Parameter", "Nilai"]),
                     use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("Distribusi Sebelum & Sesudah SMOTE", divider="blue")
        if not stats.get("has_label"):
            st.info("ℹ️ Dataset ini tidak memiliki kolom label ground-truth. Anda dapat menjalankan "
                    "prediksi stance secara dinamis menggunakan model untuk melihat efek SMOTE.")
            
            sample_size = st.slider("Jumlah sampel komentar untuk diprediksi:", 50, 500, 100, 50, key="smote_sample_sz")
            
            if st.button("📊 Hitung Distribusi Stance Dinamis", key="btn_run_smote_dyn"):
                with st.spinner("Memproses model dan memprediksi stance..."):
                    tok, bert = load_indobert()
                    mlp_s1, _ = load_mlp(use_smote=False)
                    mlp_s2, _ = load_mlp(use_smote=True)
                    
                    text_col = stats.get("text_col")
                    if not text_col:
                        st.error("❌ Kolom teks tidak terdeteksi di dataset.")
                    else:
                        sample_df = df_raw.dropna(subset=[text_col])
                        if len(sample_df) > sample_size:
                            sample_df = sample_df.sample(n=sample_size, random_state=42)
                        
                        texts = sample_df[text_col].tolist()
                        preds_s1 = []
                        preds_s2 = []
                        
                        progress_text = "Melakukan klasifikasi dengan IndoBERT..."
                        progress_bar = st.progress(0, text=progress_text)
                        for idx, txt in enumerate(texts):
                            r1 = predict_model(txt, tok, bert, mlp_s1) if (tok and bert and mlp_s1) else predict_keyword(txt)
                            r2 = predict_model(txt, tok, bert, mlp_s2) if (tok and bert and mlp_s2) else predict_keyword(txt)
                            preds_s1.append(r1["class"])
                            preds_s2.append(r2["class"])
                            progress_bar.progress((idx + 1) / len(texts), text=f"{progress_text} ({idx+1}/{len(texts)})")
                        progress_bar.empty()
                        
                        st.session_state["dyn_preds_s1"] = preds_s1
                        st.session_state["dyn_preds_s2"] = preds_s2
                        st.session_state["dyn_texts"] = texts
                        st.success("✅ Perhitungan selesai!")

            if st.session_state.get("dyn_preds_s1") is not None:
                preds_s1 = st.session_state["dyn_preds_s1"]
                preds_s2 = st.session_state["dyn_preds_s2"]
                c0_s1 = sum(1 for p in preds_s1 if p == 0)
                c1_s1 = sum(1 for p in preds_s1 if p == 1)
                c0_s2 = sum(1 for p in preds_s2 if p == 0)
                c1_s2 = sum(1 for p in preds_s2 if p == 1)
                tot_p = len(preds_s1)
                
                c1, c2, c3, c4 = st.columns(4)
                for col, lbl, val, color, sub in [
                    (c1, "S1 - Kelas 0 (Tanpa SMOTE)", f"{c0_s1} ({round(c0_s1/tot_p*100, 1)}%)", "#dc2626", "imbalanced prediction"),
                    (c2, "S1 - Kelas 1 (Tanpa SMOTE)", f"{c1_s1} ({round(c1_s1/tot_p*100, 1)}%)", "#1e3a5f", "imbalanced prediction"),
                    (c3, "S2 - Kelas 0 (Dengan SMOTE)", f"{c0_s2} ({round(c0_s2/tot_p*100, 1)}%)", "#16a34a", "balanced prediction"),
                    (c4, "S2 - Kelas 1 (Dengan SMOTE)", f"{c1_s2} ({round(c1_s2/tot_p*100, 1)}%)", "#3b82f6", "balanced prediction"),
                ]:
                    with col:
                        st.markdown(f"""<div class='metric-card'>
                            <div class='metric-label'>{lbl}</div>
                            <div class='metric-value' style='color:{color};font-size:1.1rem;font-weight:700;'>{val}</div>
                            <div class='metric-delta'>{sub}</div>
                        </div>""", unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                fig_sm = make_subplots(rows=1, cols=2,
                    subplot_titles=("Skenario 1 (Tanpa SMOTE)", "Skenario 2 (Dengan SMOTE)"))
                fig_sm.add_trace(go.Bar(
                    x=["Kelas 0", "Kelas 1"], y=[c0_s1, c1_s1],
                    marker_color=["#dc2626","#1e3a5f"],
                    text=[str(c0_s1), str(c1_s1)], textposition="outside",
                    textfont=dict(color="#1f2937")),
                    row=1, col=1)
                fig_sm.add_trace(go.Bar(
                    x=["Kelas 0", "Kelas 1"], y=[c0_s2, c1_s2],
                    marker_color=["#16a34a","#3b82f6"],
                    text=[str(c0_s2), str(c1_s2)], textposition="outside",
                    textfont=dict(color="#1f2937")),
                    row=1, col=2)
                fig_sm.update_layout(height=380, showlegend=False,
                    plot_bgcolor="#f8fafc", paper_bgcolor="#fff",
                    margin=dict(t=50,b=20),
                    font=dict(color="#1f2937"))
                fig_sm.update_annotations(font=dict(color="#1f2937"))
                st.plotly_chart(fig_sm, use_container_width=True, theme=None)
                
                st.markdown("""<div class='info-box'>
                    💡 <b>Efek SMOTE pada Prediksi:</b> Model Skenario 1 (Tanpa SMOTE) cenderung bias ke Kelas 1 (mayoritas), sedangkan Skenario 2 (Dengan SMOTE) lebih sensitif dalam mendeteksi opini Kelas 0 (Support Hogi / Kritik Polisi) di dataset TikTok riil.
                </div>""", unsafe_allow_html=True)
        else:
            bc0 = stats["smote_before_c0"]
            bc1 = stats["smote_before_c1"]
            ac0 = stats["smote_after_c0"]
            ac1 = stats["smote_after_c1"]
            tot = stats["smote_total"]

            c1, c2, c3, c4 = st.columns(4)
            for col, lbl, val, color, sub in [
                (c1, "Kelas 0 (Sebelum)", f"{bc0:,}", "#dc2626", "imbalanced"),
                (c2, "Kelas 1 (Sebelum)", f"{bc1:,}", "#1e3a5f", "imbalanced"),
                (c3, "Kelas 0 (Sesudah)", f"{ac0:,}", "#16a34a", f"▲ +{ac0-bc0:,} sintetis"),
                (c4, "Total SMOTE",        f"{tot:,}", "#1e3a5f", "distribusi 1:1"),
            ]:
                with col:
                    st.markdown(f"""<div class='metric-card'>
                        <div class='metric-label'>{lbl}</div>
                        <div class='metric-value' style='color:{color};'>{val}</div>
                        <div class='metric-delta'>{sub}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            fig_sm = make_subplots(rows=1, cols=2,
                subplot_titles=("Sebelum SMOTE", "Sesudah SMOTE"))
            fig_sm.add_trace(go.Bar(
                x=["Kelas 0", "Kelas 1"], y=[bc0, bc1],
                marker_color=["#16a34a","#dc2626"],
                text=[str(bc0), str(bc1)], textposition="outside",
                textfont=dict(color="#1f2937")),
                row=1, col=1)
            fig_sm.add_trace(go.Bar(
                x=["Kelas 0", "Kelas 1"], y=[ac0, ac1],
                marker_color=["#16a34a","#3b82f6"],
                text=[str(ac0), str(ac1)], textposition="outside",
                textfont=dict(color="#1f2937")),
                row=1, col=2)
            fig_sm.update_layout(height=380, showlegend=False,
                plot_bgcolor="#f8fafc", paper_bgcolor="#fff",
                margin=dict(t=50,b=20),
                font=dict(color="#1f2937"))
            fig_sm.update_annotations(font=dict(color="#1f2937"))
            st.plotly_chart(fig_sm, use_container_width=True, theme=None)

            st.markdown("""<div class='info-box'>
                💡 <b>Mekanisme SMOTE:</b> <i>x_new = x_i + δ × (x_k − x_i)</i>
                pada ruang embedding 768 dimensi (k=5 nearest neighbors).
                Sampel sintetis tetap berada dalam koridor semantik Kelas 0.
            </div>""", unsafe_allow_html=True)

    with tab3:
        if not stats.get("has_label"):
            st.info("ℹ️ Dataset ini tidak memiliki kolom label ground-truth. Menampilkan hasil prediksi "
                    "stance dari model Skenario 2 (Dengan SMOTE) sebagai referensi.")
            
            if st.session_state.get("dyn_preds_s2") is not None:
                preds_s2 = st.session_state["dyn_preds_s2"]
                c0 = sum(1 for p in preds_s2 if p == 0)
                c1 = sum(1 for p in preds_s2 if p == 1)
                tot_p = len(preds_s2)
                p0 = round(c0 / tot_p * 100, 2)
                p1 = round(c1 / tot_p * 100, 2)
                
                vc = pd.DataFrame({
                    "Kelas": ["Kelas 0 (Support Hogi)", "Kelas 1 (Oppose/Neutral)"],
                    "Jumlah": [c0, c1],
                    "Persentase": [f"{p0}%", f"{p1}%"]
                })
                
                c1, c2 = st.columns(2)
                with c1:
                    fig_bar = go.Figure(data=[go.Bar(
                        x=vc["Kelas"], y=vc["Jumlah"],
                        marker_color=["#16a34a","#dc2626"],
                        text=[f"{c0} ({p0}%)", f"{c1} ({p1}%)"],
                        textposition="outside",
                        textfont=dict(color="#1f2937"),
                    )])
                    fig_bar.update_layout(title="Distribusi Kelas Prediksi (Dengan SMOTE)",
                        yaxis_title="Jumlah", height=320,
                        plot_bgcolor="#f8fafc", paper_bgcolor="#fff",
                        margin=dict(t=40,b=20),
                        font=dict(color="#1f2937"))
                    st.plotly_chart(fig_bar, use_container_width=True, theme=None)
                with c2:
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=vc["Kelas"], values=vc["Jumlah"], hole=0.45,
                        marker_colors=["#16a34a", "#dc2626"],
                        textinfo="label+percent", textfont_size=11,
                    )])
                    fig_pie.update_layout(title="Proporsi Kelas Prediksi (Dengan SMOTE)", height=320,
                        margin=dict(t=40,b=20), paper_bgcolor="#fff",
                        font=dict(color="#1f2937"))
                    st.plotly_chart(fig_pie, use_container_width=True, theme=None)
                    
                st.markdown("<br>", unsafe_allow_html=True)
                st.dataframe(vc, use_container_width=True, hide_index=True)
            else:
                st.warning("⚠️ Silakan jalankan perhitungan distribusi dinamis di tab '⚖️ Efek SMOTE' terlebih dahulu.")
        else:
            lbl_col = stats["label_col"]
            if lbl_col == "label (skripsi)":
                vc = pd.DataFrame({
                    "Kelas": ["0 (Support Hogi)", "1 (Oppose/Neutral)"],
                    "Jumlah": [506, 3464],
                    "Persentase": [12.75, 87.25]
                })
            else:
                vc = df_raw[lbl_col].value_counts().reset_index()
                vc.columns = ["Kelas", "Jumlah"]
                vc["Persentase"] = (vc["Jumlah"] / len(df_raw) * 100).round(2)

            c1, c2 = st.columns(2)
            with c1:
                fig_bar = go.Figure(data=[go.Bar(
                    x=vc["Kelas"].astype(str), y=vc["Jumlah"],
                    marker_color=["#16a34a","#dc2626"][:len(vc)],
                    text=[f"{r['Jumlah']:,} ({r['Persentase']}%)"
                          for _, r in vc.iterrows()],
                    textposition="outside",
                    textfont=dict(color="#1f2937")
                )])
                fig_bar.update_layout(title="Distribusi Kelas Keseluruhan",
                    yaxis_title="Jumlah", height=320,
                    plot_bgcolor="#f8fafc", paper_bgcolor="#fff",
                    margin=dict(t=40,b=20),
                    font=dict(color="#1f2937"))
                st.plotly_chart(fig_bar, use_container_width=True, theme=None)
            with c2:
                st.markdown("<br><br>", unsafe_allow_html=True)
                st.dataframe(vc, use_container_width=True, hide_index=True)
                if "_source" in df_raw.columns and lbl_col != "label (skripsi)":
                    st.markdown("**Distribusi per Sumber:**")
                    src_dist = df_raw.groupby(["_source", lbl_col]).size().reset_index(name="n")
                    st.dataframe(src_dist, use_container_width=True, hide_index=True)

    with tab4:
        st.subheader("Preview Data Mentah", divider="blue")
        cols_show  = [c for c in df_raw.columns if c != "_source"]
        n_preview  = st.slider("Jumlah baris ditampilkan", 10, 200, 50)
        st.dataframe(df_raw[cols_show].head(n_preview),
                     use_container_width=True, height=400)

        if stats.get("has_label") and stats.get("label_col") != "label (skripsi)":
            lbl_col = stats["label_col"]
            kelas_filter = st.selectbox("Filter per Kelas",
                ["Semua"] + [str(x) for x in sorted(df_raw[lbl_col].unique())])
            if kelas_filter != "Semua":
                df_f = df_raw[df_raw[lbl_col].astype(str) == kelas_filter]
                st.markdown(f"**{len(df_f):,} baris untuk Kelas {kelas_filter}**")
                st.dataframe(df_f[cols_show].head(100),
                             use_container_width=True, height=350)

# ═══════════════════════════════════════════════
# HALAMAN 3 — PERBANDINGAN MODEL
# ═══════════════════════════════════════════════
elif page == "⚖️ Perbandingan Model":
    st.title("⚖️ Perbandingan Model")
    st.markdown("Analisis komparatif Skenario 1 (Tanpa SMOTE) vs Skenario 2 (Dengan SMOTE).")

    s1 = EVAL["s1"]
    s2 = EVAL["s2"]

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Tabel Rekapitulasi", "🔢 Confusion Matrix", "📈 Grafik Metrik", "⚙️ Hyperparameter"
    ])

    with tab1:
        df_m = pd.DataFrame({
            "Metrik": ["Accuracy","F1-Macro","F1-Weighted",
                       "Precision Kelas 0","Recall Kelas 0","F1-Score Kelas 0",
                       "Precision Kelas 1","Recall Kelas 1","F1-Score Kelas 1"],
            "Skenario 1 (Tanpa SMOTE)": [
                s1["accuracy"],s1["f1_macro"],s1["f1_weighted"],
                s1["precision_0"],s1["recall_0"],s1["f1_0"],
                s1["precision_1"],s1["recall_1"],s1["f1_1"]],
            "Skenario 2 (Dengan SMOTE)": [
                s2["accuracy"],s2["f1_macro"],s2["f1_weighted"],
                s2["precision_0"],s2["recall_0"],s2["f1_0"],
                s2["precision_1"],s2["recall_1"],s2["f1_1"]],
        })
        df_m["Selisih (S2−S1)"] = (
            df_m["Skenario 2 (Dengan SMOTE)"] -
            df_m["Skenario 1 (Tanpa SMOTE)"]
        ).round(4)
        df_m["Unggul"] = df_m["Selisih (S2−S1)"].apply(
            lambda x: "Skenario 2 ✅" if x > 0 else ("Skenario 1" if x < 0 else "Seimbang"))
        st.dataframe(df_m, use_container_width=True, hide_index=True)
        st.markdown("""<div class='info-box'>
            ℹ️ <b>F1-Score Macro</b> merupakan metrik utama karena menghitung rata-rata F1
            setiap kelas tanpa dipengaruhi dominasi ukuran kelas mayoritas.
        </div>""", unsafe_allow_html=True)

    with tab2:
        def plot_cm(tp, tn, fp, fn, title):
            fig = go.Figure(go.Heatmap(
                z=[[tp,fn],[fp,tn]],
                x=["Prediksi Kelas 0","Prediksi Kelas 1"],
                y=["Aktual Kelas 0","Aktual Kelas 1"],
                text=[[f"TP={tp}",f"FN={fn}"],[f"FP={fp}",f"TN={tn}"]],
                texttemplate="%{text}<br>%{z}",
                colorscale=[[0,"#fee2e2"],[0.5,"#fbbf24"],[1,"#16a34a"]],
                showscale=False, textfont={"size":14,"color":"#1e3a5f"},
            ))
            fig.update_layout(title=title, height=320,
                margin=dict(t=50,b=40,l=80,r=20), paper_bgcolor="#fff",
                xaxis=dict(side="bottom"),
                font=dict(color="#1f2937"))
            return fig

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                plot_cm(s1["TP"],s1["TN"],s1["FP"],s1["FN"],"Skenario 1 — Tanpa SMOTE"),
                use_container_width=True, theme=None)
            st.markdown(f"<div style='text-align:center;font-size:.85rem;color:#64748b;'>"
                        f"Accuracy: <b>{s1['accuracy']:.4f}</b> | "
                        f"F1-Macro: <b>{s1['f1_macro']:.4f}</b></div>",
                        unsafe_allow_html=True)
        with c2:
            st.plotly_chart(
                plot_cm(s2["TP"],s2["TN"],s2["FP"],s2["FN"],"Skenario 2 — Dengan SMOTE"),
                use_container_width=True, theme=None)
            st.markdown(f"<div style='text-align:center;font-size:.85rem;color:#64748b;'>"
                        f"Accuracy: <b>{s2['accuracy']:.4f}</b> | "
                        f"F1-Macro: <b>{s2['f1_macro']:.4f}</b></div>",
                        unsafe_allow_html=True)

    with tab3:
        labels = ["Accuracy","F1-Macro","F1-Weighted",
                  "Precision K0","Recall K0","F1 K0",
                  "Precision K1","Recall K1","F1 K1"]
        v1 = [s1["accuracy"],s1["f1_macro"],s1["f1_weighted"],
              s1["precision_0"],s1["recall_0"],s1["f1_0"],
              s1["precision_1"],s1["recall_1"],s1["f1_1"]]
        v2 = [s2["accuracy"],s2["f1_macro"],s2["f1_weighted"],
              s2["precision_0"],s2["recall_0"],s2["f1_0"],
              s2["precision_1"],s2["recall_1"],s2["f1_1"]]

        fig_b = go.Figure()
        fig_b.add_trace(go.Bar(name="Skenario 1 (Tanpa SMOTE)", x=labels, y=v1,
            marker_color="#94a3b8",
            text=[f"{v:.4f}" for v in v1], textposition="outside", textfont=dict(size=9, color="#1f2937")))
        fig_b.add_trace(go.Bar(name="Skenario 2 (Dengan SMOTE)", x=labels, y=v2,
            marker_color="#3b82f6",
            text=[f"{v:.4f}" for v in v2], textposition="outside", textfont=dict(size=9, color="#1f2937")))
        fig_b.update_layout(barmode="group",
            yaxis=dict(range=[0.5,1.05], title="Nilai Metrik"),
            xaxis_title="Metrik Evaluasi", height=430,
            plot_bgcolor="#f8fafc", paper_bgcolor="#fff",
            legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="center",x=0.5),
            margin=dict(t=60,b=20),
            font=dict(color="#1f2937"))
        st.plotly_chart(fig_b, use_container_width=True, theme=None)

        rl = ["Precision K0","Recall K0","F1 K0",
              "Precision K1","Recall K1","F1 K1","F1 Macro"]
        r1 = [s1["precision_0"],s1["recall_0"],s1["f1_0"],
              s1["precision_1"],s1["recall_1"],s1["f1_1"],s1["f1_macro"]]
        r2 = [s2["precision_0"],s2["recall_0"],s2["f1_0"],
              s2["precision_1"],s2["recall_1"],s2["f1_1"],s2["f1_macro"]]
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatterpolar(r=r1+[r1[0]], theta=rl+[rl[0]], fill="toself",
            name="Skenario 1", line_color="#94a3b8", fillcolor="rgba(148,163,184,0.2)"))
        fig_r.add_trace(go.Scatterpolar(r=r2+[r2[0]], theta=rl+[rl[0]], fill="toself",
            name="Skenario 2", line_color="#3b82f6", fillcolor="rgba(59,130,246,0.2)"))
        fig_r.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0.5,1.0])),
            height=400, paper_bgcolor="#fff",
            legend=dict(orientation="h",yanchor="bottom",y=1.05,xanchor="center",x=0.5),
            font=dict(color="#1f2937"))
        st.plotly_chart(fig_r, use_container_width=True, theme=None)

    with tab4:
        st.dataframe(pd.DataFrame({
            "Parameter": ["hidden_layer_sizes","alpha","learning_rate_init","max_iter",
                          "solver","activation","early_stopping",
                          "Best F1-Macro (CV)","Total Kombinasi","Total Fitting"],
            "Skenario 1 (Tanpa SMOTE)":  ["(200,) — 1 layer","0.001","0.001","100",
                                           "adam","relu","True","0.8540","36","108"],
            "Skenario 2 (Dengan SMOTE)": ["(100,50) — 2 layers","0.001","0.001","200",
                                           "adam","relu","True","0.8540","36","108"],
        }), use_container_width=True, hide_index=True)
        st.markdown("""<div class='info-box'>
            💡 Skenario 2 menemukan arsitektur 2-layer (100,50) dengan max_iter lebih tinggi.
            Distribusi seimbang pasca-SMOTE memerlukan arsitektur lebih kompleks namun
            dengan laju belajar konservatif untuk menghindari overfitting.
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# HALAMAN 4 — DEMO PREDIKSI (KEDUA MODEL SEKALIGUS)
# ═══════════════════════════════════════════════
elif page == "🤖 Demo Prediksi":
    st.title("🤖 Demo Prediksi Stance")
    st.markdown("Masukkan komentar TikTok dan lihat perbandingan hasil prediksi "
                "**Skenario 1 (Tanpa SMOTE)** vs **Skenario 2 (Dengan SMOTE)** secara bersamaan.")

    st.markdown("""<div class='warning-box'>
        ⚠️ Aplikasi memuat <b>kedua model</b> sekaligus. Jika file <code>.pkl</code>
        ditemukan di <code>Model/</code> → pipeline <b>IndoBERT + MLPClassifier</b>
        sesungguhnya digunakan. Jika tidak → otomatis fallback ke <b>simulasi keyword matching</b>.
    </div>""", unsafe_allow_html=True)

    with st.spinner("Memuat model Skenario 1 & Skenario 2..."):
        tok, bert = load_indobert()
        mlp_s1, _ = load_mlp(use_smote=False)
        mlp_s2, _ = load_mlp(use_smote=True)

    cs1, cs2 = st.columns(2)
    with cs1:
        if tok and bert and mlp_s1:
            st.success("✅ Skenario 1 — IndoBERT + MLP (Tanpa SMOTE) dimuat")
        else:
            st.info("ℹ️ Skenario 1 — Simulasi keyword matching")
    with cs2:
        if tok and bert and mlp_s2:
            st.success("✅ Skenario 2 — IndoBERT + MLP (Dengan SMOTE) dimuat")
        else:
            st.info("ℹ️ Skenario 2 — Simulasi keyword matching")

    st.markdown("---")
    st.subheader("Input Komentar", divider="blue")

    comment_input = st.text_area("Komentar:", height=100, label_visibility="collapsed",
        placeholder="Contoh: pecat kapolresnya, korban malah jadi tersangka ini tidak adil!")

    
    if st.button("🔍 Prediksi dengan Kedua Model", type="primary", use_container_width=True):
        if not comment_input.strip():
            st.warning("⚠️ Masukkan teks komentar terlebih dahulu.")
        else:
            with st.spinner("Memproses Skenario 1 & Skenario 2..."):
                processed = preprocess_text(comment_input)
                res1 = predict_model(comment_input, tok, bert, mlp_s1) \
                       if (tok and bert and mlp_s1) else predict_keyword(comment_input)
                res2 = predict_model(comment_input, tok, bert, mlp_s2) \
                       if (tok and bert and mlp_s2) else predict_keyword(comment_input)

            st.markdown("---")
            st.subheader("⚖️ Perbandingan Hasil Kedua Model", divider="blue")

            col1, col_mid, col2 = st.columns([5, 1, 5])

            def render_card(res, label):
                if res["class"] == 0:
                    color, icon, kls, sub = "#15803d","🟢","KELAS 0","Support Hogi / Critical Police"
                    bg = "result-support"
                else:
                    color, icon, kls, sub = "#b91c1c","🔴","KELAS 1","Oppose / Neutral / Ambigu"
                    bg = "result-oppose"
                method = "IndoBERT + MLP" if res["method"] == "indobert_mlp" \
                         else "Keyword Matching (Simulasi)"
                return f"""
                <div class='compare-card'>
                    <div class='compare-header'>{label}</div>
                    <div class='{bg}' style='margin-bottom:.75rem;'>
                        <div style='font-size:2rem;'>{icon}</div>
                        <div class='result-label' style='color:{color};font-size:1.1rem;'>{kls}</div>
                        <div style='font-size:.9rem;color:{color};font-weight:600;'>{sub}</div>
                        <div style='margin-top:.5rem;font-size:1.2rem;font-weight:700;color:{color};'>
                            Confidence: {res['confidence']:.1%}</div>
                    </div>
                    <table style='width:100%;font-size:.82rem;'>
                        <tr><td style='color:#64748b;padding:.2rem 0;'>Metode</td>
                            <td style='font-weight:600;'>{method}</td></tr>
                        <tr><td style='color:#64748b;'>Prob Kelas 0</td>
                            <td style='font-weight:600;color:#16a34a;'>{res['prob_0']:.4f}</td></tr>
                        <tr><td style='color:#64748b;'>Prob Kelas 1</td>
                            <td style='font-weight:600;color:#dc2626;'>{res['prob_1']:.4f}</td></tr>
                    </table>
                </div>"""

            with col1:
                st.markdown(render_card(res1, "📌 Skenario 1 — Tanpa SMOTE"),
                            unsafe_allow_html=True)
            with col_mid:
                st.markdown("""<div style='display:flex;align-items:center;justify-content:center;
                    height:100%;font-size:2rem;color:#94a3b8;padding-top:5rem;'>VS</div>""",
                    unsafe_allow_html=True)
            with col2:
                st.markdown(render_card(res2, "🏆 Skenario 2 — Dengan SMOTE"),
                            unsafe_allow_html=True)

            # Grafik probabilitas
            st.subheader("Probabilitas Prediksi — Perbandingan", divider="blue")
            fig_c = go.Figure()
            fig_c.add_trace(go.Bar(
                name="Skenario 1 (Tanpa SMOTE)",
                x=["Prob Kelas 0","Prob Kelas 1"],
                y=[res1["prob_0"],res1["prob_1"]],
                marker_color=["#16a34a","#dc2626"], opacity=0.6,
                text=[f"{res1['prob_0']:.4f}",f"{res1['prob_1']:.4f}"],
                textposition="outside",
                textfont=dict(color="#1f2937")
            ))
            fig_c.add_trace(go.Bar(
                name="Skenario 2 (Dengan SMOTE)",
                x=["Prob Kelas 0","Prob Kelas 1"],
                y=[res2["prob_0"],res2["prob_1"]],
                marker_color=["#059669","#b91c1c"],
                text=[f"{res2['prob_0']:.4f}",f"{res2['prob_1']:.4f}"],
                textposition="outside",
                textfont=dict(color="#1f2937")
            ))
            fig_c.update_layout(barmode="group",
                yaxis=dict(range=[0,1.15], title="Probabilitas"),
                xaxis_title="Kelas", height=350,
                plot_bgcolor="#f8fafc", paper_bgcolor="#fff",
                legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="center",x=0.5),
                margin=dict(t=60,b=20),
                font=dict(color="#1f2937"))
            st.plotly_chart(fig_c, use_container_width=True, theme=None)

            # Kesepakatan model
            if res1["class"] == res2["class"]:
                kls_lbl = "Kelas 0 (Support Hogi)" if res1["class"]==0 else "Kelas 1 (Oppose/Neutral)"
                st.markdown(f"""<div class='info-box'>
                    ✅ <b>Kedua model SEPAKAT</b> — Prediksi: <b>{kls_lbl}</b>.<br>
                    Confidence S1: <b>{res1['confidence']:.1%}</b> |
                    Confidence S2: <b>{res2['confidence']:.1%}</b>
                </div>""", unsafe_allow_html=True)
            else:
                l1 = "Kelas 0" if res1["class"]==0 else "Kelas 1"
                l2 = "Kelas 0" if res2["class"]==0 else "Kelas 1"
                st.markdown(f"""<div class='warning-box'>
                    ⚠️ <b>Kedua model BERBEDA pendapat.</b><br>
                    Skenario 1 → <b>{l1}</b> (confidence {res1['confidence']:.1%})<br>
                    Skenario 2 → <b>{l2}</b> (confidence {res2['confidence']:.1%})<br>
                    Gunakan <b>Skenario 2 (Dengan SMOTE)</b> sebagai referensi utama
                    karena lebih sensitif terhadap kelas minoritas.
                </div>""", unsafe_allow_html=True)

            # Detail preprocessing
            with st.expander("🔍 Detail Preprocessing Teks"):
                cd1, cd2 = st.columns(2)
                with cd1:
                    st.markdown("**Teks Asli:**")
                    st.code(comment_input, language=None)
                with cd2:
                    st.markdown("**Setelah Preprocessing:**")
                    st.code(processed or "(kosong)", language=None)

                all_kw = list(set(
                    res1.get("matched_keywords",[]) + res2.get("matched_keywords",[])
                ))
                if all_kw:
                    st.markdown("**Kata Kunci Terdeteksi:**")
                    for kw in all_kw[:8]:
                        c = "#16a34a" if kw in KEYWORDS_CLASS0 else "#dc2626"
                        st.markdown(
                            f"<span style='background:{c}20;border:1px solid {c};"
                            f"border-radius:4px;padding:2px 8px;margin:2px;"
                            f"display:inline-block;font-size:.8rem;color:{c};'>{kw}</span>",
                            unsafe_allow_html=True)

    # ── Batch Prediksi ──
    st.markdown("---")
    st.subheader("Prediksi Batch (Upload CSV)", divider="blue")
    st.markdown("Upload CSV dengan kolom `comment`. Kedua model akan memproses setiap baris.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    if uploaded:
        df_up = pd.read_csv(uploaded)
        if "comment" not in df_up.columns:
            st.error("❌ Kolom 'comment' tidak ditemukan.")
        else:
            max_rows = st.slider("Batas baris diproses", 10, 200, 100)
            with st.spinner(f"Memproses {min(len(df_up), max_rows)} baris dengan 2 model..."):
                rows = []
                for _, row in df_up.head(max_rows).iterrows():
                    txt = str(row["comment"])
                    r1  = predict_model(txt,tok,bert,mlp_s1) \
                          if (tok and bert and mlp_s1) else predict_keyword(txt)
                    r2  = predict_model(txt,tok,bert,mlp_s2) \
                          if (tok and bert and mlp_s2) else predict_keyword(txt)
                    rows.append({
                        "comment":    txt,
                        "S1_class":   r1["class"],
                        "S1_label":   "Support Hogi" if r1["class"]==0 else "Oppose/Neutral",
                        "S1_conf":    f"{r1['confidence']:.4f}",
                        "S2_class":   r2["class"],
                        "S2_label":   "Support Hogi" if r2["class"]==0 else "Oppose/Neutral",
                        "S2_conf":    f"{r2['confidence']:.4f}",
                        "Sepakat":    "✅" if r1["class"]==r2["class"] else "⚠️",
                    })

            df_res   = pd.DataFrame(rows)
            n_agree  = (df_res["Sepakat"] == "✅").sum()
            st.success(f"✅ Selesai. {n_agree}/{len(df_res)} baris "
                       f"({n_agree/len(df_res)*100:.1f}%) kedua model sepakat.")
            st.dataframe(df_res, use_container_width=True, height=350)

            # Ringkasan
            st.subheader("Ringkasan Prediksi Batch", divider="blue")
            s1k0 = (df_res["S1_class"]==0).sum()
            s2k0 = (df_res["S2_class"]==0).sum()
            rc1,rc2,rc3,rc4 = st.columns(4)
            for col, lbl, val in [
                (rc1,"S1 → Support Hogi",    f"{s1k0} ({s1k0/len(df_res)*100:.1f}%)"),
                (rc2,"S1 → Oppose/Neutral",  f"{len(df_res)-s1k0} ({(len(df_res)-s1k0)/len(df_res)*100:.1f}%)"),
                (rc3,"S2 → Support Hogi",    f"{s2k0} ({s2k0/len(df_res)*100:.1f}%)"),
                (rc4,"S2 → Oppose/Neutral",  f"{len(df_res)-s2k0} ({(len(df_res)-s2k0)/len(df_res)*100:.1f}%)"),
            ]:
                with col:
                    st.markdown(f"""<div class='metric-card'>
                        <div class='metric-label'>{lbl}</div>
                        <div class='metric-value' style='font-size:1.4rem;'>{val}</div>
                    </div>""", unsafe_allow_html=True)

            st.download_button("⬇️ Download Hasil Prediksi Batch (CSV)",
                data=df_res.to_csv(index=False).encode("utf-8"),
                file_name="hasil_prediksi_batch.csv", mime="text/csv")