import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

st.set_page_config(page_title="Credit Card Default Dashboard", layout="wide")

st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        padding-top: 20px;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    [data-testid="stSidebar"] h1 {
        font-size: 1.3rem;
        margin-bottom: 0.2rem;
    }

    [data-testid="stSidebar"] .sidebar-subtitle {
        color: #9aa4b2;
        font-size: 0.88rem;
        margin-top: -0.4rem;
        margin-bottom: 0.8rem;
    }

    [data-testid="stSidebar"] .sidebar-badge {
        background: rgba(34, 139, 230, 0.16);
        border: 1px solid rgba(52, 152, 219, 0.35);
        border-radius: 10px;
        padding: 8px 10px;
        margin-bottom: 12px;
        font-size: 0.85rem;
    }
            
    div[role="radiogroup"] {
    width: 100%;
    }

    div[role="radiogroup"] > label:hover {
    background-color: rgba(255, 255, 255, 0.1); /* terang dikit */
    border-radius: 10px;
    width: full;
    cursor: pointer;
    }

    /* Menghilangkan dot/bulatan radio */
    div[role="radiogroup"] [data-testid="stMarkdownArmchair"] {
        display: none;
    }
    
    /* Kasih padding ke tiap item menu */
    div[role="radiogroup"] > label {
        padding: 12px 15px;
        margin-bottom: 5px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_raw_data():
    df = pd.read_csv('default of credit card clients.csv', sep=';', header=1)
    return df

# Helper to format data based on original notebook
@st.cache_data
def load_cleaned_data():
    df = load_raw_data().copy()
    df = df.drop("ID", axis=1)
    df.rename(columns={
        "PAY_0": "PAY_1",
        "default payment next month": "default"
    }, inplace=True)
    df["EDUCATION"] = df["EDUCATION"].replace([0,5,6], 4)
    df["MARRIAGE"] = df["MARRIAGE"].replace([0], 3)
    duplicate = df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    return [duplicate, df]

df_raw = load_raw_data()
duplicate, df_clean = load_cleaned_data()
default_rate = df_clean["default"].mean() * 100

st.sidebar.title("Credit Risk App")
st.sidebar.markdown("<p class='sidebar-subtitle'>Dashboard Analitik & Prediksi</p>", unsafe_allow_html=True)
st.sidebar.markdown(
    f"<div class='sidebar-badge'><b>Data Snapshot</b><br/>"
    f"Records: {df_clean.shape[0]:,}<br/>"
    f"Default Rate: {default_rate:.2f}%</div>",
    unsafe_allow_html=True
)
st.sidebar.markdown("-----")
menu = st.sidebar.radio( 
    "Navigation",
    ["🏠 Home", "📊 Dataset & EDA", "⚙️ Preprocessing", "🧠 Training Model", "🎯 Prediction Demo"],
    label_visibility="collapsed")


if menu == "🏠 Home":
    st.title("Credit Card Default Dashboard")
    st.caption("Platform analitik untuk evaluasi risiko gagal bayar kartu kredit berbasis Logistic Regression.")

    total_data = df_clean.shape[0]
    total_fitur = df_clean.drop("default", axis=1).shape[1]

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Jumlah Data", f"{total_data:,}")
    with m2:
        st.metric("Jumlah Fitur", total_fitur)
    with m3:
        st.metric("Default Rate", f"{default_rate:.2f}%")

    st.write("---")
    st.subheader("Tentang Project")
    st.write(
        "Aplikasi ini digunakan untuk memprediksi apakah nasabah kartu kredit akan gagal bayar bulan depan"
        "(Default Payment). Dataset ini berasal dari UCI Machine Learning Repository, tentang 'Default of Credit Card Clients Dataset'.\n\n"
        "Dataset yang digunakan berasal dari UCI Machine Learning Repository (Default of Credit Card Clients). "
        "Aplikasi ini menyajikan alur analitik end-to-end: eksplorasi data, preprocessing, pelatihan model, "
        "hingga prediksi risiko pada profil nasabah baru."
    )

    st.image(
        "https://images.unsplash.com/photo-1554224155-6726b3ff858f?q=80&w=1600&auto=format&fit=crop",
        width="stretch"
    )

    st.write("---")
    st.subheader("Kapabilitas Utama")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        1. Meninjau struktur data dan statistik inti.
        2. Mengeksplorasi visualisasi utama: target, umur, korelasi, dan limit saldo.
        3. Menjalankan preprocessing interaktif: split dan scaling.
        """)
    with c2:
        st.markdown("""
        4. Melatih Logistic Regression dengan konfigurasi parameter berbeda.
        5. Mengevaluasi performa model melalui classification report dan confusion matrix.
        6. Mensimulasikan prediksi risiko pada profil nasabah custom.
        """)

    st.write("---")
    st.subheader("Alur Kerja Cepat")
    st.info("Alur penggunaan: Dataset & EDA → Preprocessing → Training Model → Prediction Demo")

    with st.expander("Kenapa alurnya seperti ini?"):
        st.write(
            "EDA membantu memetakan pola data serta potensi bias awal. Preprocessing memastikan data siap untuk "
            "pemodelan. Tahap training digunakan untuk mendapatkan konfigurasi model yang optimal, sementara "
            "Prediction Demo menampilkan hasil inferensi pada input baru secara real-time."
        )

    st.write("---")
    st.subheader("Preview Target Distribution")
    target_preview = df_clean["default"].value_counts().sort_index().rename({0: "Not Default", 1: "Default"})
    st.bar_chart(target_preview)

    st.markdown(
        "Pada distribusi target terlihat jumlah nasabah **Not Default** lebih tinggi dibanding **Default**. "
        "Temuan ini menunjukkan kelas tidak seimbang (class imbalance) ringan, sehingga evaluasi model tidak "
        "cukup hanya dengan akurasi, tetapi juga perlu melihat precision, recall, dan confusion matrix."
    )
    
    st.info(
        "💡 **Ini adalah preview ringkas.** Untuk analisis visualisasi lebih detail dan interaktif "
        "(target distribution, umur, education, korelasi, limit saldo), kunjungi menu **📊 Dataset & EDA**."
    )

elif menu == "📊 Dataset & EDA":
    st.title("Dataset & Exploratory Data Analysis (EDA)")
    
    st.subheader("Data Overview (Raw Data)")
    st.write(df_raw.head())
    st.write(f"Baris: {df_raw.shape[0]}, Kolom: {df_raw.shape[1]}")
    
    st.write("---")
    st.subheader("Visualisasi EDA")
    grafik = st.selectbox("Pilih Grafik yang ingin ditampilkan:", [
        "Distribusi Target (Default vs Not Default)",
        "Distribusi Umur",
        "Distribusi Education",
        "Korelasi Heatmap",
        "Limit Saldo terhadap Target"
    ])
    
    fig, ax = plt.subplots(figsize=(10,6))
    if grafik == "Distribusi Target (Default vs Not Default)":
        sns.countplot(data=df_clean, x='default', palette='viridis', ax=ax)
        ax.set_title("Distribusi Nasabah Default (1) vs Non-Default (0)")
    elif grafik == "Distribusi Umur":
        sns.histplot(df_clean['AGE'], bins=30, kde=True, ax=ax, color='skyblue')
        ax.set_title("Distribusi Umur Nasabah")
    elif grafik == "Distribusi Education":
        sns.countplot(data=df_clean, x='EDUCATION', palette='muted', ax=ax)
        ax.set_title("Sebaran Tingkat Pendidikan")
    elif grafik == "Korelasi Heatmap":
        corr = df_clean[['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'PAY_AMT1', 'default']].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title("Heatmap Korelasi Beberapa Fitur Utama")
    elif grafik == "Limit Saldo terhadap Target":
        sns.boxplot(data=df_clean, x='default', y='LIMIT_BAL', palette='pastel', ax=ax)
        ax.set_title("Limit Saldo vs Status Default")
        
    st.pyplot(fig)

elif menu == "⚙️ Preprocessing":
    st.title("Data Preprocessing")
    st.write("Pada tahap ini kita akan mempersiapkan data sebelum dimasukkan ke model prediksi.")
    
    st.subheader("1. Pembersihan Data")
    st.write("Sesuai pada Notebook asli: Menghapus ID, mengganti nama `PAY_0` dan target `default`, serta menangani duplikat/outlier kelas EDUCATION & MARRIAGE.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("Cek Missing Values:")
        st.write(df_clean.isnull().sum())
    with col2:
        st.write("Cek Duplikat:")
        st.write(f"Sebelumnya {duplicate} duplicated, telah dihapus.")
        
    st.write("---")
    st.subheader("2. Data Splitting & Scaling")
    
    test_size = st.slider("Tentukan Test Size (%)", min_value=10, max_value=50, value=20, step=5)
    scaler_choice = st.selectbox("Pilih Scaler", ["StandardScaler", "MinMaxScaler"])
    shuffle_data = st.checkbox("Shuffle Data?", value=True)
    
    if st.button("Proses Data"):
        X = df_clean.drop("default", axis=1)
        y = df_clean["default"]
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=(test_size/100.0), random_state=42, shuffle=shuffle_data
        )
        
        # Scale
        if scaler_choice == "StandardScaler":
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
            
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Simpan ke session_state agar bisa digunakan di menu Training
        st.session_state['X_train_scaled'] = X_train_scaled
        st.session_state['X_test_scaled'] = X_test_scaled
        st.session_state['y_train'] = y_train
        st.session_state['y_test'] = y_test
        st.session_state['scaler_used'] = scaler
        st.session_state['features_list'] = list(X.columns)
        
        st.success(f"Berhasil! Data dibagi menjadi Training: {X_train.shape[0]} baris, Testing: {X_test.shape[0]} baris.")
        st.write("Preview X_train Scaled:")
        st.dataframe(pd.DataFrame(X_train_scaled, columns=X.columns).head())

elif menu == "🧠 Training Model":
    st.title("Training Model Machine Learning")
    
    if 'X_train_scaled' not in st.session_state:
        st.warning("Silakan ke menu ⚙️ Preprocessing terlebih dahulu lalu tekan 'Proses Data'.")
    else:
        st.write("Kita akan melatih model **Logistic Regression**.")
        
        col1, col2 = st.columns(2)
        with col1:
            C_param = st.selectbox("Pilih Parameter C (Regularization)", [0.01, 0.1, 1.0, 10.0])
            max_iter = st.slider("Maksimal Iterasi (max_iter)", 100, 1000, 100, step=100)
            
        with col2:
            solver_param = st.selectbox("Pilih Solver", ["liblinear", "lbfgs"])
            
        if st.button("Train Model!"):
            with st.spinner("Training sedang berlangsung..."):
                model = LogisticRegression(C=C_param, solver=solver_param, max_iter=max_iter)
                
                X_train = st.session_state['X_train_scaled']
                y_train = st.session_state['y_train']
                X_test = st.session_state['X_test_scaled']
                y_test = st.session_state['y_test']
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                acc = accuracy_score(y_test, y_pred)
                
                # Simpan model sementara
                st.session_state['trained_model'] = model
                
                st.success(f"Training Selesai! Accuracy: **{acc*100:.2f}%**")
                
                col3, col4 = st.columns(2)
                with col3:
                    st.write("**Classification Report:**")
                    report = classification_report(y_test, y_pred, output_dict=True)
                    st.dataframe(pd.DataFrame(report).transpose())
                    
                with col4:
                    st.write("**Confusion Matrix:**")
                    cm = confusion_matrix(y_test, y_pred)
                    fig_cm, ax_cm = plt.subplots(figsize=(4,3))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                    st.pyplot(fig_cm)

elif menu == "🎯 Prediction Demo":
    st.title("Prediction Demo (Result)")
    st.write("Gunakan form di bawah ini untuk mensimulasikan apakah data input profil custom ini akan Default atau tidak.")
    
    if st.checkbox("Gunakan Model dari Notebook File (logistic_model.pkl)?", value=True):
        try:
            model = joblib.load('logistic_model.pkl')
            scaler = StandardScaler()
            
            # Khusus untuk model notebook, perlukan fit dummy biar scaler bisa transform
            X = df_clean.drop("default", axis=1)
            scaler.fit(X)
            
            st.success("Loaded model dari logistic_model.pkl !")
        except:
            st.error("Model logistic_model.pkl tidak ditemukan!")
            model = None
            scaler = None
    else:
        if 'trained_model' in st.session_state:
            model = st.session_state['trained_model']
            scaler = st.session_state['scaler_used']
            st.success("Loaded model dari hasil Training Interactive!")
        else:
            st.warning("Anda belum men-train model di menu sebelumnya!")
            model = None
            scaler = None
            
    if model and scaler:
        st.subheader("Input Dummy Data Nasabah")
        
        with st.form("demo_form"):
            colA, colB, colC = st.columns(3)
            with colA:
                limit_bal = st.number_input("LIMIT_BAL", min_value=10000, value=50000)
                sex = st.selectbox("SEX (1=Male, 2=Female)", [1, 2])
                education = st.selectbox("EDUCATION (1=Grad, 2=Univ, 3=HS, 4=Others)", [1, 2, 3, 4])
                marriage = st.selectbox("MARRIAGE (1=Married, 2=Single, 3=Others)", [1, 2, 3])
                age = st.number_input("AGE", min_value=21, value=30)
                
            with colB:
                pay_1 = st.slider("PAY_1 (Sep)", -2, 8, 0)
                pay_2 = st.slider("PAY_2 (Aug)", -2, 8, 0)
                pay_3 = st.slider("PAY_3 (Jul)", -2, 8, 0)
                pay_4 = st.slider("PAY_4 (Jun)", -2, 8, 0)
                pay_5 = st.slider("PAY_5 (May)", -2, 8, 0)
                pay_6 = st.slider("PAY_6 (Apr)", -2, 8, 0)
                
            with colC:
                bill_amt1 = st.number_input("BILL_AMT1", value=0)
                bill_amt2 = st.number_input("BILL_AMT2", value=0)
                bill_amt3 = st.number_input("BILL_AMT3", value=0)
                bill_amt4 = st.number_input("BILL_AMT4", value=0)
                bill_amt5 = st.number_input("BILL_AMT5", value=0)
                bill_amt6 = st.number_input("BILL_AMT6", value=0)
                
                pay_amt1 = st.number_input("PAY_AMT1", value=1000)
                pay_amt2 = st.number_input("PAY_AMT2", value=1000)
                pay_amt3 = st.number_input("PAY_AMT3", value=1000)
                pay_amt4 = st.number_input("PAY_AMT4", value=1000)
                pay_amt5 = st.number_input("PAY_AMT5", value=1000)
                pay_amt6 = st.number_input("PAY_AMT6", value=1000)
                
            submit = st.form_submit_button("Prediksi Kemungkinan Default")
            
        if submit:
            input_dict = {
                'LIMIT_BAL': limit_bal, 'SEX': sex, 'EDUCATION': education, 'MARRIAGE': marriage, 'AGE': age,
                'PAY_1': pay_1, 'PAY_2': pay_2, 'PAY_3': pay_3, 'PAY_4': pay_4, 'PAY_5': pay_5, 'PAY_6': pay_6,
                'BILL_AMT1': bill_amt1, 'BILL_AMT2': bill_amt2, 'BILL_AMT3': bill_amt3, 'BILL_AMT4': bill_amt4, 'BILL_AMT5': bill_amt5, 'BILL_AMT6': bill_amt6,
                'PAY_AMT1': pay_amt1, 'PAY_AMT2': pay_amt2, 'PAY_AMT3': pay_amt3, 'PAY_AMT4': pay_amt4, 'PAY_AMT5': pay_amt5, 'PAY_AMT6': pay_amt6
            }
            input_df = pd.DataFrame([input_dict])
            
            scaled_input = scaler.transform(input_df)
            pred = model.predict(scaled_input)[0]
            prob = model.predict_proba(scaled_input)[0][1]
            
            if pred == 1:
                st.error(f"Prediksi: **Gagal Bayar (Default)** dengan probabilitas sebesar {prob*100:.2f}%")
            else:
                st.success(f"Prediksi: **Lancar / Tidak Gagal Bayar (Not Default)** dengan probabilitas default {prob*100:.2f}%")