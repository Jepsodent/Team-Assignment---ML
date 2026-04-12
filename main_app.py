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
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

    html, body, [class*="css"], * {
        font-family: 'Poppins', sans-serif !important;
    }

    /* ── MAIN BACKGROUND ── */
    .stApp, [data-testid="stAppViewContainer"] {
        background-color: #F2EFFF !important;
    }
    [data-testid="stHeader"] {
        background-color: #F2EFFF !important;
    }

    h1 { font-size: 3.8rem !important; font-weight: 700 !important; letter-spacing: -0.5px; line-height: 1.2 !important; }
    h2 { font-size: 2rem !important; font-weight: 600 !important; }
    h3 { font-size: 1.5rem !important; font-weight: 600 !important; }
    p, li { font-size: 1.15rem !important; line-height: 1.8 !important; }

    /* ── LAYOUT ── */
    .block-container {
        padding-top: 3rem !important;
        padding-bottom: 3rem !important;
        padding-left: 3.5rem !important;
        padding-right: 3.5rem !important;
        max-width: 100% !important;
    }

    /* ── SIDEBAR BACKGROUND semua lapisan ── */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div,
    section[data-testid="stSidebar"] > div > div,
    section[data-testid="stSidebar"] > div > div > div,
    section[data-testid="stSidebar"] > div > div > div > div {
        background-color: #C3C0FF !important;
    }
    section[data-testid="stSidebar"] { padding-top: 1.5rem !important; }

    /* ── SIDEBAR: paksa SEMUA teks gelap (sidebar terang) ── */
    section[data-testid="stSidebar"] * {
        color: #2d2a6e !important;
    }

    /* Judul sidebar */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] {
        color: #1e1a5e !important;
        font-size: 1.4rem !important;
        font-weight: 700 !important;
    }

    section[data-testid="stSidebar"] hr {
        border-color: rgba(79,70,229,0.25) !important;
        margin: 0.75rem 0 1rem 0 !important;
    }

    /* ── COLLAPSE BUTTON ── */
    [data-testid="stSidebarCollapseButton"] {
        position: absolute !important;
        top: 1rem !important;
        right: -1rem !important;
    }

    [data-testid="stSidebarCollapseButton"] button {
        background-color: #a5b4fc !important;
        border: 1px solid rgba(79,70,229,0.3) !important;
        border-radius: 50% !important;
        width: 32px !important;
        height: 32px !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
        transition: background 0.15s !important;
    }

    [data-testid="stSidebarCollapseButton"] button:hover {
        background-color: #4F46E5 !important;
    }

    [data-testid="stSidebarCollapseButton"] span,
    [data-testid="stSidebarCollapseButton"] span[data-testid="stIconMaterial"] {
        color: #1e1a5e !important;
        font-size: 18px !important;
    }

    [data-testid="stSidebarHeader"] {
        background-color: #C3C0FF !important;
        padding: 0.75rem 0 !important;
        min-height: 48px !important;
    }

    /* ── SIDEBAR: hapus semua padding container bawaan ── */
    section[data-testid="stSidebar"] > div:first-child {
        padding: 1.5rem 0 0 0 !important;
    }

    /* judul & hr tetap punya padding sendiri */
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] {
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
    }

    /* container stRadio juga nol padding */
    section[data-testid="stSidebar"] [data-testid="stRadio"],
    section[data-testid="stSidebar"] .stRadio,
    section[data-testid="stSidebar"] [data-testid="stRadio"] > div,
    section[data-testid="stSidebar"] [data-testid="stElementContainer"] {
        padding: 0 !important;
        margin: 0 !important;
        width: 100% !important;
    }

    /* ── RADIO GROUP wrapper ── */
    section[data-testid="stSidebar"] div[role="radiogroup"] {
        display: flex !important;
        flex-direction: column !important;
        gap: 0 !important;
        width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* tiap item label mentok full width */
    section[data-testid="stSidebar"] div[role="radiogroup"] > label {
        display: flex !important;
        align-items: center !important;
        width: 100% !important;
        box-sizing: border-box !important;
        padding: 16px 1.5rem !important;
        margin: 0 !important;
        border-radius: 0 !important;
        font-size: 1.15rem !important;
        font-weight: 500 !important;
        color: #3730a3 !important;
        background: transparent !important;
        border: none !important;
        border-left: 4px solid transparent !important;
        transition: background 0.15s, color 0.15s;
        cursor: pointer;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background-color: rgba(79,70,229,0.12) !important;
        color: #1e1a5e !important;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
        background-color: rgba(79,70,229,0.22) !important;
        color: #1e1a5e !important;
        font-weight: 700 !important;
        border-left: 4px solid #4F46E5 !important;
    }

    /* sembunyikan radio dot */
    section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }

    /* teks di dalam label */
    section[data-testid="stSidebar"] div[role="radiogroup"] p {
        font-size: 1.15rem !important;
        font-weight: inherit !important;
        color: inherit !important;
        line-height: 1 !important;
        margin: 0 !important;
    }

    /* ── BUTTONS ── */
    .stButton > button {
        font-family: 'Poppins', sans-serif !important;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        padding: 0.7rem 2rem !important;
        border-radius: 10px !important;
        border: none !important;
        background-color: #4F46E5 !important;
        color: white !important;
        transition: all 0.15s ease !important;
    }
    .stButton > button:hover {
        background-color: #3730a3 !important;
        transform: translateY(-1px);
    }

    .stSelectbox label, .stSlider label,
    .stNumberInput label, .stCheckbox label {
        font-size: 1.05rem !important;
        font-weight: 500 !important;
    }

    .stDataFrame { border-radius: 12px; overflow: hidden; border: 1px solid #e5e7eb; }
    .stAlert { border-radius: 10px !important; font-size: 1.05rem !important; }
    </style>
    """, unsafe_allow_html=True)

st.sidebar.markdown(
    '<div style="padding: 0 1.25rem 0.5rem 1.25rem; font-family: Poppins, sans-serif; font-size: 1.4rem; font-weight: 700; color: #f8fafc; letter-spacing: -0.3px;">Menu Utama</div>',
    unsafe_allow_html=True
)
st.sidebar.markdown(
    '<hr style="border: none; border-top: 1px solid rgba(255,255,255,0.1); margin: 0 0 0.5rem 0;" />',
    unsafe_allow_html=True
)
menu = st.sidebar.radio(
    "",
    ["Home", "Dataset & EDA", "Preprocessing", "Training Model", "Prediction Demo"])

@st.cache_data
def load_raw_data():
    df = pd.read_csv('default of credit card clients.csv', sep=';', header=1)
    return df

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


if menu == "Home":
    st.title("Selamat Datang di Project Machine Learning")
    st.write("""
    Aplikasi ini digunakan untuk memprediksi apakah nasabah kartu kredit akan gagal bayar bulan depan (Default Payment).
    Dataset ini berasal dari **UCI Machine Learning Repository**, tentang 'Default of Credit Card Clients Dataset'.
    
    Dashboard ini dibuat agar proses _Machine Learning_ dari dataset ini dapat lebih interaktif untuk dieksplorasi.
    
    **Gunakan Sidebar untuk Navigasi ke Halaman lain:**
    - **Dataset & EDA**: Melihat deskripsi dan visualisasi data
    - **Preprocessing**: Proses pembersihan data dan Train-Test split
    - **Training Model**: Melatih model dengan parameter yang dapat dikustomisasi
    - **Prediction Demo**: Simulasi prediksi dari data input baru
    """)
    st.image("https://via.placeholder.com/800x400.png?text=Credit+Card+Default+Analysis", use_column_width=True)

elif menu == "Dataset & EDA":
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

elif menu == "Preprocessing":
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
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=(test_size/100.0), random_state=42, shuffle=shuffle_data
        )
        
        if scaler_choice == "StandardScaler":
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
            
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        st.session_state['X_train_scaled'] = X_train_scaled
        st.session_state['X_test_scaled'] = X_test_scaled
        st.session_state['y_train'] = y_train
        st.session_state['y_test'] = y_test
        st.session_state['scaler_used'] = scaler
        st.session_state['features_list'] = list(X.columns)
        
        st.success(f"Berhasil! Data dibagi menjadi Training: {X_train.shape[0]} baris, Testing: {X_test.shape[0]} baris.")
        st.write("Preview X_train Scaled:")
        st.dataframe(pd.DataFrame(X_train_scaled, columns=X.columns).head())

elif menu == "Training Model":
    st.title("Training Model Machine Learning")
    
    if 'X_train_scaled' not in st.session_state:
        st.warning("Silakan ke menu Preprocessing terlebih dahulu lalu tekan 'Proses Data'.")
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

elif menu == "Prediction Demo":
    st.title("Prediction Demo (Result)")
    st.write("Gunakan form di bawah ini untuk mensimulasikan apakah data input profil custom ini akan Default atau tidak.")
    
    if st.checkbox("Gunakan Model dari Notebook File (logistic_model.pkl)?", value=True):
        try:
            model = joblib.load('logistic_model.pkl')
            scaler = StandardScaler()
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