import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import r2_score
import joblib


st.set_page_config(page_title="Credit Card Default Dashboard", layout="wide")

st.markdown(
    """
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
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        width: full;
        cursor: pointer;
    }

    div[role="radiogroup"] [data-testid="stMarkdownArmchair"] {
        display: none;
    }

    div[role="radiogroup"] > label {
        padding: 12px 15px;
        margin-bottom: 5px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_raw_data():
    df = pd.read_csv("default of credit card clients.csv", sep=";", header=1)
    return df


@st.cache_data
def load_cleaned_data():
    df = load_raw_data().copy()
    df = df.drop("ID", axis=1)
    df.rename(columns={"PAY_0": "PAY_1", "default payment next month": "default"}, inplace=True)
    df["EDUCATION"] = df["EDUCATION"].replace([0, 5, 6], 4)
    df["MARRIAGE"] = df["MARRIAGE"].replace([0], 3)
    duplicate = df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    return duplicate, df


def remove_outliers_iqr(df, columns):
    filtered_df = df.copy()
    kept_mask = pd.Series(True, index=filtered_df.index)

    for col in columns:
        q1 = filtered_df[col].quantile(0.25)
        q3 = filtered_df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        kept_mask &= filtered_df[col].between(lower, upper)

    filtered_df = filtered_df[kept_mask].copy()
    removed_count = int((~kept_mask).sum())
    return filtered_df, removed_count


def model_r2_score(model, x_test, y_test):
    y_prob = model.predict_proba(x_test)[:, 1]
    return r2_score(y_test, y_prob)


def extract_feature_importance(model, feature_names):
    if hasattr(model, "coef_"):
        values = abs(model.coef_[0])
        label = "|Coefficient|"
    elif hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        label = "Importance"
    else:
        return None, None

    imp_df = pd.DataFrame({"Feature": feature_names, "Importance": values})
    imp_df = imp_df.sort_values("Importance", ascending=False).reset_index(drop=True)
    return imp_df, label


df_raw = load_raw_data()
duplicate, df_clean = load_cleaned_data()
default_rate = df_clean["default"].mean() * 100

st.sidebar.title("Credit Risk App")
st.sidebar.markdown(
    f"<div class='sidebar-badge'><b>Data Snapshot</b><br/>"
    f"Records: {df_clean.shape[0]:,}<br/>"
    f"Default Rate: {default_rate:.2f}%</div>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("-----")

menu = st.sidebar.radio(
    "Navigation",
    ["ℹ️ About", "📊 Dataset & EDA", "⚙️ Preprocessing", "🧠 Training Model", "🎯 Prediction Demo"],
    label_visibility="collapsed",
)


if menu == "ℹ️ About":
    st.title("About")
    st.caption("BINUS Team Project - Credit Card Default Risk Analytics")

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
    st.subheader("Projek Machine Learning")
    st.write(
        "Project ini dibuat untuk menganalisis dan memprediksi risiko gagal bayar nasabah kartu kredit menggunakan"
        " pendekatan machine learning end-to-end: EDA, preprocessing, training, evaluasi, dan demo prediksi."
    )

    st.subheader("Anggota Tim")
    st.markdown(
        """
        1. Alexandria Natasya Beslar - 2802471801
        2. Darrell Nicholas Tandean - 2802393081
        3. Jefferson Gautama Swanto - 2802474476
        4. Timothy Alexandro Sibarani - 2802475024
        """
    )

    st.write("---")
    st.subheader("Sumber Dataset")
    st.write(
        "Dataset: Default of Credit Card Clients (UCI Machine Learning Repository). "
        "Target prediksi adalah kolom default (0 = tidak default, 1 = default bulan depan)."
    )

    st.image(
        "https://images.unsplash.com/photo-1554224155-6726b3ff858f?q=80&w=1600&auto=format&fit=crop",
        width=500,
    )


elif menu == "📊 Dataset & EDA":
    st.title("Dataset & Exploratory Data Analysis (EDA)")

    st.subheader("Deskripsi Dataset")
    st.write(
        "Dataset berisi informasi demografi, riwayat tagihan, riwayat pembayaran, dan jumlah pembayaran"
        " nasabah kartu kredit. Tujuannya adalah memprediksi potensi gagal bayar pada bulan berikutnya."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Rows (Clean)", f"{df_clean.shape[0]:,}")
    with c2:
        st.metric("Columns (Clean)", df_clean.shape[1])
    with c3:
        st.metric("Duplicates Removed", duplicate)
    with c4:
        st.metric("Missing Values", int(df_clean.isnull().sum().sum()))

    st.write("---")
    st.subheader("Sample Data")
    st.dataframe(df_clean.head(10), use_container_width=True)

    st.subheader("Statistik Ringkas")
    st.dataframe(df_clean.describe().transpose(), use_container_width=True)

    st.write("---")
    st.subheader("Visualisasi EDA")
    grafik = st.selectbox(
        "Pilih Grafik:",
        [
            "Distribusi Target (Default vs Not Default)",
            "Distribusi Umur",
            "Distribusi Limit Balance",
            "Distribusi Education",
            "Distribusi Marriage",
            "Rata-rata Default per Education",
            "Rata-rata Default per Marriage",
            "PAY_1 vs Default",
            "Scatter BILL_AMT1 vs PAY_AMT1",
            "Korelasi Heatmap",
            "Box Plot LIMIT_BAL vs Default",
        ],
    )

    show_outliers = st.checkbox("Tampilkan outliers di Box Plot", value=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    if grafik == "Distribusi Target (Default vs Not Default)":
        sns.countplot(data=df_clean, x="default", palette="viridis", ax=ax)
        ax.set_title("Distribusi Nasabah Default (1) vs Non-Default (0)")
    elif grafik == "Distribusi Umur":
        sns.histplot(df_clean["AGE"], bins=30, kde=True, ax=ax, color="skyblue")
        ax.set_title("Distribusi Umur Nasabah")
    elif grafik == "Distribusi Limit Balance":
        sns.histplot(df_clean["LIMIT_BAL"], bins=40, kde=True, ax=ax, color="teal")
        ax.set_title("Distribusi LIMIT_BAL")
    elif grafik == "Distribusi Education":
        sns.countplot(data=df_clean, x="EDUCATION", palette="muted", ax=ax)
        ax.set_title("Sebaran Tingkat Pendidikan")
    elif grafik == "Distribusi Marriage":
        sns.countplot(data=df_clean, x="MARRIAGE", palette="Set2", ax=ax)
        ax.set_title("Sebaran Status Pernikahan")
    elif grafik == "Rata-rata Default per Education":
        edu_rate = df_clean.groupby("EDUCATION")["default"].mean().reset_index()
        sns.barplot(data=edu_rate, x="EDUCATION", y="default", palette="rocket", ax=ax)
        ax.set_title("Default Rate per EDUCATION")
        ax.set_ylabel("Default Rate")
    elif grafik == "Rata-rata Default per Marriage":
        mar_rate = df_clean.groupby("MARRIAGE")["default"].mean().reset_index()
        sns.barplot(data=mar_rate, x="MARRIAGE", y="default", palette="mako", ax=ax)
        ax.set_title("Default Rate per MARRIAGE")
        ax.set_ylabel("Default Rate")
    elif grafik == "PAY_1 vs Default":
        sns.boxplot(data=df_clean, x="default", y="PAY_1", ax=ax, showfliers=show_outliers)
        ax.set_title("Riwayat PAY_1 terhadap Status Default")
    elif grafik == "Scatter BILL_AMT1 vs PAY_AMT1":
        sampled = df_clean.sample(min(5000, len(df_clean)), random_state=42)
        sns.scatterplot(data=sampled, x="BILL_AMT1", y="PAY_AMT1", hue="default", alpha=0.6, ax=ax)
        ax.set_title("BILL_AMT1 vs PAY_AMT1 (Sampled)")
    elif grafik == "Korelasi Heatmap":
        corr_cols = ["LIMIT_BAL", "AGE", "PAY_1", "BILL_AMT1", "PAY_AMT1", "default"]
        corr = df_clean[corr_cols].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Heatmap Korelasi Fitur Utama")
    elif grafik == "Box Plot LIMIT_BAL vs Default":
        sns.boxplot(data=df_clean, x="default", y="LIMIT_BAL", palette="pastel", ax=ax, showfliers=show_outliers)
        ax.set_title("LIMIT_BAL vs Status Default")

    st.pyplot(fig)


elif menu == "⚙️ Preprocessing":
    st.title("Data Preprocessing")
    st.write("Tahap ini menyiapkan data sebelum training model.")

    st.subheader("1. Pembersihan Dasar")
    st.write(
        "Langkah pembersihan: hapus ID, ubah PAY_0 -> PAY_1, ubah target menjadi default,"
        " rapikan kategori EDUCATION/MARRIAGE, dan hapus duplikat."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.write("Cek Missing Values:")
        st.write(df_clean.isnull().sum())
    with col2:
        st.write("Cek Duplikat:")
        st.write(f"Sebelumnya ada {duplicate} duplikat, sudah dihapus.")

    st.write("---")
    st.subheader("2. Opsi Buang Outliers (IQR)")
    remove_outliers = st.checkbox("Buang outliers sebelum split data", value=False)
    outlier_columns = [
        "LIMIT_BAL",
        "AGE",
        "BILL_AMT1",
        "BILL_AMT2",
        "BILL_AMT3",
        "BILL_AMT4",
        "BILL_AMT5",
        "BILL_AMT6",
        "PAY_AMT1",
        "PAY_AMT2",
        "PAY_AMT3",
        "PAY_AMT4",
        "PAY_AMT5",
        "PAY_AMT6",
    ]
    selected_outlier_cols = st.multiselect(
        "Kolom untuk deteksi outlier:",
        outlier_columns,
        default=["LIMIT_BAL", "BILL_AMT1", "PAY_AMT1"],
        disabled=not remove_outliers,
    )

    st.write("---")
    st.subheader("3. Data Splitting & Scaling")

    test_size = st.slider("Tentukan Test Size (%)", min_value=10, max_value=50, value=20, step=5)
    scaler_choice = st.selectbox("Pilih Scaler", ["StandardScaler", "MinMaxScaler"])
    shuffle_data = st.checkbox("Shuffle Data?", value=True)

    if st.button("Proses Data"):
        modeling_df = df_clean.copy()
        removed_count = 0

        if remove_outliers and selected_outlier_cols:
            modeling_df, removed_count = remove_outliers_iqr(modeling_df, selected_outlier_cols)
            st.info(f"Outlier removal aktif: {removed_count} baris dihapus. Sisa data: {len(modeling_df):,} baris.")

        X = modeling_df.drop("default", axis=1)
        y = modeling_df["default"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=(test_size / 100.0),
            random_state=42,
            shuffle=shuffle_data,
        )

        scaler = StandardScaler() if scaler_choice == "StandardScaler" else MinMaxScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        st.session_state["X_train_scaled"] = X_train_scaled
        st.session_state["X_test_scaled"] = X_test_scaled
        st.session_state["y_train"] = y_train
        st.session_state["y_test"] = y_test
        st.session_state["scaler_used"] = scaler
        st.session_state["features_list"] = list(X.columns)

        st.success(
            f"Berhasil! Train: {X_train.shape[0]} baris | Test: {X_test.shape[0]} baris"
            f" | Outlier removed: {removed_count}"
        )
        st.write("Preview X_train (scaled):")
        st.dataframe(pd.DataFrame(X_train_scaled, columns=X.columns).head(), use_container_width=True)


elif menu == "🧠 Training Model":
    st.title("Training & Perbandingan Model")

    if "X_train_scaled" not in st.session_state:
        st.warning("Silakan lakukan preprocessing dulu lalu klik Proses Data.")
    else:
        st.write("Model yang dibandingkan: Logistic Regression, Random Forest, dan Gradient Boosting.")
        st.write("Metric evaluasi tunggal sesuai request: R².")

        col1, col2, col3 = st.columns(3)
        with col1:
            C_param = st.selectbox("Logistic C", [0.01, 0.1, 1.0, 10.0])
        with col2:
            rf_estimators = st.slider("Random Forest n_estimators", 50, 400, 200, step=50)
        with col3:
            gb_estimators = st.slider("Gradient Boosting n_estimators", 50, 300, 150, step=25)

        if st.button("Train & Compare Models"):
            with st.spinner("Training model sedang berjalan..."):
                X_train = st.session_state["X_train_scaled"]
                y_train = st.session_state["y_train"]
                X_test = st.session_state["X_test_scaled"]
                y_test = st.session_state["y_test"]

                models = {
                    "Logistic Regression": LogisticRegression(C=C_param, solver="liblinear", max_iter=1000),
                    "Random Forest": RandomForestClassifier(n_estimators=rf_estimators, random_state=42),
                    "Gradient Boosting": GradientBoostingClassifier(n_estimators=gb_estimators, random_state=42),
                }

                results = []
                trained_models = {}
                for model_name, model in models.items():
                    model.fit(X_train, y_train)
                    r2 = model_r2_score(model, X_test, y_test)
                    results.append({"Model": model_name, "R2": r2})
                    trained_models[model_name] = model

                result_df = pd.DataFrame(results).sort_values("R2", ascending=False).reset_index(drop=True)
                best_model_name = result_df.iloc[0]["Model"]
                best_model = trained_models[best_model_name]

                st.session_state["trained_models"] = trained_models
                st.session_state["trained_model"] = best_model
                st.session_state["best_model_name"] = best_model_name

                st.success(f"Training selesai. Model terbaik (R² tertinggi): {best_model_name}")
                st.dataframe(result_df, use_container_width=True)

                fig_r2, ax_r2 = plt.subplots(figsize=(9, 4.5))
                sns.barplot(data=result_df, x="Model", y="R2", palette="viridis", ax=ax_r2)
                ax_r2.set_title("Perbandingan Model Berdasarkan R²")
                ax_r2.set_ylabel("R²")
                st.pyplot(fig_r2)

                st.subheader("Feature Importance")
                imp_df, imp_label = extract_feature_importance(best_model, st.session_state["features_list"])
                if imp_df is not None:
                    st.caption(f"Feature importance dari model terbaik: {best_model_name}")
                    st.dataframe(imp_df.head(10), use_container_width=True)

                    fig_imp, ax_imp = plt.subplots(figsize=(9, 5.5))
                    sns.barplot(data=imp_df.head(10), y="Feature", x="Importance", palette="magma", ax=ax_imp)
                    ax_imp.set_title(f"Top 10 Feature Importance ({best_model_name})")
                    ax_imp.set_xlabel(imp_label)
                    st.pyplot(fig_imp)
                else:
                    st.warning("Model terbaik tidak menyediakan atribut feature importance.")


elif menu == "🎯 Prediction Demo":
    st.title("Prediction Demo")
    st.write("Simulasi prediksi default untuk data nasabah baru.")

    source = st.radio(
        "Pilih sumber model:",
        ["Model dari file logistic_model.pkl", "Model hasil training (best R²)"],
    )

    model = None
    scaler = None

    if source == "Model dari file logistic_model.pkl":
        try:
            model = joblib.load("logistic_model.pkl")
            scaler = StandardScaler()
            X = df_clean.drop("default", axis=1)
            scaler.fit(X)
            st.success("Model logistic_model.pkl berhasil dimuat.")
        except Exception:
            st.error("Model logistic_model.pkl tidak ditemukan atau gagal dimuat.")
    else:
        if "trained_model" in st.session_state and "scaler_used" in st.session_state:
            model = st.session_state["trained_model"]
            scaler = st.session_state["scaler_used"]
            best_name = st.session_state.get("best_model_name", "Model Training")
            st.success(f"Menggunakan model hasil training: {best_name}")
        else:
            st.warning("Belum ada model training. Silakan train di menu Training Model.")

    if model is not None and scaler is not None:
        st.subheader("Input Data Nasabah")

        with st.form("demo_form"):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                limit_bal = st.number_input("LIMIT_BAL", min_value=10000, value=50000)
                sex = st.selectbox("SEX (1=Male, 2=Female)", [1, 2])
                education = st.selectbox("EDUCATION (1=Grad, 2=Univ, 3=HS, 4=Others)", [1, 2, 3, 4])
                marriage = st.selectbox("MARRIAGE (1=Married, 2=Single, 3=Others)", [1, 2, 3])
                age = st.number_input("AGE", min_value=21, value=30)

            with col_b:
                pay_1 = st.slider("PAY_1 (Sep)", -2, 8, 0)
                pay_2 = st.slider("PAY_2 (Aug)", -2, 8, 0)
                pay_3 = st.slider("PAY_3 (Jul)", -2, 8, 0)
                pay_4 = st.slider("PAY_4 (Jun)", -2, 8, 0)
                pay_5 = st.slider("PAY_5 (May)", -2, 8, 0)
                pay_6 = st.slider("PAY_6 (Apr)", -2, 8, 0)

            with col_c:
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

            submit = st.form_submit_button("Prediksi")

        if submit:
            input_dict = {
                "LIMIT_BAL": limit_bal,
                "SEX": sex,
                "EDUCATION": education,
                "MARRIAGE": marriage,
                "AGE": age,
                "PAY_1": pay_1,
                "PAY_2": pay_2,
                "PAY_3": pay_3,
                "PAY_4": pay_4,
                "PAY_5": pay_5,
                "PAY_6": pay_6,
                "BILL_AMT1": bill_amt1,
                "BILL_AMT2": bill_amt2,
                "BILL_AMT3": bill_amt3,
                "BILL_AMT4": bill_amt4,
                "BILL_AMT5": bill_amt5,
                "BILL_AMT6": bill_amt6,
                "PAY_AMT1": pay_amt1,
                "PAY_AMT2": pay_amt2,
                "PAY_AMT3": pay_amt3,
                "PAY_AMT4": pay_amt4,
                "PAY_AMT5": pay_amt5,
                "PAY_AMT6": pay_amt6,
            }
            input_df = pd.DataFrame([input_dict])
            scaled_input = scaler.transform(input_df)

            pred = model.predict(scaled_input)[0]
            prob = model.predict_proba(scaled_input)[0][1]

            if pred == 1:
                st.error(f"Prediksi: Gagal Bayar (Default) | Probabilitas default: {prob * 100:.2f}%")
            else:
                st.success(f"Prediksi: Tidak Default | Probabilitas default: {prob * 100:.2f}%")
