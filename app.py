import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Konfigurasi dasar Streamlit
st.set_page_config(page_title="Customer Experience App", layout="wide")
st.title("📊 Customer Experience Analysis")

# Upload file
uploaded_file = st.file_uploader("📂 Unggah dataset CSV Anda", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Dataset Awal")
    st.write(df.head())

    # Data cleaning
    df_cleaned = df.dropna().drop_duplicates()
    if 'Customer_ID' in df_cleaned.columns:
        df_cleaned = df_cleaned.drop(columns=['Customer_ID'], errors='ignore')

    # Pastikan kolom encoding ada
    encoded_cols = ['Gender_Encoded', 'Location_Encoded', 'Retention_Status_Encoded']
    for col in encoded_cols:
        if col not in df_cleaned.columns:
            st.warning(f"❗ Kolom '{col}' tidak ditemukan dalam dataset.")
            st.stop()

    X_features = df_cleaned.drop(columns=['Gender', 'Location', 'Retention_Status'], errors='ignore')
    X = X_features.drop(columns=['Retention_Status_Encoded'], errors='ignore')
    y = df_cleaned['Retention_Status_Encoded']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Tabs untuk dua analisis
    tab1, tab2 = st.tabs(["🔵 K-Means Clustering", "🟢 Logistic Regression"])

    with tab1:
        st.subheader("🔵 Visualisasi K-Means Clustering")
        k = st.slider("🔢 Pilih jumlah klaster", 2, 10, 3)
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)

        # PCA untuk reduksi dimensi
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(X_scaled)
        centroids = pca.transform(kmeans.cluster_centers_)

        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap='viridis', alpha=0.6)
        ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=150, marker='X', label='Centroid')

        # Tambahkan panah dari centroid ke titik random
        for i, center in enumerate(centroids):
            cluster_points = reduced_data[clusters == i]
            if len(cluster_points) > 0:
                sample_point = cluster_points[0]
                ax.annotate('', xy=sample_point, xytext=center,
                            arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
                ax.annotate(f"Cluster {i}", (center[0], center[1]), textcoords="offset points",
                            xytext=(0, 10), ha='center', color='red')

        ax.set_title("Visualisasi Klaster K-Means dengan PCA dan Panah Centroid")
        ax.set_xlabel("Komponen Utama 1")
        ax.set_ylabel("Komponen Utama 2")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    with tab2:
        st.subheader("🟢 Prediksi dengan Logistic Regression")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression(max_iter=300)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        st.success(f"✅ Akurasi Model: {accuracy:.2f}")

        # Interpretasi hasil prediksi
        pred_df = pd.DataFrame(X_test.copy())
        pred_df['Prediksi'] = y_pred
        pred_df['Interpretasi'] = pred_df['Prediksi'].apply(
            lambda x: '✨ Pelanggan Loyal' if x == 1 else '😔 Belum Loyal')
        st.write("📌 Contoh Hasil Prediksi dan Interpretasi:")
        st.dataframe(pred_df[['Prediksi', 'Interpretasi']].head(10))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig2, ax2 = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax2)
        ax2.set_title("🧩 Confusion Matrix")
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Actual")
        st.pyplot(fig2)

        # Classification Report (opsional)
        with st.expander("📋 Laporan Klasifikasi Lengkap"):
            report = classification_report(y_test, y_pred)
            st.text(report)

