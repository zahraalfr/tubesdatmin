import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Konfigurasi Streamlit
st.set_page_config(page_title="Customer Experience App", layout="wide")
st.title("📊 Customer Experience Analysis")

# Upload Dataset
uploaded_file = st.file_uploader("📂 Unggah dataset CSV Anda", type=["csv"])
if uploaded_file is not None:
    dataset = pd.read_csv(uploaded_file)

    # ------------------------------------
    # SECTION 1: Data Understanding
    # ------------------------------------
    st.header("1. 📘 Data Understanding")

    st.subheader("📄 Preview Data")
    st.dataframe(dataset.head())

    st.subheader("📊 Statistik Deskriptif")
    st.write(dataset.describe())

    st.subheader("🔍 Informasi Dataset")
    buffer = []
    dataset.info(buf=buffer.append)
    st.text('\n'.join(buffer))

    # ------------------------------------
    # SECTION 2: Histogram Distribusi
    # ------------------------------------
    st.header("2. 📈 Histogram Distribusi")
    numeric_cols = dataset.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.histplot(dataset[col], kde=True, bins=30, ax=ax)
        ax.set_title(f"Distribusi: {col}")
        st.pyplot(fig)

    # ------------------------------------
    # SECTION 3: Boxplot (Outlier)
    # ------------------------------------
    st.header("3. 🧪 Boxplot - Deteksi Outlier")
    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.boxplot(x=dataset[col], ax=ax)
        ax.set_title(f"Boxplot: {col}")
        st.pyplot(fig)

    # ------------------------------------
    # SECTION 4 & 5: Tabs K-Means dan Logistic Regression
    # ------------------------------------
    st.header("4 & 5. 🔀 Clustering & Classification")
    tab1, tab2 = st.tabs(["🔵 K-Means Clustering", "🟢 Logistic Regression"])

    with tab1:
        st.subheader("🔵 Visualisasi K-Means Clustering")

        drop_cols = ['Customer_ID', 'Gender', 'Location', 'Retention_Status']
        data_for_cluster = dataset.drop(columns=[col for col in drop_cols if col in dataset.columns], errors='ignore')
        X_cluster = data_for_cluster.select_dtypes(include=np.number)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cluster)

        k = st.slider("Pilih jumlah klaster (k)", 2, 10, 3)
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)

        dataset['Cluster'] = cluster_labels

        # Visualisasi dengan PCA dan panah
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(X_scaled)
        centroids = pca.transform(kmeans.cluster_centers_)

        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
        ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=150, marker='X', label='Centroid')

        for i, center in enumerate(centroids):
            cluster_points = reduced_data[cluster_labels == i]
            if len(cluster_points) > 0:
                sample_point = cluster_points[0]
                ax.annotate('', xy=sample_point, xytext=center,
                            arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
                ax.annotate(f"Cluster {i}", (center[0], center[1]), textcoords="offset points",
                            xytext=(0, 10), ha='center', color='red')

        ax.set_title("Visualisasi Klaster K-Means dengan PCA dan Panah")
        ax.set_xlabel("Komponen Utama 1")
        ax.set_ylabel("Komponen Utama 2")
        ax.legend()
        st.pyplot(fig)

        st.markdown(f"**Inertia:** {kmeans.inertia_:.2f}")
        sil_score = silhouette_score(X_scaled, cluster_labels)
        st.markdown(f"**Silhouette Score:** {sil_score:.2f}")
        st.markdown("**Cluster Centers (dalam skala PCA):**")
        st.dataframe(pd.DataFrame(centroids, columns=["PCA1", "PCA2"]))

    with tab2:
        st.subheader("🟢 Prediksi dengan Logistic Regression")

        # Pastikan label encoding ada
        if 'Retention_Status_Encoded' in dataset.columns:
            X = dataset.drop(columns=['Retention_Status_Encoded', 'Customer_ID', 'Gender', 'Location', 'Retention_Status', 'Cluster'], errors='ignore')
            y = dataset['Retention_Status_Encoded']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LogisticRegression(max_iter=300)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            st.success(f"✅ Akurasi Model: {accuracy:.2f}")

            # Interpretasi prediksi
            pred_df = pd.DataFrame(X_test.copy())
            pred_df['Prediksi'] = y_pred
            pred_df['Interpretasi'] = pred_df['Prediksi'].apply(lambda x: '✨ Pelanggan Loyal' if x == 1 else '😔 Belum Loyal')
            st.write("📌 Contoh Hasil Prediksi dan Interpretasi:")
            st.dataframe(pred_df[['Prediksi', 'Interpretasi']].head(10))

            cm = confusion_matrix(y_test, y_pred)
            fig2, ax2 = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax2)
            ax2.set_title("🧩 Confusion Matrix")
            ax2.set_xlabel("Predicted")
            ax2.set_ylabel("Actual")
            st.pyplot(fig2)

            with st.expander("📋 Laporan Klasifikasi Lengkap"):
                report = classification_report(y_test, y_pred)
                st.text(report)
        else:
            st.warning("❗ Kolom 'Retention_Status_Encoded' tidak ditemukan di dataset.")


