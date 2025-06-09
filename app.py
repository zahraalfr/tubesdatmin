import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from imblearn.over_sampling import SMOTE

# Konfigurasi awal
st.set_page_config(layout="wide")
st.title("Dashboard Analisis Customer Experience")

# === UPLOAD DATASET ===
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Pilih file CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset berhasil dimuat!")

    tab1, tab2, tab3 = st.tabs(["Data Understanding", "K-Means Clustering", "Logistic Regression"])

    with tab1:
        st.subheader("📊 Data Understanding")
        st.write("Statistik Deskriptif:")
        st.dataframe(df.describe())

        num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

        for col in num_cols:
            st.markdown(f"#### Distribusi: {col}")
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, bins=30, ax=ax)
            st.pyplot(fig)

            st.markdown(f"#### Boxplot: {col}")
            fig, ax = plt.subplots()
            sns.boxplot(x=df[col], ax=ax)
            st.pyplot(fig)

    with tab2:
        st.subheader("📌 Segmentasi Pelanggan - KMeans Clustering")
        fitur_segmentasi = ['Age', 'Num_Interactions', 'Products_Purchased', 'Time_Spent_on_Site']
        data_segmentasi = df[fitur_segmentasi]

        # Normalisasi
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_segmentasi)

        # Tentukan jumlah cluster terbaik
        silhouette_scores = {}
        for k in range(2, 7):
            model = KMeans(n_clusters=k, random_state=42)
            labels = model.fit_predict(data_scaled)
            silhouette_scores[k] = silhouette_score(data_scaled, labels)

        best_k = max(silhouette_scores, key=silhouette_scores.get)

        kmeans_final = KMeans(n_clusters=best_k, random_state=42)
        cluster_labels = kmeans_final.fit_predict(data_scaled)
        df['Cluster'] = cluster_labels

        # Visualisasi
        st.markdown(f"### Visualisasi KMeans (k={best_k})")
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = sns.scatterplot(
            x=df['Num_Interactions'],
            y=df['Time_Spent_on_Site'],
            hue=df['Cluster'],
            palette='tab10',
            s=60,
            alpha=0.6,
            ax=ax
        )

        centroids_unscaled = scaler.inverse_transform(kmeans_final.cluster_centers_)
        ni_idx = fitur_segmentasi.index('Num_Interactions')
        ts_idx = fitur_segmentasi.index('Time_Spent_on_Site')
        centroids_plot = centroids_unscaled[:, [ni_idx, ts_idx]]

        for i, (x, y) in enumerate(centroids_plot):
            ax.annotate(
                f'Cluster {i}', xy=(x, y), xytext=(x + 0.8, y + 2),
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.7),
                arrowprops=dict(arrowstyle="->", color='gray', lw=1)
            )

        ax.set_title("Segmentasi Pelanggan Berdasarkan Aktivitas Interaksi dan Waktu")
        ax.set_xlabel("Jumlah Interaksi Pelanggan")
        ax.set_ylabel("Waktu yang Dihabiskan di Situs (menit)")
        ax.grid(True)
        ax.legend(title='Cluster')
        st.pyplot(fig)

        st.markdown(f"Silhouette Score: **{silhouette_score(data_scaled, cluster_labels):.4f}**")

    with tab3:
        st.subheader("📈 Prediksi Retensi Pelanggan - Logistic Regression")

        X = df.drop(columns=['Retention_Status_Encoded', 'Cluster'], errors='ignore')
        y = df['Retention_Status_Encoded']

        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, stratify=y, random_state=42
        )

        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        model = LogisticRegression(max_iter=1000, class_weight='balanced')
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        st.markdown(f"**Akurasi Model:** {acc:.2%}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        st.markdown("#### Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Churned", "Retained"], yticklabels=["Churned", "Retained"], ax=ax)
        ax.set_ylabel("Aktual")
        ax.set_xlabel("Prediksi")
        st.pyplot(fig)

        # Classification report
        st.markdown("#### Laporan Klasifikasi")
        report = classification_report(y_test, y_pred, target_names=["Churned", "Retained"], output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose().round(2))

        st.markdown("#### Penjelasan:")
        st.info("""
        Model ini digunakan untuk memprediksi apakah pelanggan akan tetap bertahan atau berhenti.
        Confusion matrix menunjukkan performa prediksi aktual vs prediksi.
        Laporan klasifikasi memberikan metrik akurasi, presisi, recall, dan F1-score untuk mengevaluasi model.
        """)

else:
    st.warning("📁 Silakan unggah file CSV terlebih dahulu.")
