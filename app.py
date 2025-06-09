import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Konfigurasi halaman
st.set_page_config(layout="wide")
st.title("📊 Segmentasi Pelanggan dengan KMeans")

# Upload file
uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset berhasil dimuat!")

    fitur_segmentasi = ['Age', 'Num_Interactions', 'Products_Purchased', 'Time_Spent_on_Site']
    if not all(col in df.columns for col in fitur_segmentasi):
        st.error("Dataset tidak memiliki semua kolom yang dibutuhkan: " + ", ".join(fitur_segmentasi))
    else:
        # Normalisasi data
        scaler = StandardScaler()
        data_segmentasi = df[fitur_segmentasi]
        if data_segmentasi.isnull().any().any():
            st.warning("Data mengandung nilai null. Nilai null akan diisi dengan 0.")
            data_segmentasi = data_segmentasi.fillna(0)
        data_scaled = scaler.fit_transform(data_segmentasi)

        # Tentukan jumlah cluster optimal
        silhouette_scores = {}
        for k in range(2, 7):
            model = KMeans(n_clusters=k, random_state=42)
            labels = model.fit_predict(data_scaled)
            silhouette_scores[k] = silhouette_score(data_scaled, labels)

        best_k = max(silhouette_scores, key=silhouette_scores.get)

        # Model akhir
        kmeans_final = KMeans(n_clusters=best_k, random_state=42)
        cluster_labels = kmeans_final.fit_predict(data_scaled)
        df['Cluster'] = cluster_labels

        # Plot hasil
        st.markdown(f"### Visualisasi KMeans (k={best_k})")
        fig, ax = plt.subplots(figsize=(12, 7))
        scatter = sns.scatterplot(
            x=df['Num_Interactions'],
            y=df['Time_Spent_on_Site'],
            hue=df['Cluster'],
            palette='tab10',
            s=60,
            alpha=0.8,
            ax=ax
        )

        # Anotasi centroid
        centroids_unscaled = scaler.inverse_transform(kmeans_final.cluster_centers_)
        ni_idx = fitur_segmentasi.index('Num_Interactions')
        ts_idx = fitur_segmentasi.index('Time_Spent_on_Site')
        centroids_plot = centroids_unscaled[:, [ni_idx, ts_idx]]

        custom_labels = {
            0: {"xy": centroids_plot[0], "xytext": (centroids_plot[0][0] - 1.0, centroids_plot[0][1])},
            1: {"xy": centroids_plot[1], "xytext": (centroids_plot[1][0] - 2.5, centroids_plot[1][1] - 8)},
            2: {"xy": centroids_plot[2], "xytext": (centroids_plot[2][0] - 1.0, centroids_plot[2][1] - 7)},
            3: {"xy": centroids_plot[3], "xytext": (centroids_plot[3][0] - 2.2, centroids_plot[3][1] + 2)},
            4: {"xy": centroids_plot[4], "xytext": (centroids_plot[4][0] - 2.4, centroids_plot[4][1] - 1)},
            5: {"xy": centroids_plot[5], "xytext": (centroids_plot[5][0] + 1.0, centroids_plot[5][1])},
        }

        for i in range(best_k):
            if i in custom_labels:
                x, y = custom_labels[i]['xy']
                xt, yt = custom_labels[i]['xytext']
                ax.annotate(
                    f'Cluster {i}', xy=(x, y), xytext=(xt, yt),
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.7),
                    arrowprops=dict(arrowstyle="->", color='gray', lw=1)
                )

        ax.set_title("Segmentasi Pelanggan Berdasarkan Aktivitas Interaksi dan Waktu", fontsize=14)
        ax.set_xlabel("Jumlah Interaksi Pelanggan")
        ax.set_ylabel("Waktu yang Dihabiskan di Situs (menit)")
        ax.grid(True)
        ax.legend(title='Kelompok Pelanggan (Cluster)', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        st.pyplot(fig)

else:
    st.info("Silakan upload file dataset CSV terlebih dahulu.")

