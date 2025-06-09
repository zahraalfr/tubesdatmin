import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, confusion_matrix, classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Konfigurasi halaman
st.set_page_config(layout="wide")
st.title("📊 Analisis Customer Experience")

# Upload file
uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset berhasil dimuat!")

    tab1, tab2, tab3 = st.tabs(["Data Understanding", "KMeans Clustering", "Logistic Regression"])

    with tab1:
        st.header("🧾 Data Understanding")
        st.write("Statistik Deskriptif:")
        st.dataframe(df.describe())

        num_cols = df.select_dtypes(include=['int64', 'float64']).columns

        for col in num_cols:
            st.markdown(f"#### Histogram: {col}")
            fig, ax = plt.subplots()
            sns.histplot(df[col].dropna(), kde=True, ax=ax, bins=30)
            st.pyplot(fig)

            st.markdown(f"#### Boxplot: {col}")
            fig, ax = plt.subplots()
            sns.boxplot(x=df[col], ax=ax)
            st.pyplot(fig)

    with tab2:
        st.header("📌 KMeans Clustering")
        fitur_segmentasi = ['Age', 'Num_Interactions', 'Products_Purchased', 'Time_Spent_on_Site']
        if all(col in df.columns for col in fitur_segmentasi):
            data_segmentasi = df[fitur_segmentasi].fillna(0)
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_segmentasi)

            silhouette_scores = {}
            for k in range(2, 7):
                model = KMeans(n_clusters=k, random_state=42)
                labels = model.fit_predict(data_scaled)
                silhouette_scores[k] = silhouette_score(data_scaled, labels)

            best_k = max(silhouette_scores, key=silhouette_scores.get)
            kmeans_final = KMeans(n_clusters=best_k, random_state=42)
            cluster_labels = kmeans_final.fit_predict(data_scaled)
            df['Cluster'] = cluster_labels

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
            ax.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.error("Kolom yang dibutuhkan tidak tersedia dalam dataset.")

    with tab3:
        st.header("📈 Logistic Regression")
        if 'Retention_Status_Encoded' in df.columns:
            try:
                X = df.drop(columns=['Retention_Status_Encoded', 'Cluster'], errors='ignore')
                X = X.select_dtypes(include=['int64', 'float64']).fillna(0)
                y = df['Retention_Status_Encoded']

                scaler = StandardScaler()
                X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)
                smote = SMOTE(random_state=42)
                X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

                model = LogisticRegression(max_iter=1000, class_weight='balanced')
                model.fit(X_train_res, y_train_res)
                y_pred = model.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                st.metric(label="Akurasi Model", value=f"{acc:.2%}")

                cm = confusion_matrix(y_test, y_pred)
                st.markdown("#### Confusion Matrix")
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Churned", "Retained"], yticklabels=["Churned", "Retained"], ax=ax)
                ax.set_ylabel("Aktual")
                ax.set_xlabel("Prediksi")
                st.pyplot(fig)

                st.markdown("#### Laporan Klasifikasi")
                report = classification_report(y_test, y_pred, output_dict=True, target_names=["Churned", "Retained"])
                st.dataframe(pd.DataFrame(report).transpose().round(2))

                st.markdown("#### Penjelasan")
                st.info("""
                Model Logistic Regression digunakan untuk memprediksi apakah pelanggan akan bertahan (retained) atau berhenti (churned).
                Visualisasi Confusion Matrix membantu melihat performa model dalam memisahkan dua kelas.
                Laporan klasifikasi menunjukkan metrik penting seperti precision, recall, dan F1-score.
                """)
            except Exception as e:
                st.error(f"Terjadi kesalahan saat menjalankan Logistic Regression: {e}")
        else:
            st.warning("Kolom 'Retention_Status_Encoded' tidak ditemukan dalam dataset.")

else:
    st.info("Silakan upload file dataset CSV terlebih dahulu.")
