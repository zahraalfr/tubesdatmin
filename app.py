import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Konfigurasi dasar Streamlit
st.set_page_config(page_title="Customer Experience Analysis", layout="wide")
st.title("📊 Analisis Data Mining - Customer Experience")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("customer_experience_data.csv")

dataset = load_data()

# -----------------------------
# SECTION 1: Data Understanding
# -----------------------------
st.header("1. Data Understanding")

st.subheader("Preview Data")
st.dataframe(dataset.head())

st.subheader("Statistik Deskriptif")
st.write(dataset.describe())

st.subheader("Informasi Dataset")
buffer = []
dataset.info(buf=buffer.append)
st.text('\n'.join(buffer))

# -----------------------------
# SECTION 2: Histogram Distribusi
# -----------------------------
st.header("2. Histogram Distribusi Data")
numeric_cols = dataset.select_dtypes(include=np.number).columns.tolist()

for col in numeric_cols:
    fig, ax = plt.subplots()
    sns.histplot(dataset[col], kde=True, bins=30, ax=ax)
    ax.set_title(f"Distribusi: {col}")
    st.pyplot(fig)

# -----------------------------
# SECTION 3: Boxplot (Outlier)
# -----------------------------
st.header("3. Boxplot - Deteksi Outlier")

for col in numeric_cols:
    fig, ax = plt.subplots()
    sns.boxplot(x=dataset[col], ax=ax)
    ax.set_title(f"Boxplot: {col}")
    st.pyplot(fig)

# -----------------------------
# SECTION 4: K-Means Clustering
# -----------------------------
st.header("4. K-Means Clustering")

drop_cols = ['Customer_ID', 'Gender', 'Location', 'Retention_Status']
data_for_cluster = dataset.drop(columns=[col for col in drop_cols if col in dataset.columns], errors='ignore')
X_cluster = data_for_cluster.select_dtypes(include=np.number)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

dataset['Cluster'] = cluster_labels

# Visualisasi K-Means
fig, ax = plt.subplots()
sns.scatterplot(x=X_cluster.iloc[:, 0], y=X_cluster.iloc[:, 1], hue=cluster_labels, palette='viridis', ax=ax)
ax.set_title("Visualisasi K-Means Clustering")
st.pyplot(fig)

# Informasi tambahan
st.markdown(f"*Inertia:* {kmeans.inertia_:.2f}")
sil_score = silhouette_score(X_scaled, cluster_labels)
st.markdown(f"*Silhouette Score:* {sil_score:.2f}")
st.markdown("*Cluster Centers:*")
st.dataframe(pd.DataFrame(kmeans.cluster_centers_, columns=X_cluster.columns))

# -----------------------------
# SECTION 5: Logistic Regression
# -----------------------------
st.header("5. Logistic Regression")

if 'Retention_Status_Encoded' in dataset.columns:
    y = dataset['Retention_Status_Encoded']
    X = dataset.drop(columns=['Retention_Status_Encoded', 'Customer_ID', 'Gender', 'Location', 'Retention_Status', 'Cluster'], errors='ignore')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # Classification Report
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=False)
    st.text(report)
else:
    st.warning("Kolom 'Retention_Status_Encoded' tidak ditemukan pada dataset.")