import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ✅ Load local dataset (no internet required)
data = pd.read_csv("Mall_Customers.csv")

# ✅ Show sample data
print("Sample data:\n", data.head())

# ✅ Select features for clustering
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# ✅ Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Elbow Method to find optimal clusters
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method to Determine Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

# ✅ Apply KMeans with k=5
kmeans = KMeans(n_clusters=5, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# ✅ Visualize Clusters
colors = ['red', 'blue', 'green', 'orange', 'purple']
plt.figure(figsize=(8, 6))
for i in range(5):
    plt.scatter(X_scaled[data['Cluster'] == i, 0], X_scaled[data['Cluster'] == i, 1],
                c=colors[i], s=100, label=f'Cluster {i}')

# ✅ Plot Centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=300, c='yellow', marker='*', label='Centroids')

plt.title('Customer Segmentation using KMeans')
plt.xlabel('Annual Income (scaled)')
plt.ylabel('Spending Score (scaled)')
plt.legend()
plt.grid(True)
plt.show()

# ✅ Optional: Save clustered data to a new CSV
data.to_csv("Clustered_Mall_Customers.csv", index=False)
print("Clustering complete! Output saved to Clustered_Mall_Customers.csv")
