import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Load Dataset
df = pd.read_csv("Mall_Customers.csv")

# Display Dataset Information
print("========== DATASET PREVIEW ==========")
print(df.head())

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== COLUMN NAMES ==========")
print(df.columns)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# -----------------------------------
# GRAPH 1: Age Distribution
# -----------------------------------
plt.figure(figsize=(8,5))
sns.histplot(df['Age'], bins=20, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.savefig('age_distribution.png')
plt.show()

# -----------------------------------
# GRAPH 2: Gender Distribution
# -----------------------------------
plt.figure(figsize=(6,5))
sns.countplot(x='Gender', data=df)
plt.title('Gender Distribution')
plt.savefig('gender_distribution.png')
plt.show()

# -----------------------------------
# Features for Clustering
# -----------------------------------
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# -----------------------------------
# GRAPH 3: Elbow Method
# -----------------------------------
wcss = []

for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i,
        init='k-means++',
        random_state=42,
        n_init=10
    )
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1,11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.savefig('elbow_method.png')
plt.show()

# -----------------------------------
# K-Means Clustering
# -----------------------------------
kmeans = KMeans(
    n_clusters=5,
    init='k-means++',
    random_state=42,
    n_init=10
)

df['Cluster'] = kmeans.fit_predict(X)

# -----------------------------------
# GRAPH 4: Customer Segmentation
# -----------------------------------
plt.figure(figsize=(10,6))

sns.scatterplot(
    data=df,
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='Cluster',
    palette='Set1',
    s=100
)

plt.scatter(
    kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1],
    s=300,
    c='black',
    marker='X',
    label='Centroids'
)

plt.title('Customer Segmentation using K-Means')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.savefig('customer_segmentation.png')
plt.show()

# -----------------------------------
# Cluster Summary
# -----------------------------------
print("\n========== CLUSTER SUMMARY ==========")
print(
    df.groupby('Cluster')[
        ['Annual Income (k$)', 'Spending Score (1-100)']
    ].mean()
)

# Save Output
df.to_csv("Customer_Segments_Output.csv", index=False)

print("\nProject Completed Successfully!")
print("Output saved as Customer_Segments_Output.csv")