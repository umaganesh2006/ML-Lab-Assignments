import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import random

# ==========================================
# A2: Encoding Functions
# ==========================================
def label_encode(series, mapping):
    """Encodes ordinal categorical variables based on a provided dictionary."""
    return series.map(mapping)

def one_hot_encode(df, column):
    """Performs one-hot encoding for nominal categorical variables."""
    return pd.get_dummies(df, columns=[column], dtype=int)

# ==========================================
# A4: Minkowski Distance
# ==========================================
def calc_minkowski_distance(v1, v2, p):
    """Calculates Minkowski distance between two vectors for a given p."""
    # Ensure vectors are numpy arrays for element-wise operations
    v1, v2 = np.array(v1), np.array(v2)
    return np.sum(np.abs(v1 - v2)**p)**(1/p)

# ==========================================
# A7: Vector Mathematics
# ==========================================
def custom_dot_product(v1, v2):
    """Calculates the dot product of two vectors."""
    return sum(x * y for x, y in zip(v1, v2))

def custom_vector_length(v):
    """Calculates the Euclidean norm (length) of a vector."""
    return sum(x**2 for x in v)**0.5

# ==========================================
# A8: Statistical Functions (Matrix-based)
# ==========================================
def custom_mean_vector(matrix):
    """Calculates the mean for each column in a 2D matrix."""
    return [sum(col)/len(col) for col in zip(*matrix)]

def custom_variance_vector(matrix):
    """Calculates the variance for each column in a 2D matrix."""
    means = custom_mean_vector(matrix)
    return [sum((x - m)**2 for x in col)/len(col) for col, m in zip(zip(*matrix), means)]

def custom_std_vector(matrix):
    """Calculates the standard deviation for each column in a 2D matrix."""
    return [var**0.5 for var in custom_variance_vector(matrix)]

# ==========================================
# A11: K-Means Algorithm
# ==========================================
def assign_clusters(data, centroids):
    """Assigns each data point to the closest centroid using Euclidean distance."""
    clusters = [[] for _ in range(len(centroids))]
    for point in data:
        # p=2 for Euclidean distance
        distances = [calc_minkowski_distance(point, c, 2) for c in centroids]
        closest_index = distances.index(min(distances))
        clusters[closest_index].append(point)
    return clusters

def update_centroids(clusters, num_features):
    """Calculates new centroids as the mean of the assigned cluster points."""
    new_centroids = []
    for cluster in clusters:
        if not cluster: # Handle empty clusters
            new_centroids.append([0.0] * num_features)
        else:
            new_centroids.append(custom_mean_vector(cluster))
    return new_centroids

def basic_kmeans(data, k, max_iterations=100):
    """Implementation of Basic K-means algorithm [Algorithm 8.1]."""
    # 1: Select K points as initial centroids (random selection)
    indices = random.sample(range(len(data)), k)
    centroids = [data[i] for i in indices]
    
    for _ in range(max_iterations):
        # 3: Form K clusters
        clusters = assign_clusters(data, centroids)
        # 4: Recompute centroid
        new_centroids = update_centroids(clusters, len(data[0]))
        
        # 5: Check convergence
        # If centroids haven't changed, break
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
        
    return clusters, centroids

# ==========================================
# Main Execution Block
# ==========================================
def main():
    # Load dataset
    file_path = "Lab Session Data (1).xlsx"
    df = pd.read_excel(file_path, sheet_name="marketing_campaign")
    
    # Preprocessing: drop completely null rows for clean execution
    df = df.dropna()

    # --- A2 & A3: Encoding & Dimensionality ---
    print("=== A3: Encoding & Dimensionality ===")
    print(f"Original Dimensionality (Columns): {df.shape[1]}")
    
    edu_mapping = {'Basic': 0, '2n Cycle': 1, 'Graduation': 2, 'Master': 3, 'PhD': 4}
    df['Education'] = label_encode(df['Education'], edu_mapping)
    df_encoded = one_hot_encode(df, 'Marital_Status')
    
    print(f"New Dimensionality (Columns): {df_encoded.shape[1]}\n")

    # --- A4, A5, A6: Minkowski Distance ---
    print("=== A5 & A6: Minkowski Distances ===")
    # Filter for numeric data to do math on
    numeric_df = df_encoded.select_dtypes(include=np.number)
    v1 = numeric_df.iloc[0].values
    v2 = numeric_df.iloc[1].values
    
    p_values = list(range(1, 11))
    custom_dists = [calc_minkowski_distance(v1, v2, p) for p in p_values]
    scipy_dists = [distance.minkowski(v1, v2, p) for p in p_values]
    
    for p, c_dist, s_dist in zip(p_values[:3], custom_dists[:3], scipy_dists[:3]):
         print(f"p={p} | Custom: {c_dist:.2f} | Scipy: {s_dist:.2f}")
    print("...")
    
    # Plotting A5
    plt.figure(figsize=(8, 4))
    plt.plot(p_values, custom_dists, marker='o', linestyle='-', color='b')
    plt.title("Minkowski Distance vs. Order Parameter (p)")
    plt.xlabel("p value")
    plt.ylabel("Distance")
    plt.grid(True)
    plt.show()

    # --- A7: Vector Math ---
    print("\n=== A7: Vector Dot Product & Length ===")
    print(f"Custom Dot Product: {custom_dot_product(v1, v2):.2f}")
    print(f"Numpy Dot Product:  {np.dot(v1, v2):.2f}")
    print(f"Custom Length (v1): {custom_vector_length(v1):.2f}")
    print(f"Numpy Length (v1):  {np.linalg.norm(v1):.2f}\n")

    # --- A8 & A9: Statistics Comparison ---
    print("=== A8 & A9: Mean & Standard Deviation ===")
    data_matrix = numeric_df.values
    
    # Selecting a specific index (e.g., 'Income' column which is index 3) for display
    test_col_idx = 3 
    c_mean = custom_mean_vector(data_matrix)
    n_mean = np.mean(data_matrix, axis=0)
    
    c_std = custom_std_vector(data_matrix)
    n_std = np.std(data_matrix, axis=0)
    
    print(f"Feature Index {test_col_idx} Mean -> Custom: {c_mean[test_col_idx]:.2f} | Numpy: {n_mean[test_col_idx]:.2f}")
    print(f"Feature Index {test_col_idx} Std  -> Custom: {c_std[test_col_idx]:.2f} | Numpy: {n_std[test_col_idx]:.2f}\n")

    # --- A10: Histogram & Density ---
    print("=== A10: Feature Density (Income) ===")
    income_data = df['Income'].dropna().values
    inc_mean = np.mean(income_data)
    inc_var = np.var(income_data)
    print(f"Income Mean: {inc_mean:.2f}")
    print(f"Income Variance: {inc_var:.2f}")
    
    plt.figure(figsize=(8, 4))
    plt.hist(income_data, bins=30, color='skyblue', edgecolor='black')
    plt.title("Histogram of Income Distribution")
    plt.xlabel("Income")
    plt.ylabel("Frequency")
    plt.show()

    # --- A11: K-Means Application ---
    print("\n=== A11: K-Means Execution ===")
    # Using a small subset of features (e.g. Income and MntWines) for a quick demonstration
    kmeans_data = df[['Income', 'MntWines']].dropna().values.tolist()
    
    # Run custom k-means
    k = 3
    final_clusters, final_centroids = basic_kmeans(kmeans_data, k)
    
    print(f"K-Means completed with k={k}.")
    for i, centroid in enumerate(final_centroids):
        print(f"Cluster {i+1} Centroid: {centroid}")
        print(f"Cluster {i+1} Size: {len(final_clusters[i])} items")

if __name__ == "__main__":
    main()