import pandas as pd
import numpy as np
import time
import unittest

# =====================================================================
# A1: AI-Assisted Modular Functions (Lab 03 Replication)
# =====================================================================

# --- Feature Encoding ---
# Generated with the assistance of ChatGPT (GPT-4)
def ai_label_encode(series, mapping_dict):
    """Encodes ordinal categorical variables using pandas map."""
    return series.map(mapping_dict)

def ai_one_hot_encode(df, column):
    """Performs one-hot encoding using pandas get_dummies."""
    return pd.get_dummies(df, columns=[column], dtype=int)

# --- Distance & Vector Math ---
# Generated with the assistance of Claude 3.5 Sonnet
def ai_minkowski_distance(v1, v2, p):
    """Calculates Minkowski distance using highly optimized NumPy norms."""
    v1, v2 = np.array(v1), np.array(v2)
    if v1.shape != v2.shape:
        raise ValueError("Vector dimensions must match.")
    return np.linalg.norm(v1 - v2, ord=p)

def ai_vector_dot(v1, v2):
    """Calculates dot product using NumPy."""
    return np.dot(v1, v2)

def ai_vector_length(v):
    """Calculates Euclidean length of a vector."""
    return np.linalg.norm(v)

# --- AI-Optimized Vectorized K-Means ---
# Generated with the assistance of ChatGPT (GPT-4) for NumPy broadcasting
def ai_kmeans(data, k, max_iterations=100):
    """
    Optimized K-Means using NumPy broadcasting for distance calculations.
    Replaces slow for-loops with fast matrix operations.
    """
    data = np.array(data)
    # 1. Randomly initialize centroids
    random_indices = np.random.choice(data.shape[0], k, replace=False)
    centroids = data[random_indices]
    
    for _ in range(max_iterations):
        # 2. Vectorized distance calculation & assignment
        # data[:, np.newaxis] broadcasts data to shape (N, 1, Features)
        # centroids broadcasts to (1, K, Features)
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        # 3. Update centroids
        new_centroids = np.array([data[labels == i].mean(axis=0) if np.any(labels == i) 
                                  else data[np.random.choice(data.shape[0])] # Handle empty clusters
                                  for i in range(k)])
        
        # 4. Check convergence
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
        
    return labels, centroids


# =====================================================================
# Legacy Lab 03 K-Means (For Performance Comparison)
# =====================================================================
def manual_minkowski(v1, v2, p=2):
    return sum(abs(x - y)**p for x, y in zip(v1, v2))**(1/p)

def manual_kmeans(data, k, max_iterations=100):
    """Iterative, loop-based K-Means from Lab 03."""
    import random
    data_list = data.tolist() if isinstance(data, np.ndarray) else data
    centroids = random.sample(data_list, k)
    
    for _ in range(max_iterations):
        clusters = [[] for _ in range(k)]
        # Slow iterative assignment
        for point in data_list:
            distances = [manual_minkowski(point, c, 2) for c in centroids]
            clusters[distances.index(min(distances))].append(point)
            
        # Slow iterative update
        new_centroids = []
        for cluster in clusters:
            if not cluster:
                new_centroids.append([0.0] * len(data_list[0]))
            else:
                new_centroids.append([sum(col)/len(col) for col in zip(*cluster)])
                
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return clusters, centroids


# =====================================================================
# A2: Automated Unit Testing 
# Generated with the assistance of Claude 3.5 Sonnet
# =====================================================================
class TestLabFunctions(unittest.TestCase):
    
    def test_minkowski_distance_euclidean(self):
        """Test p=2 (Euclidean) distance."""
        v1 = np.array([0, 0])
        v2 = np.array([3, 4])
        self.assertEqual(ai_minkowski_distance(v1, v2, p=2), 5.0)
        
    def test_minkowski_distance_manhattan(self):
        """Test p=1 (Manhattan) distance."""
        v1 = np.array([0, 0])
        v2 = np.array([3, 4])
        self.assertEqual(ai_minkowski_distance(v1, v2, p=1), 7.0)
        
    def test_vector_dimension_mismatch(self):
        """Test if proper error is raised when dimensions don't match."""
        v1 = np.array([1, 2, 3])
        v2 = np.array([1, 2])
        with self.assertRaises(ValueError):
            ai_minkowski_distance(v1, v2, p=2)
            
    def test_vector_dot_product(self):
        """Test dot product calculation."""
        v1 = np.array([1, 3, -5])
        v2 = np.array([4, -2, -1])
        # 1*4 + 3*-2 + -5*-1 = 4 - 6 + 5 = 3
        self.assertEqual(ai_vector_dot(v1, v2), 3)


# =====================================================================
# A3: Performance Benchmarking
# =====================================================================
def run_performance_comparison():
    print("\n--- A3: K-Means Performance Comparison ---")
    # Generate synthetic dataset simulating the marketing campaign numeric features
    np.random.seed(42)
    sample_data = np.random.rand(2000, 10) * 100 # 2000 rows, 10 features
    k = 3
    
    # 1. Benchmark Manual K-Means
    start_time = time.perf_counter()
    manual_kmeans(sample_data, k, max_iterations=20)
    manual_duration = time.perf_counter() - start_time
    print(f"Manual Lab 03 K-Means Execution Time: {manual_duration:.4f} seconds")
    
    # 2. Benchmark AI-Optimized Vectorized K-Means
    start_time = time.perf_counter()
    ai_kmeans(sample_data, k, max_iterations=20)
    ai_duration = time.perf_counter() - start_time
    print(f"AI-Optimized Vectorized K-Means Execution Time: {ai_duration:.4f} seconds")
    
    # Calculate Speedup
    speedup = manual_duration / ai_duration
    print(f"Performance Gain: AI version is {speedup:.2f}x faster.")


# =====================================================================
# Main Execution Block
# =====================================================================
if __name__ == '__main__':
    # 1. Run Unit Tests (A2)
    print("--- A2: Running Unit Tests ---")
    # We use argv=[''] to prevent unittest from trying to read Jupyter notebook arguments
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    
    # 2. Run Performance Comparison (A3)
    run_performance_comparison()