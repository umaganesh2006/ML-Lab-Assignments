import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import jaccard
import warnings
warnings.filterwarnings('ignore')

# A1: Linear Algebra & Pseudo-Inverse

def task_a1(file_path):
    df = pd.read_excel(file_path, sheet_name="Purchase data").dropna(how='all').dropna(axis=1, how='all')
    X = df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].dropna().values
    y = df['Payment (Rs)'].dropna().values
    
    rank_X = np.linalg.matrix_rank(X)
    X_pinv = np.linalg.pinv(X)
    costs = X_pinv.dot(y)
    
    return X.shape[1], X.shape[0], rank_X, costs

# A3: Statistics & Probability (IRCTC Data)
def custom_mean(arr):
    return sum(arr) / len(arr)

def custom_var(arr):
    m = custom_mean(arr)
    return sum((x - m)**2 for x in arr) / len(arr)

def measure_complexity(func, data, runs=10):
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        func(data)
        times.append(time.perf_counter() - start)
    return np.mean(times)

def task_a3(file_path):
    df = pd.read_excel(file_path, sheet_name="IRCTC Stock Price")
    prices = df['Price'].values
    chg = df['Chg%'].values
    
    np_mean, np_var = np.mean(prices), np.var(prices)
    c_mean, c_var = custom_mean(prices), custom_var(prices)
    
    t_np_mean = measure_complexity(np.mean, prices)
    t_c_mean = measure_complexity(custom_mean, prices)
    
    wed_mean = np.mean(df[df['Day'] == 'Wed']['Price'].values)
    apr_mean = np.mean(df[df['Month'] == 'Apr']['Price'].values)
    
    is_loss = list(map(lambda x: x < 0, chg))
    prob_loss = sum(is_loss) / len(chg)
    
    profit_wed = len(df[(df['Chg%'] > 0) & (df['Day'] == 'Wed')])
    prob_profit_wed = profit_wed / len(df)
    
    total_wed = len(df[df['Day'] == 'Wed'])
    cond_prob = profit_wed / total_wed if total_wed > 0 else 0
    
    return (np_mean, np_var, t_np_mean, t_c_mean, wed_mean, apr_mean, 
            prob_loss, prob_profit_wed, cond_prob, df)

def plot_scatter(df):
    plt.figure(figsize=(8, 5))
    plt.scatter(df['Day'], df['Chg%'], alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.title("A3: Chg% vs Day of the Week")
    plt.show()

# A4 - A9: Thyroid Data Exploration & Similarities

def clean_thyroid_data(file_path):
    df = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")
    df.replace('?', np.nan, inplace=True)
    return df

def task_a4_explore(df):
    num_cols = df.select_dtypes(include=np.number).columns
    stats = {}
    for col in num_cols:
        stats[col] = {'mean': df[col].mean(), 'var': df[col].var(), 
                      'min': df[col].min(), 'max': df[col].max(), 
                      'missing': df[col].isna().sum()}
    return stats

def calculate_similarities(v1_bin, v2_bin, v1_full, v2_full):
    f11 = sum((v1_bin == 1) & (v2_bin == 1))
    f00 = sum((v1_bin == 0) & (v2_bin == 0))
    f01 = sum((v1_bin == 0) & (v2_bin == 1))
    f10 = sum((v1_bin == 1) & (v2_bin == 0))
    
    jc = f11 / (f01 + f10 + f11) if (f01 + f10 + f11) > 0 else 0
    smc = (f11 + f00) / (f00 + f01 + f10 + f11)
    
    cos_sim = np.dot(v1_full, v2_full) / (np.linalg.norm(v1_full) * np.linalg.norm(v2_full))
    return jc, smc, cos_sim

def task_a7_heatmap(df_encoded, n=20):
    subset = df_encoded.iloc[:n].drop('Record ID', axis=1).values
    cos_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            norm_i = np.linalg.norm(subset[i])
            norm_j = np.linalg.norm(subset[j])
            if norm_i > 0 and norm_j > 0:
                cos_matrix[i, j] = np.dot(subset[i], subset[j]) / (norm_i * norm_j)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cos_matrix, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("A7: Cosine Similarity Heatmap (First 20 vectors)")
    plt.show()

def task_a8_a9_impute_normalize(df):
    df_imputed = df.copy()
    num_cols = df_imputed.select_dtypes(include=np.number).columns
    cat_cols = df_imputed.select_dtypes(exclude=np.number).columns
    
    # A8 Imputation
    for col in num_cols:
        df_imputed[col].fillna(df_imputed[col].median(), inplace=True) # Assuming outliers
    for col in cat_cols:
        df_imputed[col].fillna(df_imputed[col].mode()[0], inplace=True)
        
    # A9 Normalization (Min-Max)
    for col in num_cols:
        if col != 'Record ID':
            min_val, max_val = df_imputed[col].min(), df_imputed[col].max()
            if max_val > min_val:
                df_imputed[col] = (df_imputed[col] - min_val) / (max_val - min_val)
    return df_imputed

# Optional Tasks (O1 - O3) Stubs

def task_optional_o1(file_path):
    df = pd.read_excel(file_path, sheet_name="Purchase data").dropna(how='all').dropna(axis=1, how='all')
    X_sq = df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].dropna().values[:3, :]
    y_sq = df['Payment (Rs)'].dropna().values[:3]
    return np.linalg.matrix_rank(X_sq), np.linalg.pinv(X_sq).dot(y_sq)

# Main Execution Block

def main():
    file = "Lab Session Data (1).xlsx"
    
    print("=== A1: PURCHASE DATA ===")
    dim, num_vec, rank_X, costs = task_a1(file)
    print(f"Dimensionality: {dim}\nTotal Vectors: {num_vec}")
    print(f"Rank of Feature Matrix X: {rank_X}")
    print(f"Costs -> Candy: Rs.{costs[0]:.2f}, Mango (kg): Rs.{costs[1]:.2f}, Milk: Rs.{costs[2]:.2f}\n")
    
    print("=== A3: IRCTC STOCK PRICE ===")
    a3_results = task_a3(file)
    print(f"Numpy Mean: {a3_results[0]:.2f} | Numpy Variance: {a3_results[1]:.2f}")
    print(f"Time Numpy Mean: {a3_results[2]:.8f}s | Time Custom Mean: {a3_results[3]:.8f}s")
    print(f"Wednesday Mean: {a3_results[4]:.2f} | April Mean: {a3_results[5]:.2f}")
    print(f"Prob of Loss: {a3_results[6]:.4f}")
    print(f"Prob of Profit on Wed: {a3_results[7]:.4f}")
    print(f"Cond Prob of Profit | Wed: {a3_results[8]:.4f}\n")
    
    print("=== A4-A9: THYROID DATA ===")
    df_thy = clean_thyroid_data(file)
    
    binary_cols = ['on thyroxine', 'query on thyroxine', 'on antithyroid medication', 'sick', 'pregnant']
    v1_bin = df_thy.iloc[0][binary_cols].map({'t': 1, 'f': 0}).values
    v2_bin = df_thy.iloc[1][binary_cols].map({'t': 1, 'f': 0}).values
    
    df_enc = df_thy.copy()
    for c in binary_cols: df_enc[c] = df_enc[c].map({'t': 1, 'f': 0})
    
    s1 = df_enc.iloc[0].drop(['Record ID', 'referral source', 'Condition', 'sex'], errors='ignore')
    s2 = df_enc.iloc[1].drop(['Record ID', 'referral source', 'Condition', 'sex'], errors='ignore')
    
    v1_full = pd.to_numeric(s1, errors='coerce').fillna(0).values
    v2_full = pd.to_numeric(s2, errors='coerce').fillna(0).values
    
    jc, smc, cos = calculate_similarities(v1_bin, v2_bin, v1_full, v2_full)
    print(f"First 2 Vectors -> Jaccard: {jc:.2f} | SMC: {smc:.2f} | Cosine: {cos:.2f}\n")
    
    df_processed = task_a8_a9_impute_normalize(df_thy)
    print(f"Missing Values after Imputation: {df_processed.isna().sum().sum()}")
    
    print("\n=== OPTIONAL (O1) ===")
    sq_rank, sq_costs = task_optional_o1(file)
    print(f"Square Matrix Rank: {sq_rank} | Calculated Costs: {sq_costs}")

if __name__ == "__main__":
    main()