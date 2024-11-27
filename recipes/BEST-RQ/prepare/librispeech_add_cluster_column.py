import pandas as pd

if __name__ == '__main__':
    # speechbrain prepared file
    file1_path = "../results/bestrq/csv/train-clean-100.csv"

    # cluster file
    file2_path = "./results-clustering/librispeech-train-clean-100-clusters-6/clustering-results-train-clean-100-n_clusters-6.csv"
    output_path = "../recipes/BEST-RQ/results/bestrq/csv/train-clean-100-w-cluster.csv"

    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path, header=None, names=['wav', 'cluster'])

    merged_df = pd.merge(df1, df2, on='wav', how='left')
    merged_df = merged_df.dropna(subset=['cluster'])
    merged_df.to_csv(output_path, index=False)
