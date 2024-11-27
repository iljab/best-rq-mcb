import librosa
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import umap.umap_ as umap
import os
import pickle
import seaborn as sns
from tqdm import tqdm


def extract_features(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=400)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_delta = np.mean(librosa.feature.delta(mfcc), axis=1)

    # Spectral Contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=400)
    spectral_contrast_mean = np.mean(spectral_contrast, axis=1)

    # Spectral Roll-off
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=400)
    spectral_rolloff_mean = np.mean(spectral_rolloff)

    # Zero-Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)

    feature_vector = np.hstack([mfcc_mean, mfcc_delta, spectral_contrast_mean, spectral_rolloff_mean, zcr_mean])

    return feature_vector


def process_directory(directory_path):
    filenames = []
    for root, _, files in os.walk(directory_path):
        for file in tqdm(files, desc="Processing audio files", unit="file"):
            if file.endswith('.flac') or file.endswith('.wav'):
                file_path = os.path.join(root, file)
                feature_vector = extract_features(file_path)
                features.append(feature_vector)
                filenames.append(file_path)
    return np.array(features), filenames


if __name__ == "__main__":
    dataset_path = '/data/LibriSpeech/train'
    dataset_valid_path = '/data/LibriSpeech/dev-clean'
    n_clusters = 6
    out_file = 'clustering-results-train-clean-n_clusters-6.csv'
    out_valid_file = 'clustering-results-dev-clean-n_clusters-6.csv'

    features, filenames = process_directory(dataset_path)

    kmeans = KMeans(n_clusters=n_clusters, random_state=1234)
    clusters = kmeans.fit_predict(features)

    with open('clustering-kmeans-model.pkl', 'wb') as f:
        pickle.dump(kmeans, f)

    with open(out_file, 'w') as f:
        for i in range(len(filenames)):
            f.write(f"{filenames[i]},{clusters[i]}\n")

    features_val, filenames_val = process_directory(dataset_valid_path)
    clusters_val = kmeans.predict(features_val)

    with open(out_valid_file, 'w') as f:
        for i in range(len(filenames_val)):
            f.write(f"{filenames_val[i]},{clusters_val[i]}\n")

