import faiss
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import gc
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def flatten_embeddings_in_batches(df, embedding_col, batch_size=1000, use_gpu=True):
    flattened_embeddings = []
    total_samples = len(df)
    
    for start in range(0, total_samples, batch_size):
        end = min(start + batch_size, total_samples)
        logging.info(f"Flattening embeddings: batch {start} to {end}")
        batch_embeddings = df[embedding_col].iloc[start:end].values
        if use_gpu:
            batch_embeddings = torch.tensor(np.stack(batch_embeddings)).to('cuda')
            batch_embeddings = batch_embeddings.view(batch_embeddings.size(0), -1)
            batch_embeddings = batch_embeddings.cpu().numpy()
        else:
            batch_embeddings = np.vstack(batch_embeddings)
            batch_embeddings = batch_embeddings.reshape(batch_embeddings.shape[0], -1)
        flattened_embeddings.append(batch_embeddings)
        torch.cuda.empty_cache()
        gc.collect()
    
    return np.vstack(flattened_embeddings)

def faiss_cosine_similarity_search(df, embedding_col, label_col, test_size=0.3, batch_size=100, train_batch_size=5000):
    logging.info("Starting data split into train and test sets")
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

    logging.info("Flattening train embeddings for FAISS processing in batches")
    train_embeddings = flatten_embeddings_in_batches(train_df, embedding_col, use_gpu=True)
    test_embeddings = flatten_embeddings_in_batches(test_df, embedding_col, use_gpu=True)

    train_names = train_df[label_col].values
    test_names = test_df[label_col].values

    logging.info("Normalizing embeddings for cosine similarity")
    faiss.normalize_L2(train_embeddings)
    faiss.normalize_L2(test_embeddings)

    logging.info("Initializing FAISS GPU resources and creating the index")
    res = faiss.StandardGpuResources()
    index_flat = faiss.IndexFlatIP(train_embeddings.shape[1])
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index_flat)

    total_train_samples = len(train_embeddings)
    for start in range(0, total_train_samples, train_batch_size):
        end = min(start + train_batch_size, total_train_samples)
        logging.info(f"Adding train embeddings to index: batch {start} to {end}")
        gpu_index.add(train_embeddings[start:end])
        torch.cuda.empty_cache()
        gc.collect()

    logging.info("Starting batch processing of test samples")
    correct_matches = 0
    total_samples = len(test_embeddings)

    for start in range(0, total_samples, batch_size):
        end = min(start + batch_size, total_samples)
        logging.info(f"Processing test batch: {start} to {end}")
        test_batch = test_embeddings[start:end]

        D, I = gpu_index.search(test_batch, 1)

        for i in range(end - start):
            most_similar_train_index = I[i][0]
            test_name = test_names[start + i]
            train_name = train_names[most_similar_train_index]

            if test_name == train_name:
                correct_matches += 1

        del test_batch, D, I
        torch.cuda.empty_cache()
        gc.collect()

    accuracy = correct_matches / total_samples
    logging.info(f"Accuracy: {accuracy * 100:.2f}%")
    
    logging.info("Clearing GPU index and other variables")
    del gpu_index, train_embeddings, test_embeddings, train_names, test_names
    torch.cuda.empty_cache()
    gc.collect()
    
    return accuracy

df = pd.read_pickle("df_32.pkl")
accuracy = faiss_cosine_similarity_search(df, embedding_col='float32_embeddings', label_col='scientificName', test_size=0.3, batch_size=10, train_batch_size=5000)
