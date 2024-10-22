import numpy as np
import pandas as pd
import faiss
import gc
import torch
from memory_profiler import profile

#requires 40gb ram 4 cpus each of 10 gb 
@profile
def convert_embeddings_column_to_float32(df, column_name, batch_size=100):
    num_embeddings = len(df)
    df['float32_embeddings'] = None
    float32_embeddings = []

    # Convert embeddings in batches
    for start in range(0, num_embeddings, batch_size):
        end = min(start + batch_size, num_embeddings)
        
        # Extract the batch of embeddings and convert to float32
        batch = df[column_name].iloc[start:end].apply(lambda x: x.astype(np.float32))
        
        # Add the batch to the new column
        float32_embeddings.extend(batch)
        
        # Delete temporary variables
        del batch
        gc.collect()

    # Assign the new 32-bit embeddings to the new column
    df['float32_embeddings'] = float32_embeddings

    # Delete the original column if no longer needed
    gc.collect()

    return df


# Example usage:
df = pd.read_pickle("df_cleaned.pkl")  # Load your DataFrame containing float16 embeddings
df = convert_embeddings_column_to_float32(df, "image_embeddings")
df.to_pickle("df_cleaned_float16.pkl")
print(df.head())
# embeddings_float32 = np.stack(df["image_embeddings"].values)
# index = index_faiss_cosine_similarity(embeddings_float32)
# query_embeddings = embeddings_float32[:10]  # Use a subset as queries
# indices, distances = search_faiss_index(index, query_embeddings)
