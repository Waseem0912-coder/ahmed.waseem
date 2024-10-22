import os
from PIL import Image
import torch

def get_image_embeddings(image_filename, preprocess, model, device):
    image_folder = "downloaded_images"
    
    try:
        image_path = os.path.join(image_folder, image_filename)
        image = Image.open(image_path)
        inputs = preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():  
            image_embeddings = model.encode_image(inputs)
        
        return image_embeddings.cpu().numpy()
    
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


def worker_function(image_filenames, start_idx, end_idx, result_list, preprocess, model, device, progress_counter):
    for idx in range(start_idx, end_idx):
        embedding = get_image_embeddings(image_filenames[idx], preprocess, model, device)
        result_list[idx] = embedding

        with progress_counter.get_lock():
            progress_counter.value += 1