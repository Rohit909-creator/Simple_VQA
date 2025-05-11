import os
import torch
import clip
from PIL import Image
import re
import json
from tqdm import tqdm
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

def parse_qa_file(file_path):
    """
    Parse the QA file and extract question-answer pairs organized by image.
    
    Returns:
        dict: A dictionary mapping image_id to a list of (question, answer) tuples
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    qa_pairs = {}
    i = 0
    
    while i < len(lines):
        # Get question line
        question_line = lines[i].strip()
        
        # Extract image name from the question
        image_match = re.search(r'in the (image\d+)', question_line)
        if not image_match:
            i += 1
            continue
            
        image_id = image_match.group(1)
        
        # Get answer line
        if i + 1 < len(lines):
            answer = lines[i + 1].strip()
            
            # Add this QA pair to our dictionary
            if image_id not in qa_pairs:
                qa_pairs[image_id] = []
            
            qa_pairs[image_id].append((question_line, answer))
            
            i += 2  # Move to the next question
        else:
            break
    
    return qa_pairs

def create_unique_answers_dict(qa_pairs):
    """
    Create a dictionary mapping unique answers to unique indices.
    
    Returns:
        dict: A dictionary mapping answer text to a unique index
    """
    unique_answers = set()
    for image_id, pairs in qa_pairs.items():
        for _, answer in pairs:
            # Split answers if they are comma-separated
            if ',' in answer:
                for ans in answer.split(','):
                    unique_answers.add(ans.strip())
            else:
                unique_answers.add(answer)
    
    # Create a mapping of answers to indices
    answer_to_idx = {ans: idx for idx, ans in enumerate(sorted(unique_answers))}
    
    # Save this mapping for later use
    return answer_to_idx

def preprocess_dataset(dataset_dir, cache_dir, batch_size=32):
    """
    Preprocess the entire dataset and cache the results.
    
    Args:
        dataset_dir: Directory containing the dataset
        cache_dir: Directory to store preprocessed data
        batch_size: Batch size for processing
    """
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # Parse the QA pairs
    qa_file_path = os.path.join(dataset_dir, "all_qa_pairs.txt")
    qa_pairs = parse_qa_file(qa_file_path)
    
    # Create answer to index mapping
    answer_to_idx = create_unique_answers_dict(qa_pairs)
    
    # Save the answer to index mapping
    with open(os.path.join(cache_dir, "answer_to_idx.json"), "w") as f:
        json.dump(answer_to_idx, f)
    
    # Process images and questions in batches
    all_image_features = []
    all_text_features = []
    all_targets = []
    all_image_ids = []
    all_questions = []
    all_answers = []
    
    # Group QA pairs by image for efficient processing
    for image_id, pairs in tqdm(qa_pairs.items(), desc="Processing images"):
        image_path = os.path.join(dataset_dir, "images", f"{image_id}.png")
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        # Process image
        try:
            image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
            
            # Process image in batches to avoid memory issues
            with torch.no_grad():
                image_features = model.encode_image(image)
                
            # Process questions and answers for this image
            for question, answer in pairs:
                # Tokenize the question
                text_tokens = clip.tokenize([question]).to(device)
                
                # Get text features
                with torch.no_grad():
                    text_features = model.encode_text(text_tokens)
                
                # Process the answer (could be multiple comma-separated answers)
                if ',' in answer:
                    answer_indices = [answer_to_idx[ans.strip()] for ans in answer.split(',')]
                    target = torch.tensor([answer_indices[0]])  # Take the first answer as target for now
                else:
                    target = torch.tensor([answer_to_idx[answer]])
                
                # Save features
                all_image_features.append(image_features.cpu())
                all_text_features.append(text_features.cpu())
                all_targets.append(target)
                all_image_ids.append(image_id)
                all_questions.append(question)
                all_answers.append(answer)
                
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    # Concatenate all features
    all_image_features = torch.cat(all_image_features, dim=0)
    all_text_features = torch.cat(all_text_features, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Save everything
    torch.save(all_image_features, os.path.join(cache_dir, "images_embs.pt"))
    torch.save(all_text_features, os.path.join(cache_dir, "text_embs.pt"))
    torch.save(all_targets, os.path.join(cache_dir, "targets.pt"))
    
    # Save metadata
    metadata = {
        "image_ids": all_image_ids,
        "questions": all_questions,
        "answers": all_answers
    }
    
    with open(os.path.join(cache_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)
    
    print(f"Preprocessing complete. Data saved to {cache_dir}")
    print(f"Processed {len(all_targets)} QA pairs across {len(set(all_image_ids))} images")
    print(f"Total unique answers: {len(answer_to_idx)}")
    
    return {
        "image_features": all_image_features,
        "text_features": all_text_features,
        "targets": all_targets,
        "answer_to_idx": answer_to_idx,
        "metadata": metadata
    }

class ClipVQADataset(Dataset):
    """
    Dataset class for CLIP VQA dataset.
    Can load from preprocessed cache or preprocess on the fly.
    """
    def __init__(self, dataset_dir=None, cache_dir="dataset_cache", force_preprocess=False):
        """
        Args:
            dataset_dir: Directory containing the dataset (images and QA pairs)
            cache_dir: Directory to store/load preprocessed data
            force_preprocess: Whether to force preprocessing even if cache exists
        """
        self.cache_dir = cache_dir
        
        # Check if preprocessed data exists and load it
        if not force_preprocess and os.path.exists(os.path.join(cache_dir, "targets.pt")):
            self.load_from_cache()
        elif dataset_dir:
            # Preprocess the dataset if needed
            result = preprocess_dataset(dataset_dir, cache_dir)
            self.image_features = result["image_features"]
            self.text_features = result["text_features"]
            self.targets = result["targets"]
            self.answer_to_idx = result["answer_to_idx"]
            self.metadata = result["metadata"]
        else:
            raise ValueError("Either dataset_dir must be provided or cache must exist")
    
    def load_from_cache(self):
        """Load preprocessed data from cache directory"""
        self.image_features = torch.load(os.path.join(self.cache_dir, "images_embs.pt"))
        self.text_features = torch.load(os.path.join(self.cache_dir, "text_embs.pt"))
        self.targets = torch.load(os.path.join(self.cache_dir, "targets.pt"))
        
        with open(os.path.join(self.cache_dir, "answer_to_idx.json"), "r") as f:
            self.answer_to_idx = json.load(f)
        
        with open(os.path.join(self.cache_dir, "metadata.json"), "r") as f:
            self.metadata = json.load(f)
        
        print(f"Loaded {len(self.targets)} QA pairs from cache")
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return {
            "image_features": self.image_features[idx],
            "text_features": self.text_features[idx],
            "target": self.targets[idx],
            "image_id": self.metadata["image_ids"][idx],
            "question": self.metadata["questions"][idx],
            "answer": self.metadata["answers"][idx]
        }
    
    def get_num_classes(self):
        """Return the number of unique answer classes"""
        return len(self.answer_to_idx)
    
    def idx_to_answer(self, idx):
        """Convert index to answer text"""
        # Invert the answer_to_idx dictionary
        idx_to_answer = {v: k for k, v in self.answer_to_idx.items()}
        return idx_to_answer.get(idx, "unknown")


# Example usage
if __name__ == "__main__":
    # Example usage of the preprocessor
    # dataset_dir = r"C:\Users\Rohit Francis\Desktop\Codes\Datasets\VQA\dataset"
    # cache_dir = "./dataset_cache"
    
    # # Preprocess and save
    # preprocess_dataset(dataset_dir, cache_dir)
    
    # # Or load from the Dataset class
    # dataset = ClipVQADataset(dataset_dir=dataset_dir, cache_dir=cache_dir)
    
    # # Display some stats
    # print(f"Dataset size: {len(dataset)}")
    # print(f"Number of classes: {dataset.get_num_classes()}")
    
    # # Example of how to access an item
    # if len(dataset) > 0:
    #     item = dataset[0]
    #     print(f"Sample image ID: {item['image_id']}")
    #     print(f"Sample question: {item['question']}")
    #     print(f"Sample answer: {item['answer']}")
    #     print(f"Sample target index: {item['target'].item()}")
    #     print(f"Sample target answer: {dataset.idx_to_answer(item['target'].item())}")
    
    text_embs = torch.load('./dataset_cache/text_embs.pt')
    images_embs = torch.load('./dataset_cache/images_embs.pt')
    
    print(text_embs.shape, images_embs.shape)