import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from scipy.spatial.distance import cosine

#file_path = '/Users/michielcastelein/Thesis/Data/Diff/Diffprompt.csv'
file_path = '/Users/michielcastelein/Thesis/Data/GPT/GPTprompt.csv'

data = pd.read_csv(file_path)

# Extract the initial prompts (first row) and the rest as input prompts
initial_prompts = data.iloc[0].tolist()
input_prompts = data.iloc[1:].values.tolist()

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Get BERT embeddings for initial prompts
initial_embeddings = [get_bert_embedding(prompt) for prompt in initial_prompts]

# Get BERT embeddings for input prompts and compute similarity
similarities = []
for i, initial_emb in enumerate(initial_embeddings):
    col_similarities = []
    for prompt in input_prompts:
        input_emb = get_bert_embedding(prompt[i])
        similarity = 1 - cosine(initial_emb, input_emb)
        col_similarities.append(similarity)
    similarities.append(col_similarities)

# Create a DataFrame to display similarities
similarity_df = pd.DataFrame(similarities).T
similarity_df.columns = [f"Similarity to Initial Prompt {i+1}" for i in range(len(initial_prompts))]

# Save the similarity DataFrame to a CSV file
#output_file_path = '/Users/michielcastelein/Thesis/Data/Diff/Diff_similarity.csv'
output_file_path = '/Users/michielcastelein/Thesis/Data/GPT/GPT_similarity.csv'

similarity_df.to_csv(output_file_path, index=False)

print(f"Similarity DataFrame saved to {output_file_path}")
