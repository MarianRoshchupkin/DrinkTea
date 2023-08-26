import numpy as np
import torch
from .models import QuestionAnswering  # Импортируйте модель
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from torch.utils.data import Dataset, DataLoader

AI_MODEL = torch.load("ai_rubert_model/ruBert.pt")
TOKENIZER = torch.load("ai_rubert_model/tokenizer.pt")


class CustomDataset(Dataset):

    def __init__(self, questions_set):
        self.questions_set = questions_set

    def tokenize(self, text):
        return TOKENIZER(text, return_tensors='pt', padding='max_length', truncation=True, max_length=150)

    def __len__(self):
        return len(self.questions_set)

    def __getitem__(self, index):
        output = self.questions_set[index]
        output = self.tokenize(output)
        return {k: v.reshape(-1) for k, v in output.items()}


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output['last_hidden_state']
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def convert_in_vector_massive(questions_set):
    eval_ds = CustomDataset(questions_set)
    eval_dataloader = DataLoader(eval_ds, batch_size=1)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    AI_MODEL.to(device)
    AI_MODEL.eval()

    embeddings = torch.Tensor().to(device)

    with torch.no_grad():
        for n_batch, batch in enumerate(tqdm(eval_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = AI_MODEL(**batch)
            embeddings = torch.cat([embeddings, mean_pooling(outputs, batch['attention_mask'])])
        embeddings = embeddings.cpu().numpy()
    return embeddings


def calculate_similarity(query, questions_in_vector_format):
    mean_pooled = convert_in_vector_massive(np.array([query]))
    similarity = []

    for rec in questions_in_vector_format:
        rec_flat = [item for sublist in rec for item in sublist]
        rec_array = np.fromiter(rec_flat, dtype=np.float32)

        # Check if the shape of rec_array matches the expected shape
        if mean_pooled.shape[1] == rec_array.shape[0]:
            cos_sim = cosine_similarity(mean_pooled, rec_array.reshape(1, -1))
            similarity.append(cos_sim[0][0])  # Extract the cosine similarity value
        else:
            similarity.append(0.0)  # Handle mismatched dimensions

    index = np.argmax(similarity)
    return index
