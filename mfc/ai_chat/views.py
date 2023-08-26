from django.shortcuts import render, HttpResponse
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import numpy as np
import pandas as pd
import torch
import os
import json
from django.http import JsonResponse

from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from torch.utils.data import Dataset, DataLoader


model = torch.load(r"C:\Users\jet\Desktop\pythonProject3\ruBert.pt")
tokenizer = torch.load(r"C:\Users\jet\Desktop\pythonProject3\tokenizer.pt")

class CustomDataset(Dataset):

    def __init__(self, X):
        self.text = X

    def tokenize(self, text):
        return tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=150)

    def __len__(self):
        return self.text.shape[0]

    def __getitem__(self, index):
        output = self.text[index]
        output = self.tokenize(output)
        return {k: v.reshape(-1) for k, v in output.items()}

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output['last_hidden_state']
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def get_embedding(quary):
  eval_ds = CustomDataset(quary)
  eval_dataloader = DataLoader(eval_ds, batch_size=1)

  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  model.to(device)
  model.eval()

  embeddings = torch.Tensor().to(device)

  with torch.no_grad():
      for n_batch, batch in enumerate(tqdm(eval_dataloader)):
          batch = {k: v.to(device) for k, v in batch.items()}
          outputs = model(**batch)
          embeddings = torch.cat([embeddings, mean_pooling(outputs, batch['attention_mask'])])
      embeddings = embeddings.cpu().numpy()
  return embeddings

df = pd.read_excel(r'C:\Users\jet\Desktop\pythonProject3\train_dataset_Датасет.xlsx')

TQA_train =df['QUESTION'].values

for i in range(len(TQA_train)):
  TQA_train[i] = str(TQA_train[i])

Quastions = get_embedding(TQA_train)

def calculate_similarity(query):
  mean_pooled = get_embedding(np.array([query]))
  similarity = []
  for rec in Quastions:
      cos_sim = cosine_similarity([mean_pooled[0]],
                [np.fromiter(rec, dtype=np.float32)])
      similarity.append(cos_sim)
  index = np.argmax(similarity)
  return index
def ai_chat(request):
    return render(request, "ai_chat/chat.html")


def ai_request(request):
    body = json.loads(request.body)
    print(df.QUESTION[calculate_similarity(body)])
    ai_answer = {"response": f"ai_request работает {body}"}
    return JsonResponse(ai_answer)
