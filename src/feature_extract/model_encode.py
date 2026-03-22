import torch
from torch import nn
from transformers import AutoModel, WavLMModel, BertModel, CLIPVisionModel


class PhoBERTEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim=768, projection_dim=512):
        super().__init__()
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base")
        self.project = nn.Sequential( 
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, projection_dim)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :] 
        projected = self.project(pooled_output)
        return pooled_output, projected


class BERTEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim=768, projection_dim=512):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.project = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, projection_dim)
        )

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask=attention_mask)
        hidden = output.last_hidden_state  
        mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1) 
        return pooled, self.project(pooled)

class WavLMEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim=768, projection_dim=512):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base")
        self.projection = nn.Sequential(  
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, projection_dim)
        )

    def forward(self, input_values):
        outputs = self.wavlm(input_values=input_values)
        hidden_states = outputs.last_hidden_state  
        pooled_output = hidden_states.mean(dim=1)  
        projected = self.projection(pooled_output)
        return pooled_output, projected


class CLIPVideoEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim=768, projection_dim=512):
        super().__init__()
        self.clip_vision = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, projection_dim),
        )

    def forward(self, pixel_values):
        outputs = self.clip_vision(pixel_values=pixel_values)
        pooled_output = outputs.pooler_output
        projected = self.projection(pooled_output)
        return pooled_output, projected
