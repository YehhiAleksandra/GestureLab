"""
CSLR модель с CTC loss для непрерывной жестовой речи.
Архитектура: BiLSTM + Linear + CTC
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple


class CSLRDataset(Dataset):
    """Датасет для CSLR обучения."""
    
    def __init__(self, data_path: Path, max_frames: int = 500):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.segments = data['segments']
        self.max_frames = max_frames
        
        # Создаём vocabulary из labels
        self.vocab = {}
        self.inv_vocab = {}
        for i, seg in enumerate(self.segments):
            label = seg['label']
            if label not in self.vocab:
                idx = len(self.vocab)
                self.vocab[label] = idx
                self.inv_vocab[idx] = label
        
        # Добавляем blank token для CTC
        self.blank_idx = len(self.vocab)
        self.vocab['<blank>'] = self.blank_idx
        self.inv_vocab[self.blank_idx] = '<blank>'
        
        print(f"Vocabulary size: {len(self.vocab)} (including blank)")
        print(f"Segments: {len(self.segments)}")
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        seg = self.segments[idx]
        features = np.array(seg['features'], dtype=np.float32)  # [T, 126]
        label = seg['label']
        
        # Ограничиваем длину
        if len(features) > self.max_frames:
            features = features[:self.max_frames]
        
        # Конвертируем label в последовательность индексов
        # Для простоты: каждый label = один токен (не последовательность)
        label_idx = self.vocab[label]
        label_seq = [label_idx]  # [1] — один токен
        
        return {
            'features': torch.from_numpy(features),  # [T, 126]
            'label': torch.tensor(label_seq, dtype=torch.long),  # [1]
            'length': len(features),
        }


class CSLRModel(nn.Module):
    """CSLR модель: BiLSTM + CTC."""
    
    def __init__(self, input_dim: int = 126, hidden_dim: int = 256, num_layers: int = 2, num_classes: int = 21):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if num_layers > 1 else 0.0,
        )
        
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 для bidirectional
    
    def forward(self, x, lengths):
        """
        x: [batch, T, input_dim]
        lengths: [batch]
        """
        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # LSTM
        packed_out, _ = self.lstm(packed)
        
        # Unpack
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        # out: [batch, T, hidden*2]
        
        # Linear
        logits = self.fc(out)
        # logits: [batch, T, num_classes]
        
        return logits


def collate_fn(batch):
    """Collate function для DataLoader."""
    features = [item['features'] for item in batch]
    labels = [item['label'] for item in batch]
    lengths = [item['length'] for item in batch]
    
    # Pad features
    max_len = max(lengths)
    padded_features = torch.zeros(len(batch), max_len, features[0].shape[1])
    for i, (feat, length) in enumerate(zip(features, lengths)):
        padded_features[i, :length] = feat
    
    # Pad labels
    max_label_len = max(len(l) for l in labels)
    padded_labels = torch.full((len(batch), max_label_len), -1, dtype=torch.long)
    for i, label in enumerate(labels):
        padded_labels[i, :len(label)] = label
    
    return {
        'features': padded_features,
        'labels': padded_labels,
        'lengths': torch.tensor(lengths),
        'label_lengths': torch.tensor([len(l) for l in labels]),
    }


def train_model(
    dataset_path: Path,
    output_path: Path,
    num_epochs: int = 50,
    batch_size: int = 4,
    learning_rate: float = 0.001,
):
    """Обучение CSLR модели."""
    
    # Загружаем датасет
    dataset = CSLRDataset(dataset_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    
    # Модель
    model = CSLRModel(
        input_dim=126,
        hidden_dim=256,
        num_layers=2,
        num_classes=len(dataset.vocab),
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # CTC Loss
    ctc_loss = nn.CTCLoss(blank=dataset.blank_idx, zero_infinity=True)
    
    # Обучение
    print(f"\nНачинаю обучение на {num_epochs} эпох...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        for batch in dataloader:
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            lengths = batch['lengths']
            label_lengths = batch['label_lengths']
            
            # Forward
            logits = model(features, lengths)  # [batch, T, num_classes]
            log_probs = F.log_softmax(logits, dim=2)
            
            # CTC loss требует [T, batch, num_classes]
            log_probs = log_probs.permute(1, 0, 2)
            
            # Input lengths (после LSTM, длина может уменьшиться)
            input_lengths = torch.full((features.shape[0],), log_probs.shape[0], dtype=torch.long)
            
            # Loss
            loss = ctc_loss(log_probs, labels, input_lengths, label_lengths)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # Сохраняем модель
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': dataset.vocab,
        'inv_vocab': dataset.inv_vocab,
        'blank_idx': dataset.blank_idx,
    }, output_path)
    
    print(f"\nМодель сохранена: {output_path}")


if __name__ == '__main__':
    dataset_path = Path('datasets/youtube_rsl/cslr_dataset.json')
    output_path = Path('models/cslr_model.pt')
    
    train_model(
        dataset_path=dataset_path,
        output_path=output_path,
        num_epochs=50,
        batch_size=4,
        learning_rate=0.001,
    )
