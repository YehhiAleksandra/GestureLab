"""
CSLR модель v3: обучение на реальных MediaPipe landmarks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from pathlib import Path


class CSLRDatasetV3(Dataset):
    """Датасет с реальными landmarks."""
    
    def __init__(self, data_path: Path):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.segments = data['segments']
        
        # Создаём vocabulary
        self.vocab = {}
        self.inv_vocab = {}
        for seg in self.segments:
            gesture = seg['gesture']
            if gesture not in self.vocab:
                idx = len(self.vocab)
                self.vocab[gesture] = idx
                self.inv_vocab[idx] = gesture
        
        self.blank_idx = len(self.vocab)
        self.vocab['<blank>'] = self.blank_idx
        self.inv_vocab[self.blank_idx] = '<blank>'
        
        print(f"Vocabulary size: {len(self.vocab)} (including blank)")
        print(f"Segments: {len(self.segments)}")
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        seg = self.segments[idx]
        
        # Реальные landmarks
        features = np.array(seg['features'], dtype=np.float32)  # [100, 126]
        
        label_idx = self.vocab[seg['gesture']]
        label_seq = [label_idx]
        
        return {
            'features': torch.from_numpy(features),  # [100, 126]
            'label': torch.tensor(label_seq, dtype=torch.long),  # [1]
            'length': features.shape[0],
            'gesture': seg['gesture'],
        }


class CSLRModelV3(nn.Module):
    """CSLR модель v3: BiLSTM + CTC на реальных landmarks."""
    
    def __init__(self, input_dim: int = 126, hidden_dim: int = 512, num_layers: int = 3, num_classes: int = 350):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if num_layers > 1 else 0.0,
        )
        
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        logits = self.fc(out)
        return logits


def collate_fn(batch):
    features = [item['features'] for item in batch]
    labels = [item['label'] for item in batch]
    lengths = torch.tensor([item['length'] for item in batch])
    
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
        'lengths': lengths,
        'label_lengths': torch.tensor([len(l) for l in labels]),
    }


def train_model_v3(
    dataset_path: Path,
    output_path: Path,
    num_epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 0.001,
):
    """Обучение CSLR модели v3 на реальных landmarks."""
    
    dataset = CSLRDatasetV3(dataset_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    
    model = CSLRModelV3(
        input_dim=126,
        hidden_dim=512,
        num_layers=3,
        num_classes=len(dataset.vocab),
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    ctc_loss = nn.CTCLoss(blank=dataset.blank_idx, zero_infinity=True)
    
    print(f"\nНачинаю обучение на {num_epochs} эпох...")
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        for batch in dataloader:
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            lengths = batch['lengths']
            label_lengths = batch['label_lengths']
            
            logits = model(features, lengths)
            log_probs = F.log_softmax(logits, dim=2)
            log_probs = log_probs.permute(1, 0, 2)
            
            input_lengths = torch.full((features.shape[0],), log_probs.shape[0], dtype=torch.long)
            
            loss = ctc_loss(log_probs, labels, input_lengths, label_lengths)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab': dataset.vocab,
                'inv_vocab': dataset.inv_vocab,
                'blank_idx': dataset.blank_idx,
                'best_loss': best_loss,
            }, output_path)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Best: {best_loss:.4f}")
    
    print(f"\nОбучение завершено. Best loss: {best_loss:.4f}")
    print(f"Модель сохранена: {output_path}")


if __name__ == '__main__':
    dataset_path = Path('datasets/youtube_rsl/cslr_dataset_with_real_landmarks.json')
    output_path = Path('models/cslr_model_v3.pt')
    
    train_model_v3(
        dataset_path=dataset_path,
        output_path=output_path,
        num_epochs=100,
        batch_size=16,
        learning_rate=0.001,
    )
