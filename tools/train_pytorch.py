#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Полноценное обучение на PyTorch с GPU.
GRU с batch norm, dropout, AdamW, scheduler.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class GestureDataset(Dataset):
    """Датасет жестов."""
    
    def __init__(self, npz_path, train=True):
        data = np.load(npz_path, allow_pickle=True)
        X = data["X"]
        y = data["y"]
        is_val = data["is_val"]
        
        # Фильтруем train/val
        mask = ~is_val if train else is_val
        self.X = torch.FloatTensor(X[mask])
        self.y = torch.LongTensor(y[mask])
        
        print(f"{'Train' if train else 'Val'}: {len(self.X)} примеров")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class GestureGRU(nn.Module):
    """GRU модель с batch norm и dropout."""
    
    def __init__(self, input_dim=126, hidden_dim=128, num_classes=100, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.batch_norm = nn.BatchNorm1d(input_dim)
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 для bidirectional
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        
        # Batch norm на каждом временном шаге
        x = x.reshape(-1, x.shape[-1])  # (batch*seq, input)
        x = self.batch_norm(x)
        x = x.reshape(batch_size, seq_len, -1)  # (batch, seq, input)
        
        # GRU
        output, hidden = self.gru(x)
        
        # Берём последний выход из обоих направлений
        output = output[:, -1, :]  # (batch, hidden*2)
        
        # Dropout + FC
        output = self.dropout(output)
        output = self.fc(output)
        
        return output


def train_model(model, train_loader, val_loader, device, epochs=50, lr=0.001):
    """Обучение модели."""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100.0 * train_correct / train_total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_top3_correct = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                # Top-3 accuracy
                _, top3_predicted = outputs.topk(3, dim=1)
                val_top3_correct += top3_predicted.eq(labels.unsqueeze(1)).any(dim=1).sum().item()
        
        val_acc = 100.0 * val_correct / val_total
        val_top3_acc = 100.0 * val_top3_correct / val_total
        
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pt')
        
        print(f'Epoch {epoch+1}/{epochs} | '
              f'Train Loss: {train_loss/len(train_loader):.3f} | '
              f'Train Acc: {train_acc:.1f}% | '
              f'Val Acc: {val_acc:.1f}% | '
              f'Val Top-3: {val_top3_acc:.1f}%')
    
    print(f'\nBest Val Acc: {best_val_acc:.1f}%')
    return best_val_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='Путь к NPZ датасету')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--save', default='gesture_model_pytorch.pt')
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    
    # Data
    print('\nЗагрузка данных...')
    train_dataset = GestureDataset(args.dataset, train=True)
    val_dataset = GestureDataset(args.dataset, train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    num_classes = len(np.unique(np.concatenate([train_dataset.y.numpy(), val_dataset.y.numpy()])))
    print(f'Классов: {num_classes}')
    
    # Model
    print('\nСоздание модели...')
    model = GestureGRU(
        input_dim=126,
        hidden_dim=args.hidden,
        num_classes=num_classes,
        num_layers=2,
        dropout=0.3
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Параметров: {total_params:,}')
    
    # Train
    print('\nОбучение...')
    best_acc = train_model(model, train_loader, val_loader, device, args.epochs, args.lr)
    
    # Save
    print(f'\nСохранение модели...')
    model.load_state_dict(torch.load('best_model.pt'))
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
        'hidden_dim': args.hidden,
        'val_acc': best_acc
    }, args.save)
    
    print(f'Сохранено: {args.save}')
    print(f'Best Val Acc: {best_acc:.1f}%')


if __name__ == '__main__':
    main()
