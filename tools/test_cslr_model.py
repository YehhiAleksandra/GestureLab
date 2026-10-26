"""
Тестирование CSLR модели на Bible RSL видео.
"""
import torch
import torch.nn.functional as F
import json
import numpy as np
from pathlib import Path
from train_cslr_model import CSLRModel, CSLRDataset


def decode_ctc_greedy(log_probs, blank_idx):
    """
    Greedy CTC декодирование.
    log_probs: [T, num_classes]
    """
    # Argmax по классам
    preds = torch.argmax(log_probs, dim=1)  # [T]
    
    # Удаляем повторы и blank
    result = []
    prev = -1
    for p in preds:
        if p != blank_idx and p != prev:
            result.append(p.item())
        prev = p
    
    return result


def test_model(
    model_path: Path,
    dataset_path: Path,
    test_video_id: str = None,
):
    """Тестирование модели."""
    
    # Загружаем модель
    print(f"Загружаю модель: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    vocab = checkpoint['vocab']
    inv_vocab = checkpoint['inv_vocab']
    blank_idx = checkpoint['blank_idx']
    
    # Создаём датасет для получения input_dim
    dataset = CSLRDataset(dataset_path)
    
    # Модель
    model = CSLRModel(
        input_dim=126,
        hidden_dim=256,
        num_layers=2,
        num_classes=len(vocab),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"Device: {device}")
    print(f"Vocabulary: {len(vocab)} классов\n")
    
    # Тестируем на каждом сегменте
    print("=== Тестирование на тренировочных сегментах ===")
    correct = 0
    total = 0
    
    for idx, seg in enumerate(dataset.segments):
        # Фильтр по видео
        if test_video_id and seg['video_id'] != test_video_id:
            continue
        
        features = np.array(seg['features'], dtype=np.float32)
        true_label = seg['label']
        
        # Ограничиваем длину
        if len(features) > 500:
            features = features[:500]
        
        # Конвертируем в tensor
        features_tensor = torch.from_numpy(features).unsqueeze(0).to(device)  # [1, T, 126]
        lengths = torch.tensor([len(features)])
        
        # Инференс
        with torch.no_grad():
            logits = model(features_tensor, lengths)  # [1, T, num_classes]
            log_probs = F.log_softmax(logits, dim=2)
            log_probs = log_probs.squeeze(0).cpu()  # [T, num_classes]
        
        # Декодирование
        pred_indices = decode_ctc_greedy(log_probs, blank_idx)
        pred_labels = [inv_vocab.get(i, f'<unk_{i}>') for i in pred_indices]
        pred_text = ' '.join(pred_labels) if pred_labels else '<empty>'
        
        # Проверка
        is_correct = (len(pred_indices) == 1 and pred_indices[0] == vocab[true_label])
        correct += int(is_correct)
        total += 1
        
        status = "✓" if is_correct else "✗"
        print(f"{status} {seg['video_id']} {seg['start_sec']}-{seg['end_sec']}с")
        print(f"  True: {true_label}")
        print(f"  Pred: {pred_text}")
        if not is_correct:
            print(f"  Pred indices: {pred_indices}")
        print()
    
    print(f"=== Итого ===")
    print(f"Правильных: {correct}/{total} ({correct/total*100:.1f}%)")


if __name__ == '__main__':
    model_path = Path('models/cslr_model.pt')
    dataset_path = Path('datasets/youtube_rsl/cslr_dataset.json')
    
    # Тестируем на всех сегментах
    test_model(model_path, dataset_path)
    
    # Или только на одном видео
    # test_model(model_path, dataset_path, test_video_id='26RsAcQCVBA')
