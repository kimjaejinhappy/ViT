# train/supervised.py — 학습 루프 & 평가

import os
import math
import torch
import torch.nn as nn
import torch.optim as optim

import config


# ────────────────────────────────────────────────────────────
# 학습 / 평가 함수
# ────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct    += outputs.argmax(1).eq(labels).sum().item()
        total      += labels.size(0)

    return total_loss / len(loader), 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        correct    += outputs.argmax(1).eq(labels).sum().item()
        total      += labels.size(0)

    return total_loss / len(loader), 100.0 * correct / total


# ────────────────────────────────────────────────────────────
# LR 스케줄: 선형 Warmup → Cosine 감소
# ────────────────────────────────────────────────────────────

def _warmup_cosine(epoch):
    """base LR(config.LR)에 곱할 배수(factor)를 반환.

    - epoch < WARMUP_EPOCHS : 0→1 로 선형 증가 (warmup)
    - 이후                  : 1→0 으로 cosine 감소
    ViT는 처음부터 큰 LR을 주면 발산하기 쉬워서, 초반 몇 에폭 동안
    LR을 천천히 끌어올리는 warmup이 학습 안정화에 크게 도움을 준다.
    """
    if epoch < config.WARMUP_EPOCHS:
        return (epoch + 1) / config.WARMUP_EPOCHS
    progress = (epoch - config.WARMUP_EPOCHS) / max(1, config.EPOCHS - config.WARMUP_EPOCHS - 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# ────────────────────────────────────────────────────────────
# 전체 학습 루프
# ────────────────────────────────────────────────────────────

def run_training(model, train_loader, val_loader, device):
    """학습 실행 후 (history, best_val_acc) 반환.

    - 손실 : CrossEntropy + Label Smoothing (config.LABEL_SMOOTHING)
    - 스케줄: 선형 Warmup → Cosine 감소 (LambdaLR)
    - 체크포인트: '검증' 정확도가 최고일 때 저장 (테스트셋은 학습에 관여 안 함)
    """
    os.makedirs(config.RUNS_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    optimizer = optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, _warmup_cosine)

    history  = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    best_acc = 0.0

    print(f'\n{"Epoch":>6} | {"Train Loss":>10} | {"Train Acc":>10} | {"Val Loss":>9} | {"Val Acc":>8} | {"LR":>9}')
    print('-' * 70)

    for epoch in range(1, config.EPOCHS + 1):
        lr = scheduler.get_last_lr()[0]          # 이번 에폭에 실제로 쓰이는 LR

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(lr)

        # 최고 '검증' 정확도 체크포인트 저장
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), config.CHECKPOINT)

        if epoch % 5 == 0 or epoch == 1:
            print(f'{epoch:>6} | {train_loss:>10.4f} | {train_acc:>9.2f}% | '
                  f'{val_loss:>9.4f} | {val_acc:>7.2f}% | {lr:>9.2e}')

    print('-' * 70)
    print(f'[train] 완료! 최고 검증 정확도: {best_acc:.2f}%')
    print(f'[train] 체크포인트 저장: {config.CHECKPOINT}')
    return history, best_acc


# ────────────────────────────────────────────────────────────
# 최종 테스트 (학습이 끝난 뒤 한 번)
# ────────────────────────────────────────────────────────────

def final_test(model, test_loader, device):
    """최고 검증 정확도 모델로 '테스트셋'을 단 한 번 평가.

    검증셋으로 모델을 골랐으니, 진짜 일반화 성능은 테스트셋으로 측정한다.
    테스트 손실은 비교 편의를 위해 label smoothing 없이 측정.
    """
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f'\n[test] 최종 테스트 정확도: {test_acc:.2f}%  (loss {test_loss:.4f})')
    return test_loss, test_acc


__all__ = ['train_one_epoch', 'evaluate', 'run_training', 'final_test']
