# train/data.py — 데이터셋 준비 & DataLoader

import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

import config


def get_transforms():
    """학습/평가용 전처리 파이프라인 반환"""
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(config.CIFAR10_MEAN, config.CIFAR10_STD),
    ])

    # 검증/테스트는 데이터 증강 없이 정규화만 적용
    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config.CIFAR10_MEAN, config.CIFAR10_STD),
    ])

    return transform_train, transform_eval


def get_loaders():
    """CIFAR-10 (학습 / 검증 / 테스트) DataLoader 반환

    - 학습/검증은 train split 50,000장을 (1 - VAL_RATIO):(VAL_RATIO)로 나눠 사용
    - 같은 train split을 두 번 만들되 transform만 다르게 해서
      같은 인덱스(Subset)로 나눠도 학습엔 증강, 검증엔 정규화만 들어가게 함
    - 분할은 config.SEED로 고정 → 매번 같은 검증셋
    """
    transform_train, transform_eval = get_transforms()

    # 같은 train split, transform만 다른 두 버전
    full_train = torchvision.datasets.CIFAR10(
        root=config.DATA_DIR, train=True, download=True, transform=transform_train
    )
    full_val = torchvision.datasets.CIFAR10(
        root=config.DATA_DIR, train=True, download=True, transform=transform_eval
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=config.DATA_DIR, train=False, download=True, transform=transform_eval
    )

    # 학습/검증 인덱스 분리 (시드 고정으로 결정적 분할)
    n = len(full_train)
    val_size   = int(config.VAL_RATIO * n)
    train_size = n - val_size

    generator = torch.Generator().manual_seed(config.SEED)
    indices   = torch.randperm(n, generator=generator).tolist()
    train_idx, val_idx = indices[:train_size], indices[train_size:]

    train_dataset = Subset(full_train, train_idx)   # 증강 O
    val_dataset   = Subset(full_val,   val_idx)     # 증강 X (정규화만)

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,   batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,  batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )

    print(f'[data] 학습: {len(train_dataset)}장 | 검증: {len(val_dataset)}장 | 테스트: {len(test_dataset)}장')
    return train_loader, val_loader, test_loader


def denormalize(tensor):
    """정규화된 텐서를 시각화용으로 복원"""
    mean = torch.tensor(config.CIFAR10_MEAN).view(3, 1, 1)
    std  = torch.tensor(config.CIFAR10_STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


__all__ = ['get_transforms', 'get_loaders', 'denormalize']
