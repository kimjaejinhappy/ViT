# ViT on CIFAR-10

논문 *An Image is Worth 16×16 Words* (Dosovitskiy et al., 2020)의 Vision Transformer를
밑바닥부터 구현해 CIFAR-10 분류를 학습합니다.

## 폴더 구조

```
vit_project/
├── main.py              # 실행 진입점 (학습 / 평가)
├── config.py            # 모든 하이퍼파라미터
├── models/
│   ├── __init__.py
│   └── vit.py           # ViT 모델 (PatchEmbedding · MHSA · Encoder · MLP Head)
├── train/
│   ├── __init__.py
│   ├── data.py          # CIFAR-10 DataLoader · 정규화 복원
│   ├── supervised.py    # 학습/평가 루프 (train_one_epoch · evaluate · run_training)
│   └── visualize.py     # 패치 분할 · 학습 곡선 · 예측 · Attention Map 시각화
├── data/                # CIFAR-10 자동 다운로드 (git 추적 제외)
└── runs/                # 체크포인트 · 시각화 결과 (git 추적 제외)
    └── plots/
```

## 실행

```bash
python main.py              # 처음부터 학습
python main.py --eval-only  # 저장된 체크포인트로 평가만
```

> 항상 `vit_project/` 폴더 안에서 실행하세요. `models`, `train`, `config`를
> import하는 기준 경로가 이 폴더입니다.
