# config.py — 모든 하이퍼파라미터를 한 곳에서 관리해요

# ── 이미지 / 패치 설정 ──────────────────────────────────────
IMAGE_SIZE  = 32   # CIFAR-10 이미지 크기 (32×32)
PATCH_SIZE  = 4    # 패치 크기 (4×4). 논문 원본은 16×16
IN_CHANNELS = 3    # RGB 채널 수

NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2   # 64개
PATCH_DIM   = IN_CHANNELS * PATCH_SIZE ** 2     # 48 (펼친 패치 크기)

# ── Transformer 구조 설정 ────────────────────────────────────
EMBED_DIM  = 128   # 임베딩 차원 D. 논문 ViT-Base는 768
NUM_HEADS  = 8     # Multi-Head Attention 헤드 수
NUM_LAYERS = 6     # Transformer Encoder 블록 반복 횟수
MLP_DIM    = 256   # MLP 내부 히든 차원 (보통 EMBED_DIM × 4)
DROPOUT    = 0.1

# ── 학습 설정 ────────────────────────────────────────────────
NUM_CLASSES = 10   # CIFAR-10 클래스 수
BATCH_SIZE  = 128
EPOCHS      = 30
LR          = 3e-4
WEIGHT_DECAY = 1e-4

WARMUP_EPOCHS   = 5     # LR 워밍업 에폭 (이후 cosine 감소) — ViT 학습 안정화의 핵심
LABEL_SMOOTHING = 0.1   # 라벨 스무딩 계수 (CrossEntropyLoss)
VAL_RATIO       = 0.1   # 학습 데이터 중 검증용으로 떼어낼 비율 (10%)
SEED            = 42    # 검증 분할을 항상 같게 만들기 위한 시드

# ── 경로 설정 ────────────────────────────────────────────────
DATA_DIR      = './data'
RUNS_DIR      = './runs'
CHECKPOINT    = './runs/best_vit.pth'
PLOTS_DIR     = './runs/plots'

# ── CIFAR-10 클래스 이름 ─────────────────────────────────────
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# ── 정규화 통계 (CIFAR-10 전체 데이터셋 기준) ────────────────
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)
