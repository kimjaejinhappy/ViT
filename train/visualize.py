# train/visualize.py — 패치 분할 · 학습 곡선 · 예측 · Attention Map 시각화

import os
import torch
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm


def _set_korean_font():
    """설치된 한글 폰트를 자동 탐지해 적용 (없으면 기본 폰트 유지).

    Windows=맑은 고딕, macOS=애플고딕, Colab/Linux=나눔고딕·Noto 순으로 탐색.
    Colab에서는 먼저 `!apt-get install -y fonts-nanum` 로 폰트를 설치하세요.
    """
    candidates = ['Malgun Gothic', 'AppleGothic', 'NanumGothic',
                  'NanumBarunGothic', 'Noto Sans CJK KR', 'Noto Sans KR']
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            matplotlib.rcParams['font.family'] = name
            break
    else:
        print('[viz] 한글 폰트를 못 찾았어요. 제목이 □로 보이면 한글 폰트를 설치하세요.')
    matplotlib.rcParams['axes.unicode_minus'] = False   # 마이너스 기호 깨짐 방지


_set_korean_font()

import config
from .data import denormalize


# ────────────────────────────────────────────────────────────
# 1. 패치 분할 시각화
# ────────────────────────────────────────────────────────────
def show_patches(image_tensor, save_path=None):
    """이미지가 패치로 어떻게 나뉘는지 시각화"""
    img = denormalize(image_tensor).permute(1, 2, 0).numpy()
    n = config.IMAGE_SIZE // config.PATCH_SIZE  # 한 방향 패치 수

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(img)
    axes[0].set_title(f'원본 이미지 ({config.IMAGE_SIZE}x{config.IMAGE_SIZE})')
    axes[0].axis('off')

    axes[1].imshow(img)
    for i in range(n + 1):
        axes[1].axhline(i * config.PATCH_SIZE - 0.5, color='red', lw=0.8)
        axes[1].axvline(i * config.PATCH_SIZE - 0.5, color='red', lw=0.8)
    axes[1].set_title(f'패치 분할 ({n}x{n} = {n*n}개, 각 {config.PATCH_SIZE}x{config.PATCH_SIZE})')
    axes[1].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'[viz] 패치 시각화 저장: {save_path}')
    plt.show()


# ────────────────────────────────────────────────────────────
# 2. 학습 곡선
# ────────────────────────────────────────────────────────────
def plot_history(history, best_acc):
    """Loss & Accuracy 학습 곡선 저장"""
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(epochs, history['train_loss'], label='Train', color='steelblue')
    ax1.plot(epochs, history['val_loss'],   label='Val',   color='tomato', linestyle='--')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('Loss'); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, history['train_acc'], label='Train', color='steelblue')
    ax2.plot(epochs, history['val_acc'],   label='Val',   color='tomato', linestyle='--')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy'); ax2.legend(); ax2.grid(alpha=0.3)

    plt.suptitle(f'ViT on CIFAR-10  (Best Val Acc: {best_acc:.2f}%)', fontsize=13)
    plt.tight_layout()

    save_path = os.path.join(config.PLOTS_DIR, 'history.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'[viz] 학습 곡선 저장: {save_path}')
    plt.show()


# ────────────────────────────────────────────────────────────
# 3. 예측 결과
# ────────────────────────────────────────────────────────────
def plot_predictions(model, test_loader, device):
    """테스트 샘플 8개 예측 결과 시각화"""
    model.eval()
    images, labels = next(iter(test_loader))

    with torch.no_grad():
        preds = model(images[:8].to(device)).argmax(1).cpu()

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        img = denormalize(images[i]).permute(1, 2, 0).numpy()
        true = config.CLASSES[labels[i]]
        pred = config.CLASSES[preds[i]]
        ok   = (labels[i] == preds[i])

        ax.imshow(img)
        ax.set_title(f'정답: {true}\n예측: {pred}', color='green' if ok else 'red', fontsize=9)
        ax.axis('off')

    plt.suptitle('예측 결과 (초록=정답, 빨강=오답)', fontsize=12)
    plt.tight_layout()

    save_path = os.path.join(config.PLOTS_DIR, 'predictions.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'[viz] 예측 결과 저장: {save_path}')
    plt.show()


# ────────────────────────────────────────────────────────────
# 4. Attention Map
# ────────────────────────────────────────────────────────────
def plot_attention_map(model, test_loader, device):
    """[CLS] 토큰의 Attention Map 시각화"""
    model.eval()
    images, labels = next(iter(test_loader))

    def _get_attn(img_tensor):
        attn_weights = []

        def hook(module, inp, out):
            B, N, D = inp[0].shape
            H = module.num_heads
            qkv = module.qkv(inp[0]).reshape(B, N, 3, H, module.head_dim).permute(2, 0, 3, 1, 4)
            q, k, _ = qkv.unbind(0)
            attn = (q @ k.transpose(-2, -1)) * module.scale
            attn_weights.append(attn.softmax(dim=-1).detach().cpu())

        last_block = list(model.transformer.children())[-1]
        handle = last_block.attn.register_forward_hook(hook)
        with torch.no_grad():
            model(img_tensor.unsqueeze(0).to(device))
        handle.remove()

        attn  = attn_weights[0][0]          # (num_heads, N+1, N+1)
        cls_a = attn[:, 0, 1:].mean(0)      # (num_patches,) — 헤드 평균
        n     = config.IMAGE_SIZE // config.PATCH_SIZE
        return cls_a.reshape(n, n).numpy()

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for i in range(4):
        img  = denormalize(images[i]).permute(1, 2, 0).numpy()
        amap = _get_attn(images[i])

        axes[0][i].imshow(img)
        axes[0][i].set_title(config.CLASSES[labels[i]], fontsize=10)
        axes[0][i].axis('off')

        axes[1][i].imshow(img)
        axes[1][i].imshow(amap, alpha=0.6, cmap='hot',
                          extent=[0, config.IMAGE_SIZE, config.IMAGE_SIZE, 0],
                          interpolation='bilinear')
        axes[1][i].set_title('[CLS] Attention', fontsize=10)
        axes[1][i].axis('off')

    plt.suptitle('Attention Map — [CLS]가 어디를 주목하는지', fontsize=12)
    plt.tight_layout()

    save_path = os.path.join(config.PLOTS_DIR, 'attention_map.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'[viz] Attention Map 저장: {save_path}')
    plt.show()


__all__ = ['show_patches', 'plot_history', 'plot_predictions', 'plot_attention_map']
