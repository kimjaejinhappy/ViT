# main.py — 실행 진입점
#
# 사용법:
#   python main.py              # 처음부터 학습
#   python main.py --eval-only  # 저장된 모델로 평가만

import os
import argparse
import torch

import config
from train.data       import get_loaders
from train.supervised import run_training, final_test
from train.visualize  import (
    show_patches, plot_history, plot_predictions, plot_attention_map,
)
from models.vit       import ViT


def main(eval_only: bool = False):
    # ── 디바이스 설정 ─────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[main] 디바이스: {device}')
    if torch.cuda.is_available():
        print(f'[main] GPU: {torch.cuda.get_device_name(0)}')

    # ── 데이터 준비 ───────────────────────────────────────────
    train_loader, val_loader, test_loader = get_loaders()

    # ── 모델 초기화 ───────────────────────────────────────────
    os.makedirs(config.RUNS_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    # 패치 분할 시각화 (첫 번째 샘플)
    sample_img, sample_label = train_loader.dataset[0]
    print(f'[main] 샘플 클래스: {config.CLASSES[sample_label]}')
    show_patches(sample_img, save_path=os.path.join(config.PLOTS_DIR, 'patches.png'))

    model = ViT().to(device)
    total, trainable = model.count_params()
    print(f'[main] 파라미터 수: {total:,} (학습 가능: {trainable:,})')

    # ── 학습 or 평가 ──────────────────────────────────────────
    if eval_only:
        # 저장된 체크포인트 로드 후 평가만 실행
        if not os.path.exists(config.CHECKPOINT):
            raise FileNotFoundError(f'체크포인트를 찾을 수 없어요: {config.CHECKPOINT}')
        model.load_state_dict(torch.load(config.CHECKPOINT, map_location=device))
        print(f'[main] 체크포인트 로드: {config.CHECKPOINT}')
    else:
        # 처음부터 학습 (검증셋으로 최고 모델 선택)
        history, best_acc = run_training(model, train_loader, val_loader, device)
        plot_history(history, best_acc)

        # 학습 후 최고(검증 기준) 모델로 교체
        model.load_state_dict(torch.load(config.CHECKPOINT, map_location=device))

    # ── 최종 테스트셋 평가 ────────────────────────────────────
    final_test(model, test_loader, device)

    # ── 결과 시각화 ───────────────────────────────────────────
    plot_predictions(model, test_loader, device)
    plot_attention_map(model, test_loader, device)
    print('\n[main] 완료! 결과는 ./runs/plots/ 폴더에 저장됐어요.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ViT CIFAR-10 학습')
    parser.add_argument('--eval-only', action='store_true',
                        help='저장된 체크포인트로 평가만 실행')
    args = parser.parse_args()
    main(eval_only=args.eval_only)
