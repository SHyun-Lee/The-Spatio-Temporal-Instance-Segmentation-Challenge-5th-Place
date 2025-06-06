import argparse
from pathlib import Path
import os

import cv2
import numpy as np
from tqdm import tqdm
import h5py

from evlib.processing.reconstruction import E2Vid
from sklearn.mixture import GaussianMixture


def filter_low_freq_events(events: np.ndarray,
                           height: int,
                           width: int,
                           n_components: int = 2,
                           mean_diff_threshold: float = 2.5) -> np.ndarray:
    """
    GMM 기반 픽셀 빈도 분포를 계산하여 두 Gaussian 성분의 평균 차이가
    mean_diff_threshold 이상일 때만 저빈도 이벤트만 필터링,
    그렇지 않으면 원본 events 반환
    """
    # 1) 픽셀별 이벤트 빈도 히스토그램 계산
    hist = np.zeros((height, width), dtype=int)
    ys = events[:, 0].astype(int)
    xs = events[:, 1].astype(int)
    np.add.at(hist, (ys, xs), 1)

    # 2) 빈도 > 0 픽셀 좌표 및 빈도값 추출
    coords = np.column_stack(np.nonzero(hist))
    freqs = hist[coords[:, 0], coords[:, 1]].reshape(-1, 1)

    # 이벤트가 없으면 바로 반환
    if freqs.size == 0:
        return events

    # 3) GMM 적합 및 레이블 예측
    gmm = GaussianMixture(n_components=n_components,
                          covariance_type='full',
                          random_state=0)
    gmm.fit(freqs)
    labels = gmm.predict(freqs)

    # 4) 두 성분의 평균 계산 후 정렬
    mus = gmm.means_.flatten()
    order = np.argsort(mus)
    mu_low, mu_high = mus[order[0]], mus[order[1]]

    # 5) 평균 차이가 임계치 미만이면 원본 반환
    if (mu_high - mu_low) < mean_diff_threshold:
        return events

    # 6) 평균이 더 낮은 클러스터(low_freq)만 남기기
    low_cluster = order[0]
    low_coords = {tuple(coord) for coord, lbl in zip(coords, labels) if lbl == low_cluster}

    # 7) 원본 events 중 low_freq 픽셀만 필터링
    mask = [(y, x) in low_coords for y, x in zip(ys, xs)]
    return events[np.array(mask)]


IMAGE_SHAPE = (720, 1280)  # (height, width)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str,
                        default='/home/coraldl/EV/MouseSIS/data/MouseSIS')
    parser.add_argument('--output_root', type=str,
                        default='/home/coraldl/EV/MouseSIS/figure')
    parser.add_argument('--mean_diff_thresh', type=float, default=2.5,
                        help='GMM 평균 차이 임계값')
    parser.add_argument('--gmm_components', type=int, default=2,
                        help='GMM 성분 수')
    return parser.parse_args()


def process_sequence(seq_path: Path,
                     reconstructor: E2Vid,
                     output_base: Path,
                     height: int,
                     width: int,
                     gmm_components: int,
                     mean_diff_threshold: float):
    seq_name = seq_path.stem
    # “복셀(max projection)” 이미지를 저장할 디렉토리
    voxel_img_dir = output_base / seq_name / 'voxels_png'
    voxel_img_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {seq_name} -> {voxel_img_dir} (mean diff thresh={mean_diff_threshold})")
    with h5py.File(seq_path, 'r') as f:
        raw_events = np.stack([
            f['y'][:-1], f['x'][:-1], f['t'][:-1], f['p'][:-1]
        ], axis=-1)
        ev_indices = np.concatenate(([0], f['img2event'][:]))

        for i, (start, end) in enumerate(
            tqdm(zip(ev_indices[:-1], ev_indices[1:]), total=len(ev_indices)-1)
        ):
            events = raw_events[start:end]

            filtered = filter_low_freq_events(events, height, width,
                                            n_components=gmm_components,
                                            mean_diff_threshold=mean_diff_threshold)
            if filtered.size == 0:
                continue

            # 1) 복셀 그리드 생성
            # 1) 복셀 그리드 생성
            voxel_grid = reconstructor.voxelizer(filtered)  # shape: (n_bins, H, W)

            # 2) max projection (시간축 축소)
            max_proj = voxel_grid.max(axis=0)  # shape: (H, W), float

            # 3) 0~255 그레이스케일로 정규화
            min_val, max_val = max_proj.min(), max_proj.max()
            if max_val > min_val:
                normalized = (max_proj - min_val) / (max_val - min_val)
            else:
                normalized = np.zeros_like(max_proj, dtype=float)

            img_8bit = (normalized * 255).astype(np.uint8)

            # 4) 배경(max_proj == 0)인 픽셀을 중간 회색(127)으로 지정
            img_8bit[max_proj == 0] = 127

            # 5) PNG로 저장
            filename = f"{i:08d}.png"
            cv2.imwrite(str(voxel_img_dir / filename), img_8bit)

            # **원한다면 컬러맵 적용해서 컬러 이미지로 저장할 수도 있습니다**
            # 예를 들어, Matplotlib의 컬러맵을 사용하려면:
            # import matplotlib.cm as cm
            # colored = cm.jet(normalized)  # RGBA float in [0,1]
            # colored_bgr = (colored[..., :3] * 255).astype(np.uint8)[..., ::-1]  # RGB→BGR
            # cv2.imwrite(str(voxel_img_dir / f"{i:08d}_jet.png"), colored_bgr)


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    source_dir = data_root / 'top' / 'train'

    test_base = Path(args.output_root)

    for seq_path in source_dir.glob('*.h5'):
        # 오직 'seq09'만 처리
        if seq_path.stem != 'seq14':
            continue

        # E2Vid 생성자: 복셀화(=voxelizer)만 사용하므로 image_shape만 전달
        reconstructor = E2Vid(image_shape=IMAGE_SHAPE, use_gpu=True)

        process_sequence(
            seq_path=seq_path,
            reconstructor=reconstructor,
            output_base=test_base,
            height=IMAGE_SHAPE[0],
            width=IMAGE_SHAPE[1],
            gmm_components=args.gmm_components,
            mean_diff_threshold=args.mean_diff_thresh
        )


if __name__ == '__main__':
    main()
