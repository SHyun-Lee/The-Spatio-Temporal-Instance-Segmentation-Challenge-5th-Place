import argparse
from pathlib import Path
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

    :param events: [N x 4] numpy array of (y, x, t, p)
    :param height: image height
    :param width: image width
    :param n_components: GMM 성분 수 (보통 2)
    :param mean_diff_threshold: 평균 차이 임계값
    :return: filtered or original events array
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


IMAGE_SHAPE = (720, 1280)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str,
                        default='/home/coraldl/EV/MouseSIS/data/MouseSIS')
    parser.add_argument('--mean_diff_thresh', type=float, default=2.5,
                        help='GMM 평균 차이 임계값')
    return parser.parse_args()


def process_sequence(seq_path: Path,
                     reconstructor: E2Vid,
                     output_base: Path,
                     height: int,
                     width: int,
                     gmm_components: int,
                     mean_diff_threshold: float):
    seq_name = seq_path.stem
    output_dir = output_base / seq_name / 'e2vid'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {seq_path.name} -> {output_dir} (mean diff thresh={mean_diff_threshold})")
    with h5py.File(seq_path, 'r') as f:
        raw_events = np.stack([
            f['y'][:-1], f['x'][:-1], f['t'][:-1], f['p'][:-1]
        ], axis=-1)
        ev_indices = np.concatenate(([0], f['img2event'][:]))

        for i, (start, end) in enumerate(
            tqdm(zip(ev_indices[:-1], ev_indices[1:]), total=len(ev_indices)-1)
        ):
            events = raw_events[start:end]
            # GMM 평균 차이 기준 필터링
            filtered = filter_low_freq_events(events,
                                              height, width,
                                              n_components=gmm_components,
                                              mean_diff_threshold=mean_diff_threshold)
            if filtered.size == 0:
                continue

            # reconstruction
            e2vid_img = reconstructor(filtered)
            filename = f"{i:08d}.png"
            cv2.imwrite(str(output_dir / filename), e2vid_img)


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    source_dir = data_root / 'top' / 'val_ori'

    test_base = Path('/home/coraldl/EV/MouseSIS/data/val_ave')

    for seq_path in source_dir.glob('*.h5'):
        reconstructor = E2Vid(image_shape=IMAGE_SHAPE, use_gpu=True)
        process_sequence(
            seq_path,
            reconstructor,
            output_base=test_base,
            height=IMAGE_SHAPE[0],
            width=IMAGE_SHAPE[1],
            gmm_components=2,
            mean_diff_threshold=args.mean_diff_thresh
        )


if __name__ == '__main__':
    main()
