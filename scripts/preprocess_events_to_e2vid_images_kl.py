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
                           kl_threshold: float = 500.0) -> np.ndarray:
    """
    GMM 기반 픽셀 빈도 분포를 계산하여 KL(N1||N0) 발산이 kl_threshold 이상일 때만
    픽셀 빈도가 낮은 이벤트만 필터링, 그렇지 않으면 원본 events 반환

    :param events: [N x 4] numpy array of (y, x, t, p)
    :param height: image height
    :param width: image width
    :param n_components: GMM 성분 수 (보통 2)
    :param kl_threshold: KL(N1||N0) 임계값
    :return: filtered or original events array
    """
    # 히스토그램 계산
    hist = np.zeros((height, width), dtype=int)
    ys = events[:, 0].astype(int)
    xs = events[:, 1].astype(int)
    np.add.at(hist, (ys, xs), 1)

    # 빈도가 0인 좌표 제외
    coords = np.column_stack(np.nonzero(hist))
    freqs = hist[coords[:, 0], coords[:, 1]].reshape(-1, 1)

    # GMM 적합
    gmm = GaussianMixture(n_components=n_components,
                          covariance_type='full',
                          random_state=0)
    gmm.fit(freqs)
    labels = gmm.predict(freqs)

    # Gaussian params
    mus = gmm.means_.flatten()
    # full covariance -> flatten each covariance matrix to single var
    covs = np.array([gmm.covariances_[i].flatten().mean() for i in range(n_components)])
    sigmas = np.sqrt(covs)

    # 두 성분의 평균·분산 정렬
    order = np.argsort(mus)
    mu0, mu1 = mus[order[0]], mus[order[1]]
    var0, var1 = covs[order[0]], covs[order[1]]
    sigma0, sigma1 = sigmas[order[0]], sigmas[order[1]]

    # KL(N1||N0) 계산
    # D_KL(N1||N0) = ln(sigma0/sigma1) + (var1 + (mu1-mu0)^2)/(2*var0) - 0.5
    kl10 = np.log(sigma0 / sigma1) + (var1 + (mu1 - mu0)**2) / (2.0 * var0) - 0.5

    # 임계값보다 작으면 원본 events 반환
    if kl10 < kl_threshold:
        return events

    # 클러스터 0 = 낮은 빈도, 클러스터 1 = 높은 빈도라고 가정
    low_cluster = order[0]
    low_coords = {tuple(coord) for coord, lbl in zip(coords, labels) if lbl == low_cluster}

    # 원본 events 중 low_coords 픽셀만 남김
    mask = [(y, x) in low_coords for y, x in zip(ys, xs)]
    return events[np.array(mask)]


IMAGE_SHAPE = (720, 1280)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str,
                        default='/home/coraldl/EV/MouseSIS/data/MouseSIS')
    parser.add_argument('--kl_thresh', type=float, default=500.0,
                        help='KL(N1||N0) 임계값')
    return parser.parse_args()


def process_sequence(seq_path: Path,
                     reconstructor: E2Vid,
                     output_base: Path,
                     height: int,
                     width: int,
                     gmm_components: int,
                     kl_threshold: float):
    seq_name = seq_path.stem
    output_dir = output_base / seq_name / 'e2vid'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {seq_path.name} -> {output_dir} (KL thresh={kl_threshold})")
    with h5py.File(seq_path, 'r') as f:
        raw_events = np.stack([
            f['y'][:-1], f['x'][:-1], f['t'][:-1], f['p'][:-1]
        ], axis=-1)
        ev_indices = np.concatenate(([0], f['img2event'][:]))

        for i, (start, end) in enumerate(
            tqdm(zip(ev_indices[:-1], ev_indices[1:]), total=len(ev_indices)-1)
        ):
            events = raw_events[start:end]
            # KL 기준 필터링
            filtered = filter_low_freq_events(events,
                                              height, width,
                                              n_components=gmm_components,
                                              kl_threshold=kl_threshold)
            if filtered.size == 0:
                continue

            # reconstruction
            e2vid_img = reconstructor(filtered)
            filename = f"{i:08d}.png"
            cv2.imwrite(str(output_dir / filename), e2vid_img)

def main():
    args = parse_args()
    data_root = Path(args.data_root)
    source_dir = data_root / 'top' / 'test' 

    test_base  = Path('/home/coraldl/EV/MouseSIS/data/test_kl')

    for seq_path in source_dir.glob('*.h5'):
        reconstructor = E2Vid(image_shape=IMAGE_SHAPE, use_gpu=True)
        process_sequence(
            seq_path,
            reconstructor,
            output_base=test_base,
            height=IMAGE_SHAPE[0],
            width=IMAGE_SHAPE[1],
            gmm_components=2,
            kl_threshold=args.kl_thresh
        )

if __name__ == '__main__':
    main()
