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
    hist = np.zeros((height, width), dtype=int)
    ys = events[:, 0].astype(int)
    xs = events[:, 1].astype(int)
    np.add.at(hist, (ys, xs), 1)

    coords = np.column_stack(np.nonzero(hist))
    freqs = hist[coords[:, 0], coords[:, 1]].reshape(-1, 1)

    if freqs.size == 0:
        return events

    gmm = GaussianMixture(n_components=n_components,
                          covariance_type='full',
                          random_state=0)
    gmm.fit(freqs)
    labels = gmm.predict(freqs)

    mus = gmm.means_.flatten()
    order = np.argsort(mus)
    mu_low, mu_high = mus[order[0]], mus[order[1]]

    if (mu_high - mu_low) < mean_diff_threshold:
        return events

    low_cluster = order[0]
    low_coords = {tuple(coord) for coord, lbl in zip(coords, labels) if lbl == low_cluster}

    mask = [(y, x) in low_coords for y, x in zip(ys, xs)]
    return events[np.array(mask)]


IMAGE_SHAPE = (720, 1280)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str,
                        default='/home/coraldl/EV/MouseSIS/data/MouseSIS')
    parser.add_argument('--mean_diff_thresh', type=float, default=2.5,
                        help='GMM threshold')
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
            filtered = filter_low_freq_events(events,
                                              height, width,
                                              n_components=gmm_components,
                                              mean_diff_threshold=mean_diff_threshold)
            if filtered.size == 0:
                continue

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
