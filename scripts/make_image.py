import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import argparse
import json

def make_overlay_image(image, events):
    """이벤트를 원본 이미지에 오버레이해서 반환"""
    y, x, p = events[:, 0], events[:, 1], events[:, 3]
    image = np.copy(image)
    image[y[p == 0], x[p == 0]] = (255, 0, 0)  # 음극 이벤트→빨강
    image[y[p == 1], x[p == 1]] = (0, 0, 255)  # 양극 이벤트→파랑
    return image

def visualize_sequence(h5_path, annotation_path, output_dir):
    num_event_batch = 30000
    h5_path = Path(h5_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 어노테이션은 이벤트 시퀀스 ID 매칭용으로만 로드 (필요 없으면 삭제해도 됩니다)
    with open(annotation_path, 'r') as f:
        annotations = json.load(f)
    video_id = h5_path.stem.split('_')[-1][-2:]

    with h5py.File(h5_path, 'r') as f:
        images = f['images']
        img2ev = f['img2event']
        H, W = images[0].shape[:2]

        for i, (ev_index, img) in tqdm(enumerate(zip(img2ev, images)),
                                      total=len(img2ev),
                                      desc=f"Seq {video_id}"):

            # 이벤트 윈도우 범위 계산
            start = int(max(0, ev_index - 0.5 * num_event_batch))
            end   = int(min(ev_index + 0.5 * num_event_batch, len(f['y'])))

            # 이벤트 배열 생성 [y, x, t, p]
            events = np.zeros((end - start, 4), dtype=int)
            events[:, 0] = f['y'][start:end]
            events[:, 1] = f['x'][start:end]
            events[:, 2] = f['t'][start:end]
            events[:, 3] = f['p'][start:end]
            # 이미지 경계 밖 이벤트 제거
            events = events[(events[:,0] < H) & (events[:,1] < W)]

            # 오버레이 이미지 생성 및 저장
            overlay = make_overlay_image(img.copy(), events)
            out_path = output_dir / f'overlay_{i:05d}.png'
            Image.fromarray(overlay).save(out_path)

    print('Done!')

def parse_args():
    parser = argparse.ArgumentParser(
        description='이벤트를 오버레이한 이미지 시각화'
    )
    parser.add_argument(
        '--h5_path', required=True,
        help='.h5 파일 또는 디렉토리 경로'
    )
    parser.add_argument(
        '--annotation_json', required=True,
        help='annotation JSON 파일 경로 (오버레이에는 불필요하지만 ID 추출용)'
    )
    parser.add_argument(
        '--output_dir', default='output/visu',
        help='저장 디렉토리'
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    h5 = Path(args.h5_path)
    if h5.is_dir():
        files = sorted(h5.glob('*.h5')) + sorted(h5.glob('*.hdf5'))
    else:
        files = [h5]

    for fpath in files:
        seq_out = Path(args.output_dir) / fpath.stem
        visualize_sequence(fpath, args.annotation_json, seq_out)
