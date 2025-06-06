import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import argparse

def visualize_sequence(h5_path, output_dir):
    """HDF5 내부의 images만 프레임 단위로 저장"""
    h5_path = Path(h5_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, 'r') as f:
        images = f['images']
        for i, img in tqdm(enumerate(images), total=len(images), desc=f"Seq {h5_path.stem}"):
            out_path = output_dir / f'frame_{i:05d}.png'
            Image.fromarray(img).save(out_path)

    print(f'Done! {h5_path.stem}')

def parse_args():
    parser = argparse.ArgumentParser(description='HDF5 이미지 시각화 (이벤트 제거)')
    parser.add_argument('--h5_path', required=True, help='.h5 파일 또는 디렉토리 경로')
    parser.add_argument('--output_dir', default='output/frames', help='저장 디렉토리')
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
        visualize_sequence(fpath, seq_out)
