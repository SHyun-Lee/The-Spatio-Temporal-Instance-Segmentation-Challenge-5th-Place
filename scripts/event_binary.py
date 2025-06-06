import argparse
from pathlib import Path
import h5py
import numpy as np
import cv2
from tqdm import tqdm

IMAGE_SHAPE = (720, 1280)
output_dir = Path("/home/coraldl/EV/MouseSIS/scripts/binary")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/home/coraldl/EV/MouseSIS/data/MouseSIS')
    return parser.parse_args()

def process_sequence(seq_path: Path):
    print(f"Processing {seq_path.name}")
    with h5py.File(seq_path, 'r') as f:
        # 이벤트 데이터: [y, x, t, p] 순서로 스택 (마지막 한 요소는 제외)
        events = np.stack([f["y"][0:-1], f["x"][0:-1], f["t"][0:-1], f["p"][0:-1]], axis=-1)
        ev_indices = f['img2event'][:]
        
        # 각 img2event 구간에 대해 바이너리맵 생성
        for i, (start, end) in enumerate(tqdm(zip(ev_indices[:-1], ev_indices[1:]), total=len(ev_indices)-1)):
            segment_events = events[start:end]
            
            # 각 좌표별 이벤트 개수 계산 (y, x 순서)
            hist, _, _ = np.histogram2d(
                segment_events[:, 0], segment_events[:, 1],
                bins=IMAGE_SHAPE,
                range=[[0, IMAGE_SHAPE[0]], [0, IMAGE_SHAPE[1]]]
            )
            
            # 최대값 기준 0~255 범위로 정규화하여 8비트 이미지 생성
            if hist.max() > 0:
                norm_img = (hist / hist.max() * 255).astype(np.uint8)
            else:
                norm_img = hist.astype(np.uint8)
            
            output_path = output_dir / f"{seq_path.stem}_segment_{i:04d}_binary_map.png"
            cv2.imwrite(str(output_path), norm_img)
    
    print(f"Saved binary map images for {seq_path.name}")

def main():
    args = parse_args()
    data_root = Path(args.data_root)
    source_dir = data_root / "top"
    
    # 모든 h5 파일에 대해 처리하되, "test" 폴더는 제외
    for split_dir in source_dir.iterdir():
        if not split_dir.is_dir():
            continue
        if split_dir.name.lower() == "test":
            continue
        for seq_path in split_dir.glob('*.h5'):
            process_sequence(seq_path)

if __name__ == '__main__':
    main()
