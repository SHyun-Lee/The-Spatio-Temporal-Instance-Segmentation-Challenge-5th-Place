import argparse
from pathlib import Path
import h5py
import numpy as np
import cv2
from tqdm import tqdm

# 고정 이미지 해상도
IMAGE_SHAPE = (720, 1280)

def parse_args():
    parser = argparse.ArgumentParser(
        description="이벤트 데이터를 흰 배경 위에 점으로 그려 저장 (지정한 시퀀스만 처리)")
    parser.add_argument(
        '--data_root',
        type=str,
        default='/home/coraldl/EV/MouseSIS/data/MouseSIS'
    )
    parser.add_argument(
        '--output_root',
        type=str,
        default='/home/coraldl/EV/MouseSIS/event'
    )
    return parser.parse_args()

def process_sequence(seq_path: Path, output_root: Path):
    """
    seq_path: .h5 파일 경로
    output_root: 결과를 저장할 루트 디렉토리
    """
    seq_name = seq_path.stem
    print(f"Processing {seq_name}")

    # 이 시퀀스 전용 서브디렉토리 생성
    seq_out_dir = output_root / seq_name
    seq_out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(seq_path, 'r') as f:
        # 이벤트 배열: [y, x, t, p] (마지막 인덱스 제외)
        events = np.stack(
            [f["y"][0:-1], f["x"][0:-1], f["t"][0:-1], f["p"][0:-1]],
            axis=-1
        ).astype(np.int32)

        # 프레임별 이벤트 인덱스
        ev_indices = f['img2event'][:]

        # 각 프레임 구간마다 흰 배경에 점 찍기
        for i, (start, end) in enumerate(
            tqdm(zip(ev_indices[:-1], ev_indices[1:]), total=len(ev_indices)-1, desc=seq_name)
        ):
            segment_events = events[start:end]

            # 흰 배경 캔버스 생성
            color_img = np.ones((IMAGE_SHAPE[0], IMAGE_SHAPE[1], 3), dtype=np.uint8) * 255

            # p==0인 이벤트: 파란 점
            blue_events = segment_events[segment_events[:, 3] == 0]
            if blue_events.size > 0:
                yb, xb = blue_events[:, 0], blue_events[:, 1]
                # 흰 배경 위에 파란색 (B=255, G=0, R=0)
                color_img[yb, xb, 0] = 255
                color_img[yb, xb, 1] = 0
                color_img[yb, xb, 2] = 0

            # p==1인 이벤트: 빨간 점
            red_events = segment_events[segment_events[:, 3] == 1]
            if red_events.size > 0:
                yr, xr = red_events[:, 0], red_events[:, 1]
                # 흰 배경 위에 빨간색 (B=0, G=0, R=255)
                color_img[yr, xr, 0] = 0
                color_img[yr, xr, 1] = 0
                color_img[yr, xr, 2] = 255

            # 파일명: {시퀀스명}_segment_{프레임번호:04d}_points.png
            output_path = seq_out_dir / f"{seq_name}_segment_{i:04d}_points.png"
            cv2.imwrite(str(output_path), color_img)

    print(f"Saved point event images for {seq_name}")


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # 처리할 시퀀스 이름 리스트
    target_seqs = ['seq14']

    # 데이터가 top 폴더 안에 train/valid 등이 있다고 가정
    source_dir = data_root / "top"

    for split_dir in source_dir.iterdir():
        if not split_dir.is_dir():
            continue
        if split_dir.name.lower() == "test":
            continue

        # 각 split (예: train, valid) 내부의 .h5 파일 순회
        for seq_path in split_dir.glob('*.h5'):
            if seq_path.stem in target_seqs:
                process_sequence(seq_path, output_root)


if __name__ == '__main__':
    main()
