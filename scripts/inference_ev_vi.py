import argparse
import sys
import shutil
from pathlib import Path
import random
import json
from tqdm import tqdm
import yaml
import cv2
import numpy as np
from pycocotools import mask as mask_utils

# 프로젝트 상위 디렉토리 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

from src.detection import Sam2YoloDetector, Sam2YoloVideoDetector
from src.tracker import XMemSort
import src.utils as utils

# 재현을 위한 시드 고정
random.seed(0)

# 결과 저장 함수
def write_into_json_results(json_result, masks, ids, frame_idx, seq_id, instance_ids_list, num_frames):
    for mask, obj_id in zip(masks, ids):
        rle = mask_utils.encode(np.asfortranarray(mask))
        rle['counts'] = rle['counts'].decode('utf-8')
        if obj_id not in instance_ids_list:
            pred = {
                'video_id': seq_id,
                'score': 1,
                'instance_id': obj_id,
                'category_id': 1,
                'segmentations': [None] * num_frames
            }
            pred['segmentations'][frame_idx] = rle
            json_result.append(pred)
            instance_ids_list.append(obj_id)
        else:
            for pred in json_result:
                if pred['instance_id'] == obj_id:
                    pred['segmentations'][frame_idx] = rle
    return json_result, instance_ids_list

# e2vid 프레임 로드 함수
def load_e2vid_frames(e2vid_dir: Path):
    frames = {}
    if not e2vid_dir.exists():
        return frames
    for p in sorted(e2vid_dir.glob('*.png')):
        try:
            idx = int(p.stem)
        except ValueError:
            continue
        img = cv2.imread(str(p))
        if img is not None:
            frames[idx] = img
    return frames

# e2vid 전용 시퀀스 처리 함수
def process_sequence_e2vid_only(seq_name: str, split_dir: Path, output_dir: Path, detector, tracker, iou_th: float):
    seq_id = seq_name.replace('seq', '')
    e2vid_dir = split_dir / seq_name / 'e2vid'
    frames_dict = load_e2vid_frames(e2vid_dir)
    if not frames_dict:
        print(f"No e2vid frames for {seq_name}")
        return []

    # 1) 프레임 딕셔너리를 리스트로 변환
    fids = sorted(frames_dict.keys())
    all_frames = [frames_dict[fid] for fid in fids]
    num_frames = len(all_frames)

    # 2) detector.run에 전체 리스트 한 번만 호출
    print(f"Running detector on {num_frames} frames for {seq_name}")
    masks_seq, scores = detector.run(all_frames)
    if masks_seq is None:
        print("No detections for entire sequence.")
        return []

    # 3) 시각화 및 추적
    viz = utils.Visualizer(output_dir, save=True)
    instance_ids = []
    json_result = []

    for idx, fid in enumerate(fids):
        frame = all_frames[idx]
        masks = masks_seq[idx]  # 각 프레임의 마스크
        if masks is None or len(masks) == 0:
            viz.visualize_frame(frame)
            continue
        active = tracker.update(masks, frame)
        viz.visualize_predictions(frame, active['masks'], active['ids'])
        json_result, instance_ids = write_into_json_results(
            json_result, active['masks'], active['ids'], fid, seq_id, instance_ids, num_frames
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(json_result, f, indent=4)
    return json_result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/predict/combined_on_validation_sam2.yaml')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    with open(args.config) as cf:
        cfg = yaml.safe_load(cf)

    data_root = Path(cfg['common']['data_root'])
    split = cfg['common']['split']
    seq_ids = cfg['common'].get('sequence_ids')
    iou_th = cfg['common']['iou_threshold']
    output_folder = Path(cfg['output_dir']) / Path(args.config).stem
    output_folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, output_folder / 'config.yaml')

    detector = Sam2YoloVideoDetector(**cfg['e2vid_detector'], device=args.device)

    split_dir = data_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory {split_dir} not found")

    final_results = []
    for seq_path in sorted(split_dir.iterdir()):
        if not seq_path.is_dir():
            continue
        seq_name = seq_path.name  # e.g. 'seq03'
        sid = int(seq_name.replace('seq', ''))
        if seq_ids and sid not in seq_ids:
            continue
        out_dir = output_folder / split / seq_name
        tracker = XMemSort(**cfg['tracker'], device=args.device)
        print(f"Processing {seq_name}")
        results = process_sequence_e2vid_only(seq_name, split_dir, out_dir, detector, tracker, iou_th)
        final_results.extend(results)

    final_file = output_folder / 'final_results.json'
    with open(final_file, 'w') as f:
        json.dump(final_results, f, indent=4)
    print(f"Done. Total entries: {len(final_results)}")
