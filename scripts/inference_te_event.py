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

from src.detection import Sam2YoloDetector, SamYoloDetector
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
def process_sequence_e2vid_only(seq_name: str,
                                split_dir: Path,
                                output_dir: Path,
                                detector,
                                tracker,
                                iou_th: float,
                                yolo_out_dir: Path,
                                sam2_out_dir: Path):
    seq_id = seq_name.replace('seq', '')
    e2vid_dir = split_dir / seq_name / 'e2vid'
    frames = load_e2vid_frames(e2vid_dir)
    if not frames:
        print(f"No e2vid frames for {seq_name}")
        return []

    # 출력 폴더 생성
    viz = utils.Visualizer(output_dir, save=True)
    (yolo_out_dir / seq_name).mkdir(parents=True, exist_ok=True)
    (sam2_out_dir / seq_name).mkdir(parents=True, exist_ok=True)

    instance_ids = []
    json_result = []

    for fid in sorted(frames.keys()):
        frame = frames[fid]

        # --- 1) YOLO만 돌려서 박스 그린 이미지 저장 ---
        yolo_res = detector.detector(frame)[0]

        # 2) 시각화용 복사본
        yolo_vis = frame.copy()

        # 3) 박스와 score 가져오기
        boxes = yolo_res.boxes.xyxy.cpu().numpy().astype(int)   # (N,4)
        scores = yolo_res.boxes.conf.cpu().numpy()             # (N,)

        for (x1, y1, x2, y2), score in zip(boxes, scores):
            # 4) 사각형 그리기
            cv2.rectangle(yolo_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 5) 텍스트 (여기선 score를 소수 셋째 자리까지)
            label = f"IOU={score:.3f}"   # 실제 IOU라면 score 대신 IOU 변수를 쓰세요
            # 폰트, 크기, 두께 설정
            font       = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness  = 1
            # 텍스트를 박스의 왼쪽 위 (x1, y1-5) 에 찍기
            cv2.putText(
                yolo_vis,
                label,
                (x1, max(y1-5, 0)),   # y1-5 가 음수가 되지 않도록 안전하게 처리
                font,
                font_scale,
                (0, 255, 0),
                thickness,
                cv2.LINE_AA
            )

        # 6) 저장
        yolo_path = yolo_out_dir / seq_name / f"{fid:08d}.png"
        cv2.imwrite(str(yolo_path), yolo_vis)

        # --- 2) SAM2로 마스크 생성 ---
        masks, scores = detector.run(frame)
        if masks is None or len(masks)==0:
            # 빈 마스크 폴더에도 place-holder 저장할 수 있고, skip 도 가능합니다
            continue

        # 각 마스크를 별도 바이너리 이미지로 저장
        frame = frames[fid]  # H×W×3 배열
        h, w = frame.shape[:2]

        # 객체별로 색상 생성 (여기서는 시드로 idx 사용해 매 프레임마다 동일한 색)
        inst_viz = np.zeros((h, w, 3), dtype=np.uint8)
        for idx, mask in enumerate(masks):
            # 고정된 랜덤 색 (0~255)
            color = tuple(int(c) for c in np.random.RandomState(idx).randint(0, 256, 3))
            inst_viz[mask.astype(bool)] = color

        # 반투명 오버레이 합성
        alpha = 0.5
        overlay = cv2.addWeighted(frame, 1 - alpha, inst_viz, alpha, 0)

        # 저장
        out_path = sam2_out_dir / seq_name / f"{fid:08d}_overlay.png"
        cv2.imwrite(str(out_path), overlay)

        # --- 3) (기존) 트래킹·시각화·JSON 결과 저장 ---
        active = tracker.update(masks, frame)
        viz.visualize_predictions(frame, active['masks'], active['ids'])
        json_result, instance_ids = write_into_json_results(
            json_result, active['masks'], active['ids'],
            fid, seq_id, instance_ids, num_frames=len(frames)
        )

    # 최종 JSON
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(json_result, f, indent=4)
    return json_result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/predict/sis_challenge_baseline.yaml')
    parser.add_argument('--device', type=str, default='cuda:1')
    yolo_out = Path('/home/coraldl/EV/MouseSIS/detection/yolo')
    sam2_out = Path('/home/coraldl/EV/MouseSIS/detection/sam2')
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

    detector = Sam2YoloDetector(**cfg['e2vid_detector'], device=args.device)

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
        results = process_sequence_e2vid_only(seq_name, split_dir, out_dir, detector, tracker, iou_th, yolo_out_dir=yolo_out, sam2_out_dir=sam2_out)
        final_results.extend(results)

    final_file = output_folder / 'final_results.json'
    with open(final_file, 'w') as f:
        json.dump(final_results, f, indent=4)
    print(f"Done. Total entries: {len(final_results)}")
