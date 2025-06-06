import shutil
from pathlib import Path

# 원본(train_kl)과 대상(train_ave) 베이스 디렉토리 설정
BASE_K = Path("/home/coraldl/EV/MouseSIS/data/val_kl")
BASE_A = Path("/home/coraldl/EV/MouseSIS/data/val_ave")

# train_kl 아래의 모든 seq*/e2vid 디렉토리 순회
for seq_e2vid_dir in BASE_K.glob("seq*/e2vid"):
    seq_name = seq_e2vid_dir.parent.name  # 예: seq02
    src_dir = seq_e2vid_dir
    dst_dir = BASE_A / seq_name / "e2vid"

    if not src_dir.is_dir():
        print(f"[SKIP] {seq_name}: e2vid 폴더가 없음")
        continue

    # 대상 디렉토리가 없으면 생성
    dst_dir.mkdir(parents=True, exist_ok=True)

    # .txt 파일 복사
    copied = 0
    for txt_file in src_dir.glob("*.txt"):
        shutil.copy2(txt_file, dst_dir / txt_file.name)
        copied += 1

    print(f"[OK] {seq_name}: {copied}개의 .txt 파일 복사 완료")
