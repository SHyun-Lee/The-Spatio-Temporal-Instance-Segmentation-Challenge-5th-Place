import os
import glob

root_dir = "/home/coraldl/EV/MouseSIS/data/MouseSIS/top"

# 재귀적으로 모든 .npy 파일 찾기
npy_files = glob.glob(os.path.join(root_dir, "**", "*.npy"), recursive=True)
print(f"Found {len(npy_files)} .npy files.")

for file_path in npy_files:
    try:
        print(f"Deleting {file_path} ...")
        os.remove(file_path)
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")
