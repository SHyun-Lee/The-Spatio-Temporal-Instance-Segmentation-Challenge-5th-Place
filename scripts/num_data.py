import glob, os
base = "/home/coraldl/EV/MouseSIS/data/train_kl"
total = 0
for seq_dir in sorted(os.listdir(base)):
    path = os.path.join(base, seq_dir, "e2vid", "*.png")
    cnt = len(glob.glob(path))
    print(f"{seq_dir}: {cnt}")
    total += cnt
print(f"---\nTotal train images: {total}")