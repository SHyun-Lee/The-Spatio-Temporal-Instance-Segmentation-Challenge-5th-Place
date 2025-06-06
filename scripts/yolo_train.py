from ultralytics import YOLO

# 1) 모델 로드 (사전학습된 가중치)
model = YOLO("/home/coraldl/EV/MouseSIS/models/yolo_e2vid.pt")  

# 2) 하이퍼파라미터 튜닝 (예: learning rate, batch size 등)
hyp = {
    'lr0': 0.001,        # 초기 학습률
    'lrf': 0.1,         # 학습률 최종 스케줄 비율
    'momentum': 0.937,
    'weight_decay': 0.0005,
    # 그 외 hyp.yaml 에 정의된 파라미터들 덮어쓰기 가능
}

# 3) 학습 실행
train_results = model.train(
    data="/home/coraldl/EV/MouseSIS/configs/yolo/yolo_mouse.yaml",
    epochs=300,
    imgsz=(1280, 720),   # (width, height) 로 지정
    batch=32,             # 해상도가 커지면 메모리 이슈가 있으니 배치사이즈는 줄여주세요
    device=[0,1],
    task="detect",        # GPU 사용 시
    resume=False,
    rect=True,           # 가로세로 비율 유지한 rectangular training
    project="runs/fine_tune",
    name="yolo11n_ave",
    exist_ok=True
)

# 4) 검증
val_metrics = model.val(data="/home/coraldl/EV/MouseSIS/configs/yolo/yolo_mouse.yaml")

# 5) 추론 테스트
img_path = "/home/coraldl/EV/MouseSIS/data/test_kl/seq01/e2vid/00000032.png"
results = model(img_path, imgsz=640, conf=0.25)  # 신뢰도 임계값 0.25
results[0].show()  # 결과 시각화

# 6) ONNX 포맷으로 내보내기
onnx_path = model.export(format="onnx")
print(f"ONNX model saved to: {onnx_path}")
