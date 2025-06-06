from ultralytics import YOLO
import ultralytics

# 모델 로드
model = YOLO('/home/coraldl/EV/MouseSIS/scripts/runs/fine_tune/yolo11n_ave/weights/best.pt')

# 전체 네트워크 구조 출력
print(model.model)  
# 혹은 모델 정보
print(model.yaml)   # 구성 파일(.yaml) 내용
print(ultralytics.__version__)
print(model.version) 