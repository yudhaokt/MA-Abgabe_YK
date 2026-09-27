from ultralytics import YOLO

MODEL_PATH = r"ModelTraining/RQD-remapped_YOLO11x/RQD-remapped_train/weights/best.pt"
OUTPUT_PATH = r"ModelTraining/RQD-remapped_YOLO11x/RQD-remapped_train/weights/best_renamed.pt"

model = YOLO(MODEL_PATH)

print("Old class names:")
print(model.names)

model.model.names[3] = "3_bad_asphalt"
model.model.names[4] = "4_bad_others"

model.save(OUTPUT_PATH)

print("\nNew class names:")
print(model.names)