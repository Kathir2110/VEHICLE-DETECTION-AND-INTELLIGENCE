from ultralytics import YOLO
import cv2
import torch
import csv
from datetime import datetime

# ---------------- DEVICE ----------------
device = 0 if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Use larger model for better classification accuracy
model = YOLO("yolov8n.pt")   # better than yolov8m

# Vehicle class IDs (COCO)
vehicle_classes = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

cap = cv2.VideoCapture("traffic1.mp4")

# ---------------- COUNTERS ----------------
total_count = 0
counted_ids = set()

type_counts = {
    "car": 0,
    "bus": 0,
    "truck": 0,
    "motorcycle": 0
}

lane_counts = {
    "Lane 1": 0,
    "Lane 2": 0,
    "Lane 3": 0
}

# ---------------- CSV LOGGING ----------------
csv_file = open("traffic_log.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp", "Vehicle_ID", "Vehicle_Type", "Lane"])

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    # Draw lane boundaries
    cv2.line(frame, (213, 0), (213, 480), (255, 0, 0), 2)
    cv2.line(frame, (426, 0), (426, 480), (255, 0, 0), 2)

    # Higher confidence threshold for better classification
    results = model.track(frame, persist=True, device=device, conf=0.6)

    active_vehicle_count = 0

    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()

        for box, track_id, cls_id in zip(boxes, track_ids, class_ids):

            cls_id = int(cls_id)
            track_id = int(track_id)

            if cls_id not in vehicle_classes:
                continue

            vehicle_type = vehicle_classes[cls_id]
            active_vehicle_count += 1

            x1, y1, x2, y2 = map(int, box)
            center_x = int((x1 + x2) / 2)

            # -------- COUNT NEW VEHICLE --------
            if track_id not in counted_ids:
                counted_ids.add(track_id)
                total_count += 1
                type_counts[vehicle_type] += 1

                # Lane classification
                if center_x < 213:
                    lane = "Lane 1"
                elif center_x < 426:
                    lane = "Lane 2"
                else:
                    lane = "Lane 3"

                lane_counts[lane] += 1

                # Log to CSV
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                csv_writer.writerow([timestamp, track_id, vehicle_type, lane])

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{vehicle_type} | ID: {track_id}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2)

    # -------- DENSITY --------
    if active_vehicle_count <= 5:
        density = "LOW"
        density_color = (0, 255, 255)
    elif active_vehicle_count <= 15:
        density = "MEDIUM"
        density_color = (0, 255, 255)
    else:
        density = "HIGH"
        density_color = (0, 255, 255)

    # -------- DISPLAY --------
    cv2.putText(frame, f"Total Vehicles: {total_count}",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.putText(frame, f"Cars: {type_counts['car']}",
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(frame, f"Buses: {type_counts['bus']}",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.putText(frame, f"Trucks: {type_counts['truck']}",
                (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.putText(frame, f"Motorcycles: {type_counts['motorcycle']}",
                (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    cv2.putText(frame, f"Lane1: {lane_counts['Lane 1']}",
                (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame, f"Lane2: {lane_counts['Lane 2']}",
                (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame, f"Lane3: {lane_counts['Lane 3']}",
                (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame, f"Density: {density}",
                (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, density_color, 2)

    cv2.imshow("Smart Traffic Intelligence System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
csv_file.close()
cv2.destroyAllWindows()