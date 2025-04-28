import cv2
import torch
import os
from deep_sort_realtime.deepsort_tracker import DeepSort
import time

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.5

tracker = DeepSort(max_age=30)

prev_ids = set()
frame_count = 0
os.makedirs("output_frames", exist_ok=True)

cap = cv2.VideoCapture("input.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_video.mp4", fourcc, fps, (width, height))

start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()

    formatted = []
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        formatted.append(([x1, y1, x2, y2], conf, f"{int(cls)}"))

    tracks = tracker.update_tracks(formatted, frame=frame)
    current_ids = set()

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        current_ids.add(track_id)
        bbox = track.to_ltrb()
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if frame_count > 0:
        missing_ids = prev_ids - current_ids
        new_ids = current_ids - prev_ids

        for mid in missing_ids:
            y_pos = 50 + 20 * int(mid)
            cv2.putText(frame, f"Missing ID: {mid}", (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        for nid in new_ids:
            y_pos = 50 + 20 * int(nid)
            cv2.putText(frame, f"New ID: {nid}", (300, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    prev_ids = current_ids
    out.write(frame)

    if frame_count % 50 == 0:
        cv2.imwrite(f"output_frames/frame_{frame_count}.jpg", frame)

    print(f"[INFO] Processed frame {frame_count}")
    frame_count += 1

cap.release()
out.release()
end_time = time.time()
print(f"✅ Completed. Total frames: {frame_count}, Avg FPS: {frame_count / (end_time - start_time):.2f}")