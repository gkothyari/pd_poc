/home/gkothyari/Documents/PD/.venv/bin/python pothole_detector.py predict \
  --source "/home/gkothyari/Documents/PD/test2.mp4" \
  --model "/home/gkothyari/Documents/PD/runs/detect/pothole_with_dashcam_opt/weights/best.pt" \
  --conf 0.50 --save



/home/gkothyari/Documents/PD/.venv/bin/python pothole_detector.py train \
  --data "/home/gkothyari/Documents/PD/data.yaml" \
  --model "yolov8n.pt" \
  --epochs 50 \
  --batch 4 \
  --device 0 \
  --imgsz 640 \
  --name pothole_full_gpu


/home/gkothyari/Documents/PD/.venv/bin/python pothole_detector.py val \
  --data "/home/gkothyari/Documents/PD/data.yaml" \
  --model "/home/gkothyari/Documents/PD/runs/detect/pothole_with_dashcam_opt/weights/best.pt"
