/home/gkothyari/Documents/PD/.venv/bin/python pothole_detector.py predict \
  --source "/home/gkothyari/Documents/PD/pothole_kaggle_dataset/valid/images/pothole_1539.jpg" \
  --model "/home/gkothyari/Documents/PD/runs/detect/pothole_full_gpu/weights/best.pt" \
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
  --model "/home/gkothyari/Documents/PD/runs/detect/pothole_full_gpu/weights/best.pt"