import os

url = "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/models/YOLOv10x_gestures.pt"
output_file = "YOLOv10x_gestures.pt"  # You can specify a different file name or path here

os.system(f"wget {url} -O {output_file}")
