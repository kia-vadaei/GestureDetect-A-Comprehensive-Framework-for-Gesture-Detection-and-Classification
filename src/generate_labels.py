import cv2
import mediapipe as mp
import os
import json
from tqdm import *

class GenerateLabels:
    def __init__(self, dataset_root = '/root/.cache/kagglehub/datasets/kianooshvadaei/castume-hand-gesture-dataset/versions/1'):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

        self.dataset_root = dataset_root

    def enhance_bbox_width(self, bbox, image_width, enhancement_factor=0.4):
        x_min, y_min, x_max, y_max = bbox
        current_width = x_max - x_min

        enhancement = current_width * enhancement_factor

        new_x_min = max(0, x_min - enhancement)
        new_x_max = min(image_width, x_max + enhancement)
        return [int(new_x_min), y_min, int(new_x_max), y_max]



    def process_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return None
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                h, w, _ = image.shape
                x_min = min([lm.x for lm in hand_landmarks.landmark]) * w
                y_min = min([lm.y for lm in hand_landmarks.landmark]) * h
                x_max = max([lm.x for lm in hand_landmarks.landmark]) * w
                y_max = max([lm.y for lm in hand_landmarks.landmark]) * h
                return [int(x_min), int(y_min), int(x_max), int(y_max)]
        return None
    
    def save_as_json(self, data):
        with open('labels.json', 'w') as f:
            json.dump(data, f, indent=4)

    def generate_labels(self, save_as_json = True):

        output_labels = []

        for class_name in os.listdir(self.dataset_root):
            class_dir = os.path.join(self.dataset_root, class_name)
            if os.path.isdir(class_dir):

                for image_name in tqdm(os.listdir(class_dir)):
                    image_path = os.path.join(class_dir, image_name)
                    bbox = self.process_image(image_path)
                    if bbox:

                        image = cv2.imread(image_path)
                        image_height, image_width, _ = image.shape

                        if class_name.lower() == 'collab':
                            bbox = self.enhance_bbox_width(bbox, image_width)
                        label = {
                            'image_path': image_path,
                            'class': class_name,
                            'bbox': bbox
                        }
                        output_labels.append(label)

        if save_as_json:
            save_as_json(output_labels)
    
        self.hands.close()

    def evaluate(self, image_path):
        bbox, r = self.process_image(image_path)
        image = cv2.imread(image_path)
        _, image_width, _ = image.shape
        bbox = self.enhance_bbox_width(bbox, image_width)
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.imshow(image)

